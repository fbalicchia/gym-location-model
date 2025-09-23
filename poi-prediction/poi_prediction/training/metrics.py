"""
Evaluation metrics for POI prediction
"""

import torch
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, mean_squared_error
import warnings
warnings.filterwarnings('ignore')

from ..data.data_structures import ModelOutput, ModelTargets


class POIMetrics:
    """Comprehensive metrics for POI prediction evaluation"""
    
    def __init__(self, num_categories: int):
        self.num_categories = num_categories
    
    def compute_metrics(self, 
                       predictions: List[ModelOutput], 
                       targets: List[ModelTargets]) -> Dict[str, float]:
        """
        Compute comprehensive metrics for POI prediction
        
        Args:
            predictions: List of model outputs
            targets: List of ground truth targets
            
        Returns:
            Dictionary of computed metrics
        """
        metrics = {}
        
        # Extract and concatenate all predictions and targets
        all_poi_scores = []
        all_category_logits = []
        all_duration_preds = []
        
        all_poi_targets = []
        all_category_targets = []
        all_duration_targets = []
        all_poi_indices = []
        
        for pred, target in zip(predictions, targets):
            # POI ranking metrics
            if pred.poi_scores is not None and target.target_poi_indices is not None:
                all_poi_scores.append(pred.poi_scores)
                all_poi_indices.append(target.target_poi_indices)
            
            # Category prediction metrics
            if pred.category_logits is not None:
                # Get valid (non-padded) positions
                valid_mask = target.next_categories != -100
                if valid_mask.any():
                    valid_logits = pred.category_logits[valid_mask]
                    valid_targets = target.next_categories[valid_mask]
                    
                    all_category_logits.append(valid_logits)
                    all_category_targets.append(valid_targets)
            
            # Duration prediction metrics
            if pred.duration_predictions is not None:
                valid_mask = target.next_durations != -1.0
                if valid_mask.any():
                    valid_preds = pred.duration_predictions.squeeze(-1)[valid_mask]
                    valid_targets = target.next_durations[valid_mask]
                    
                    all_duration_preds.append(valid_preds)
                    all_duration_targets.append(valid_targets)
        
        # Compute POI ranking metrics
        if all_poi_scores:
            poi_metrics = self._compute_ranking_metrics(all_poi_scores, all_poi_indices)
            metrics.update(poi_metrics)
        
        # Compute category prediction metrics
        if all_category_logits:
            category_metrics = self._compute_classification_metrics(
                all_category_logits, all_category_targets
            )
            metrics.update(category_metrics)
        
        # Compute duration prediction metrics
        if all_duration_preds:
            duration_metrics = self._compute_regression_metrics(
                all_duration_preds, all_duration_targets
            )
            metrics.update(duration_metrics)
        
        return metrics
    
    def _compute_ranking_metrics(self, 
                                poi_scores: List[torch.Tensor], 
                                target_indices: List[torch.Tensor]) -> Dict[str, float]:
        """Compute POI ranking metrics"""
        metrics = {}
        
        # Concatenate all batches
        all_scores = torch.cat(poi_scores, dim=0)  # (total_samples, num_candidates)
        all_targets = torch.cat(target_indices, dim=0)  # (total_samples,)
        
        # Convert to numpy for easier computation
        scores_np = all_scores.detach().cpu().numpy()
        targets_np = all_targets.detach().cpu().numpy()
        
        # Compute ranking metrics for different k values
        k_values = [1, 3, 5, 10]
        
        for k in k_values:
            if k <= scores_np.shape[1]:
                # Get top-k predictions
                top_k_indices = np.argsort(scores_np, axis=1)[:, -k:]
                
                # Compute Hit Rate@k (HR@k)
                hits = np.array([target in top_k for target, top_k in zip(targets_np, top_k_indices)])
                hr_k = np.mean(hits)
                metrics[f'hr_at_{k}'] = hr_k
                
                # Compute Mean Reciprocal Rank for top-k (MRR@k)
                reciprocal_ranks = []
                for target, top_k in zip(targets_np, top_k_indices):
                    if target in top_k:
                        rank = np.where(top_k[::-1] == target)[0][0] + 1  # Reverse for descending order
                        reciprocal_ranks.append(1.0 / rank)
                    else:
                        reciprocal_ranks.append(0.0)
                
                mrr_k = np.mean(reciprocal_ranks)
                metrics[f'mrr_at_{k}'] = mrr_k
        
        # Compute NDCG@10 (Normalized Discounted Cumulative Gain)
        if scores_np.shape[1] >= 10:
            ndcg_10 = self._compute_ndcg(scores_np, targets_np, k=10)
            metrics['ndcg_at_10'] = ndcg_10
        
        return metrics
    
    def _compute_classification_metrics(self, 
                                      logits: List[torch.Tensor], 
                                      targets: List[torch.Tensor]) -> Dict[str, float]:
        """Compute category classification metrics"""
        metrics = {}
        
        # Concatenate all predictions and targets
        all_logits = torch.cat(logits, dim=0)  # (total_samples, num_categories)
        all_targets = torch.cat(targets, dim=0)  # (total_samples,)
        
        # Get predictions
        predictions = torch.argmax(all_logits, dim=-1)
        
        # Convert to numpy
        preds_np = predictions.detach().cpu().numpy()
        targets_np = all_targets.detach().cpu().numpy()
        
        # Compute accuracy
        accuracy = accuracy_score(targets_np, preds_np)
        metrics['category_accuracy'] = accuracy
        
        # Compute precision, recall, F1 (macro average)
        try:
            precision, recall, f1, _ = precision_recall_fscore_support(
                targets_np, preds_np, average='macro', zero_division=0
            )
            metrics['category_precision'] = precision
            metrics['category_recall'] = recall
            metrics['category_f1'] = f1
        except:
            metrics['category_precision'] = 0.0
            metrics['category_recall'] = 0.0
            metrics['category_f1'] = 0.0
        
        # Top-k accuracy for categories
        if all_logits.shape[1] >= 3:
            top_3_preds = torch.topk(all_logits, k=3, dim=-1)[1]
            top_3_accuracy = torch.mean(
                torch.any(top_3_preds == all_targets.unsqueeze(-1), dim=-1).float()
            ).item()
            metrics['category_top3_accuracy'] = top_3_accuracy
        
        return metrics
    
    def _compute_regression_metrics(self, 
                                  predictions: List[torch.Tensor], 
                                  targets: List[torch.Tensor]) -> Dict[str, float]:
        """Compute duration regression metrics"""
        metrics = {}
        
        # Concatenate all predictions and targets
        all_preds = torch.cat(predictions, dim=0)  # (total_samples,)
        all_targets = torch.cat(targets, dim=0)  # (total_samples,)
        
        # Convert to numpy
        preds_np = all_preds.detach().cpu().numpy()
        targets_np = all_targets.detach().cpu().numpy()
        
        # Mean Squared Error
        mse = mean_squared_error(targets_np, preds_np)
        metrics['duration_mse'] = mse
        
        # Root Mean Squared Error
        rmse = np.sqrt(mse)
        metrics['duration_rmse'] = rmse
        
        # Mean Absolute Error
        mae = np.mean(np.abs(preds_np - targets_np))
        metrics['duration_mae'] = mae
        
        # Mean Absolute Percentage Error (avoid division by zero)
        valid_mask = targets_np != 0
        if np.any(valid_mask):
            mape = np.mean(np.abs((targets_np[valid_mask] - preds_np[valid_mask]) / targets_np[valid_mask])) * 100
            metrics['duration_mape'] = mape
        else:
            metrics['duration_mape'] = 0.0
        
        # R-squared (coefficient of determination)
        ss_res = np.sum((targets_np - preds_np) ** 2)
        ss_tot = np.sum((targets_np - np.mean(targets_np)) ** 2)
        
        if ss_tot != 0:
            r2 = 1 - (ss_res / ss_tot)
            metrics['duration_r2'] = r2
        else:
            metrics['duration_r2'] = 0.0
        
        return metrics
    
    def _compute_ndcg(self, scores: np.ndarray, targets: np.ndarray, k: int = 10) -> float:
        """Compute Normalized Discounted Cumulative Gain@k"""
        ndcg_scores = []
        
        for score_row, target in zip(scores, targets):
            # Get top-k indices (sorted by score descending)
            top_k_indices = np.argsort(score_row)[::-1][:k]
            
            # Create relevance vector (1 for correct item, 0 for others)
            relevance = np.zeros(len(top_k_indices))
            if target in top_k_indices:
                target_pos = np.where(top_k_indices == target)[0][0]
                relevance[target_pos] = 1
            
            # Compute DCG
            dcg = 0
            for i, rel in enumerate(relevance):
                if rel > 0:
                    dcg += rel / np.log2(i + 2)  # i+2 because positions start from 1
            
            # Compute ideal DCG (best possible ranking)
            ideal_relevance = np.sort(relevance)[::-1]
            idcg = 0
            for i, rel in enumerate(ideal_relevance):
                if rel > 0:
                    idcg += rel / np.log2(i + 2)
            
            # Compute NDCG
            if idcg > 0:
                ndcg_scores.append(dcg / idcg)
            else:
                ndcg_scores.append(0.0)
        
        return np.mean(ndcg_scores)
    
    def compute_attention_analysis(self, 
                                 model_outputs: List[ModelOutput]) -> Dict[str, float]:
        """Analyze attention patterns"""
        metrics = {}
        
        attention_entropies = []
        max_attention_weights = []
        
        for output in model_outputs:
            if output.attention_weights:
                for layer_attention in output.attention_weights:
                    if 'self_attention' in layer_attention and layer_attention['self_attention'] is not None:
                        attn_weights = layer_attention['self_attention']
                        # attn_weights: (batch_size, num_heads, seq_len, seq_len)
                        
                        # Compute attention entropy (measure of attention dispersion)
                        attn_probs = F.softmax(attn_weights, dim=-1)
                        entropy = -torch.sum(attn_probs * torch.log(attn_probs + 1e-8), dim=-1)
                        attention_entropies.extend(entropy.flatten().tolist())
                        
                        # Compute max attention weights (measure of attention concentration)
                        max_weights = torch.max(attn_probs, dim=-1)[0]
                        max_attention_weights.extend(max_weights.flatten().tolist())
        
        if attention_entropies:
            metrics['avg_attention_entropy'] = np.mean(attention_entropies)
            metrics['avg_max_attention_weight'] = np.mean(max_attention_weights)
        
        return metrics


def evaluate_model_predictions(model, data_loader, device, top_k: int = 10) -> Dict[str, np.ndarray]:
    """
    Generate predictions for analysis and visualization
    
    Args:
        model: Trained POI transformer model
        data_loader: DataLoader for evaluation
        device: Device to run evaluation on
        top_k: Number of top predictions to return
        
    Returns:
        Dictionary with detailed predictions
    """
    model.eval()
    
    all_predictions = {
        'poi_predictions': [],
        'category_predictions': [],
        'duration_predictions': [],
        'ground_truth_poi': [],
        'ground_truth_category': [],
        'ground_truth_duration': [],
        'user_ids': [],
        'sequence_representations': []
    }
    
    with torch.no_grad():
        for batch_data, targets in data_loader:
            batch_data = batch_data.to(device)
            targets = targets.to(device)
            
            # Get model predictions
            predictions = model.predict_next_poi(batch_data, top_k=top_k)
            
            # Store predictions
            if 'top_poi_indices' in predictions:
                all_predictions['poi_predictions'].extend(
                    predictions['top_poi_indices'].cpu().numpy()
                )
            
            if 'top_category_indices' in predictions:
                all_predictions['category_predictions'].extend(
                    predictions['top_category_indices'].cpu().numpy()
                )
            
            if 'predicted_durations' in predictions:
                all_predictions['duration_predictions'].extend(
                    predictions['predicted_durations'].cpu().numpy()
                )
            
            # Store ground truth
            if targets.target_poi_indices is not None:
                all_predictions['ground_truth_poi'].extend(
                    targets.target_poi_indices.cpu().numpy()
                )
            
            # Get last position targets for categories and durations
            last_pos_categories = targets.next_categories[:, -1]
            last_pos_durations = targets.next_durations[:, -1]
            
            valid_cat_mask = last_pos_categories != -100
            valid_dur_mask = last_pos_durations != -1.0
            
            if valid_cat_mask.any():
                all_predictions['ground_truth_category'].extend(
                    last_pos_categories[valid_cat_mask].cpu().numpy()
                )
            
            if valid_dur_mask.any():
                all_predictions['ground_truth_duration'].extend(
                    last_pos_durations[valid_dur_mask].cpu().numpy()
                )
            
            # Store user IDs and sequence representations
            all_predictions['user_ids'].extend(batch_data.user_ids.cpu().numpy())
    
    # Convert lists to numpy arrays
    for key in all_predictions:
        if all_predictions[key]:
            all_predictions[key] = np.array(all_predictions[key])
    
    return all_predictions
