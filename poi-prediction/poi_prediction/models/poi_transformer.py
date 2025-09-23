"""
POI Transformer with Cross-Attention for Next POI Prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, List, Tuple
import math

from ..config.model_config import ModelConfig
from ..data.data_structures import BatchData, ModelOutput, ModelTargets
from .embeddings import MultiModalEmbedding, TemporalPositionalEncoding, UserContextEncoder
from .attention import MultiScaleCrossAttention, CrossAttentionBlock


class TransformerBlock(nn.Module):
    """Enhanced Transformer block with self-attention and cross-attention"""
    
    def __init__(self, config: ModelConfig, use_cross_attention: bool = True):
        super().__init__()
        
        self.config = config
        self.use_cross_attention = use_cross_attention
        d_model = config.d_model
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model, config.nhead, dropout=config.dropout, batch_first=True
        )
        
        # Cross-attention (multi-scale)
        if use_cross_attention:
            self.cross_attention = MultiScaleCrossAttention(config)
        
        # Feed forward network
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, config.dim_feedforward),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.dim_feedforward, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model) if use_cross_attention else None
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
    def forward(self, 
                x: torch.Tensor,
                trajectory_coords: torch.Tensor,
                user_context: Optional[torch.Tensor] = None,
                temporal_context: Optional[torch.Tensor] = None,
                candidate_coords: Optional[torch.Tensor] = None,
                attention_mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass through transformer block
        
        Args:
            x: (batch_size, seq_len, d_model)
            trajectory_coords: (batch_size, seq_len, 2)
            user_context: (batch_size, 1, d_model)
            temporal_context: (batch_size, time_len, d_model)
            candidate_coords: (batch_size, num_candidates, 2)
            attention_mask: (batch_size, seq_len) or (seq_len, seq_len)
            return_attention: bool
            
        Returns:
            Dictionary with outputs and attention weights
        """
        # Self-attention with residual connection
        attn_output, self_attn_weights = self.self_attn(
            x, x, x, 
            attn_mask=attention_mask, 
            need_weights=return_attention
        )
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross-attention (if enabled)
        cross_attn_outputs = None
        if self.use_cross_attention:
            cross_attn_outputs = self.cross_attention(
                x, trajectory_coords, user_context, temporal_context, candidate_coords
            )
            x = self.norm2(x + self.dropout(cross_attn_outputs['output'] - x))
        
        # Feed forward with residual connection
        ff_output = self.feed_forward(x)
        if self.use_cross_attention:
            x = self.norm3(x + self.dropout(ff_output))
        else:
            x = self.norm2(x + self.dropout(ff_output))
        
        # Prepare output
        outputs = {'hidden': x}
        
        if return_attention:
            outputs['self_attention'] = self_attn_weights
            if cross_attn_outputs:
                outputs.update(cross_attn_outputs)
        
        return outputs


class POITransformerWithCrossAttention(nn.Module):
    """
    Enhanced POI Transformer with Cross-Attention for next POI recommendation
    
    High-level Idea:
    The class combines:
    • Multi-modal embeddings (POI, category, time, location, region, etc.)
    • User context
    • Geographic attention  
    • Transformer blocks (some with cross-attention, some without)
    • Output heads for:
        • Next POI ranking
        • Next category prediction
        • Time-to-next POI regression
    """
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        # ============ Embedding Layers ============
        # Multi-modal embedding for trajectory features
        self.embedding = MultiModalEmbedding(config)
        
        # Temporal positional encoding
        self.pos_encoding = TemporalPositionalEncoding(
            config.d_model, 
            max_len=config.max_sequence_length
        )
        
        # User context encoder
        self.user_encoder = UserContextEncoder(config)
        
        # ============ Transformer Architecture ============
        # Transformer blocks with alternating cross-attention
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                config, 
                use_cross_attention=(i % config.cross_attention_every_n_layers == 0)
            ) for i in range(config.num_layers)
        ])
        
        # ============ Output Heads ============
        # Candidate POI encoder for scoring
        self.candidate_encoder = nn.Sequential(
            nn.Linear(config.poi_embed_dim, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.ReLU(),
            nn.Dropout(config.dropout)
        )
        
        # POI ranking head (scores candidate POIs)
        self.poi_scorer = nn.Sequential(
            nn.Linear(config.d_model * 2, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model, 1)
        )
        
        # Category prediction head
        self.category_prediction_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, config.num_categories)
        )
        
        # Time-to-next POI regression head
        self.time_regression_head = nn.Sequential(
            nn.Linear(config.d_model, config.d_model // 2),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 2, 1),
            nn.Softplus()  # Ensure positive output
        )
        
        # Dropout
        self.dropout = nn.Dropout(config.dropout)
        
        # Initialize model
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                nn.init.constant_(module.bias, 0)
                nn.init.constant_(module.weight, 1.0)
    
    def forward(self, batch_data: BatchData, return_attention: bool = False) -> ModelOutput:
        """
        Forward pass through the POI Transformer
        
        Args:
            batch_data: BatchData object containing all input features
            return_attention: Whether to return attention weights
            
        Returns:
            ModelOutput object with predictions and representations
        """
        batch_size, seq_len = batch_data.poi_ids.shape
        
        # ============ Embedding Phase ============
        # Get multi-modal embeddings for trajectory
        trajectory_emb = self.embedding(
            batch_data.poi_ids,
            batch_data.categories, 
            batch_data.hours,
            batch_data.days,
            batch_data.seasons,
            batch_data.regions,
            batch_data.coordinates,
            batch_data.durations
        )
        
        # Add positional encoding
        trajectory_emb = self.pos_encoding(
            trajectory_emb, 
            batch_data.hours, 
            batch_data.days
        )
        trajectory_emb = self.dropout(trajectory_emb)
        
        # ============ Context Encoding ============
        # Encode user context
        user_context = self.user_encoder(
            batch_data.user_ids, 
            batch_data.user_preferences
        )
        
        # ============ Transformer Processing ============
        # Create attention mask if using causal attention
        attention_mask = None
        if self.config.use_causal_mask:
            attention_mask = self._generate_causal_mask(seq_len).to(trajectory_emb.device)
        
        # Store attention weights
        all_attention_weights = []
        
        # Apply transformer blocks
        x = trajectory_emb
        for i, block in enumerate(self.transformer_blocks):
            block_outputs = block(
                x,
                trajectory_coords=batch_data.coordinates,
                user_context=user_context,
                temporal_context=None,  # Can add temporal context here
                candidate_coords=batch_data.candidate_coordinates,
                attention_mask=attention_mask,
                return_attention=return_attention
            )
            
            x = block_outputs['hidden']
            
            if return_attention:
                all_attention_weights.append(block_outputs)
        
        # ============ Output Generation ============
        # Get sequence representation
        if self.config.pooling == "last":
            sequence_repr = x[:, -1, :]  # Last position
        else:
            # Mean pooling with attention mask
            if batch_data.attention_mask is not None:
                mask = batch_data.attention_mask.unsqueeze(-1).float()
                sequence_repr = (x * mask).sum(dim=1) / mask.sum(dim=1)
            else:
                sequence_repr = x.mean(dim=1)
        
        # ============ Prediction Heads ============
        # POI candidate scoring
        poi_scores = None
        if batch_data.candidate_poi_ids is not None:
            poi_scores = self._score_candidates(
                sequence_repr, 
                batch_data.candidate_poi_ids
            )
        
        # Category prediction (for all positions)
        category_logits = self.category_prediction_head(x)
        
        # Time-to-next prediction (for all positions)
        duration_predictions = self.time_regression_head(x)
        
        # ============ Create Output ============
        output = ModelOutput(
            poi_scores=poi_scores,
            category_logits=category_logits,
            duration_predictions=duration_predictions,
            sequence_representation=sequence_repr,
            hidden_states=x
        )
        
        if return_attention:
            output.attention_weights = all_attention_weights
        
        return output
    
    def _score_candidates(self, 
                         sequence_repr: torch.Tensor, 
                         candidate_poi_ids: torch.Tensor) -> torch.Tensor:
        """
        Score candidate POIs for ranking
        
        Args:
            sequence_repr: (batch_size, d_model)
            candidate_poi_ids: (batch_size, num_candidates)
            
        Returns:
            scores: (batch_size, num_candidates)
        """
        batch_size, num_candidates = candidate_poi_ids.shape
        
        # Get candidate embeddings
        candidate_emb = self.embedding.poi_embedding(candidate_poi_ids)
        candidate_emb = self.candidate_encoder(candidate_emb)
        # (batch_size, num_candidates, d_model)
        
        # Expand sequence representation
        sequence_repr_expanded = sequence_repr.unsqueeze(1).expand(-1, num_candidates, -1)
        # (batch_size, num_candidates, d_model)
        
        # Concatenate sequence and candidate representations
        combined = torch.cat([sequence_repr_expanded, candidate_emb], dim=-1)
        # (batch_size, num_candidates, d_model * 2)
        
        # Score candidates
        scores = self.poi_scorer(combined).squeeze(-1)
        # (batch_size, num_candidates)
        
        return scores
    
    def _generate_causal_mask(self, size: int) -> torch.Tensor:
        """Generate causal mask for autoregressive attention"""
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def compute_loss(self, 
                    outputs: ModelOutput, 
                    targets: ModelTargets) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss
        
        Args:
            outputs: Model predictions
            targets: Ground truth targets
            
        Returns:
            Dictionary of losses
        """
        losses = {}
        
        # POI ranking loss (if candidates provided)
        if outputs.poi_scores is not None and targets.target_poi_indices is not None:
            poi_loss = F.cross_entropy(outputs.poi_scores, targets.target_poi_indices)
            losses['poi_loss'] = poi_loss
        
        # Category prediction loss
        if outputs.category_logits is not None:
            # Reshape for loss computation
            cat_logits = outputs.category_logits.reshape(-1, self.config.num_categories)
            cat_targets = targets.next_categories.reshape(-1)
            
            # Mask out padding tokens if needed
            valid_mask = cat_targets != -100  # Assuming -100 is padding token
            if valid_mask.sum() > 0:
                cat_loss = F.cross_entropy(cat_logits[valid_mask], cat_targets[valid_mask])
                losses['category_loss'] = cat_loss
        
        # Duration prediction loss
        if outputs.duration_predictions is not None:
            duration_pred = outputs.duration_predictions.squeeze(-1)
            duration_targets = targets.next_durations
            
            # Mask out padding if needed
            valid_mask = duration_targets != -1  # Assuming -1 is padding
            if valid_mask.sum() > 0:
                time_loss = F.mse_loss(
                    duration_pred[valid_mask], 
                    duration_targets[valid_mask]
                )
                losses['time_loss'] = time_loss
        
        # Compute weighted total loss
        total_loss = 0.0
        if 'poi_loss' in losses:
            total_loss += self.config.poi_loss_weight * losses['poi_loss']
        if 'category_loss' in losses:
            total_loss += self.config.category_loss_weight * losses['category_loss']
        if 'time_loss' in losses:
            total_loss += self.config.time_loss_weight * losses['time_loss']
        
        losses['total_loss'] = total_loss
        
        return losses
    
    def predict_next_poi(self, 
                        batch_data: BatchData,
                        top_k: int = 10) -> Dict[str, torch.Tensor]:
        """
        Predict next POI for given trajectories
        
        Args:
            batch_data: Input batch data
            top_k: Number of top predictions to return
            
        Returns:
            Dictionary with predictions
        """
        self.eval()
        
        with torch.no_grad():
            # Forward pass
            outputs = self.forward(batch_data)
            
            predictions = {}
            
            # POI ranking predictions
            if outputs.poi_scores is not None:
                poi_probs = F.softmax(outputs.poi_scores, dim=-1)
                top_poi_scores, top_poi_indices = torch.topk(poi_probs, k=top_k, dim=-1)
                predictions['top_poi_scores'] = top_poi_scores
                predictions['top_poi_indices'] = top_poi_indices
            
            # Category predictions (last position)
            if outputs.category_logits is not None:
                cat_probs = F.softmax(outputs.category_logits[:, -1, :], dim=-1)
                top_cat_scores, top_cat_indices = torch.topk(cat_probs, k=min(top_k, self.config.num_categories), dim=-1)
                predictions['top_category_scores'] = top_cat_scores
                predictions['top_category_indices'] = top_cat_indices
            
            # Duration predictions (last position)
            if outputs.duration_predictions is not None:
                predictions['predicted_durations'] = outputs.duration_predictions[:, -1, 0]
        
        return predictions
