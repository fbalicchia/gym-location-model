"""
Main POI Transformer architecture
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, Tuple

from .embeddings import MultiModalEmbedding
from .positional_encoding import TemporalPositionalEncoding


class TransformerBlock(nn.Module):
    """Custom Transformer block with multi-head attention"""
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, dropout: float):
        super().__init__()
        
        # Multi-head attention
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        
        # Feed forward
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None, 
                return_attention: bool = False) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        # Self-attention with residual connection
        attn_output, attn_weights = self.self_attn(
            x, x, x, attn_mask=mask, need_weights=return_attention
        )
        x = self.norm1(x + self.dropout(attn_output))
        
        # Feed forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        if return_attention:
            return x, attn_weights
        return x, None


class POITransformer(nn.Module):
    """Advanced Transformer for POI prediction with multiple heads"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Multi-modal embedding (using the improved version)
        self.embedding = MultiModalEmbedding(config)
        
        # Temporal positional encoding
        self.pos_encoding = TemporalPositionalEncoding(config['d_model'])
        
        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_model=config['d_model'],
                nhead=config['nhead'],
                dim_feedforward=config['dim_feedforward'],
                dropout=config['dropout']
            ) for _ in range(config['num_layers'])
        ])
        
        # Output heads
        self.poi_prediction_head = nn.Linear(config['d_model'], config['num_pois'])
        self.category_prediction_head = nn.Linear(config['d_model'], config['num_categories'])
        self.time_regression_head = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model'] // 2),
            nn.ReLU(),
            nn.Linear(config['d_model'] // 2, 1),
            nn.Softplus()  # Ensure positive output
        )
        
        # Dropout
        self.dropout = nn.Dropout(config['dropout'])
        
    def forward(self, batch_data: Dict, return_attention: bool = False) -> Dict:
        # Unpack batch data
        poi_ids = batch_data['poi_ids']
        categories = batch_data['categories']
        timestamps = batch_data['timestamps']
        days = batch_data['days']
        seasons = batch_data['seasons']
        regions = batch_data['regions']
        coords = batch_data['coords']
        
        batch_size, seq_len = poi_ids.shape
        
        # Get embeddings using the improved multi-modal embedding
        x = self.embedding(poi_ids, categories, timestamps, days, seasons, regions, coords)
        
        # Add positional encoding
        x = self.pos_encoding(x, timestamps, days)
        
        # Apply dropout
        x = self.dropout(x)
        
        # Create causal mask
        mask = self.generate_causal_mask(seq_len).to(x.device)
        
        # Store attention weights if requested
        attention_weights = []
        
        # Apply transformer blocks
        for block in self.transformer_blocks:
            x, attn = block(x, mask, return_attention=return_attention)
            if return_attention:
                attention_weights.append(attn)
        
        # Apply output heads
        poi_logits = self.poi_prediction_head(x)
        category_logits = self.category_prediction_head(x)
        time_to_next = self.time_regression_head(x)
        
        outputs = {
            'poi_logits': poi_logits,
            'category_logits': category_logits,
            'time_to_next': time_to_next
        }
        
        if return_attention:
            outputs['attention_weights'] = attention_weights
        
        return outputs
    
    def generate_causal_mask(self, size: int) -> torch.Tensor:
        """Generate causal mask"""
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))


"""
Enhanced POI Transformer with Cross-Attention for Next POI Recommendation
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List
import math

from .embeddings import MultiModalEmbedding
from .positional_encoding import TemporalPositionalEncoding


class CrossAttentionBlock(nn.Module):
    """Cross-attention block for attending between sequences and context"""
    
    def __init__(self, d_model: int, nhead: int, dropout: float):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        attn_output, attn_weights = self.cross_attn(query, key, value, attn_mask=mask)
        output = self.norm(query + self.dropout(attn_output))
        return output, attn_weights


class TransformerBlock(nn.Module):
    """Enhanced Transformer block with self-attention and cross-attention"""
    
    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, 
                 dropout: float, use_cross_attention: bool = True):
        super().__init__()
        
        # Self-attention
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        
        # Cross-attention (for user context and geographical context)
        self.use_cross_attention = use_cross_attention
        if use_cross_attention:
            self.user_cross_attn = CrossAttentionBlock(d_model, nhead, dropout)
            self.geo_cross_attn = CrossAttentionBlock(d_model, nhead, dropout)
        
        # Feed forward
        self.feed_forward = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.GELU(),  # GELU often works better than ReLU
            nn.Dropout(dropout),
            nn.Linear(dim_feedforward, d_model)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, 
                user_context: Optional[torch.Tensor] = None,
                geo_context: Optional[torch.Tensor] = None,
                mask: Optional[torch.Tensor] = None,
                return_attention: bool = False) -> Dict[str, torch.Tensor]:
        
        # Self-attention with residual connection
        attn_output, self_attn_weights = self.self_attn(
            x, x, x, attn_mask=mask, need_weights=return_attention
        )
        x = self.norm1(x + self.dropout(attn_output))
        
        # Cross-attention with user context
        user_attn_weights = None
        geo_attn_weights = None
        
        if self.use_cross_attention and user_context is not None:
            x, user_attn_weights = self.user_cross_attn(x, user_context, user_context)
            
        if self.use_cross_attention and geo_context is not None:
            x, geo_attn_weights = self.geo_cross_attn(x, geo_context, geo_context)
        
        # Feed forward with residual connection
        ff_output = self.feed_forward(x)
        x = self.norm2(x + self.dropout(ff_output))
        
        outputs = {'hidden': x}
        if return_attention:
            outputs['self_attention'] = self_attn_weights
            outputs['user_attention'] = user_attn_weights
            outputs['geo_attention'] = geo_attn_weights
            
        return outputs


class GeographicalAttention(nn.Module):
    """Geographical attention module for spatial context"""
    
    def __init__(self, d_model: int, max_distance: float = 50.0):
        super().__init__()
        self.d_model = d_model
        self.max_distance = max_distance
        
        # Distance embedding
        self.distance_embedding = nn.Sequential(
            nn.Linear(1, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, d_model)
        )
        
        # Spatial attention
        self.spatial_attention = nn.MultiheadAttention(
            d_model, num_heads=4, batch_first=True
        )
        
    def forward(self, poi_embeddings: torch.Tensor, 
                poi_coords: torch.Tensor,
                candidate_coords: torch.Tensor) -> torch.Tensor:
        """
        Compute geographical context based on distances
        """
        batch_size, seq_len, _ = poi_embeddings.shape
        num_candidates = candidate_coords.shape[1]
        
        # Compute pairwise distances
        distances = self._compute_distances(poi_coords, candidate_coords)
        
        # Normalize distances
        normalized_distances = torch.clamp(distances / self.max_distance, 0, 1)
        
        # Create distance embeddings
        distance_emb = self.distance_embedding(normalized_distances.unsqueeze(-1))
        
        # Apply spatial attention
        geo_context, _ = self.spatial_attention(
            poi_embeddings.unsqueeze(2).expand(-1, -1, num_candidates, -1).reshape(batch_size, -1, self.d_model),
            distance_emb.reshape(batch_size, -1, self.d_model),
            distance_emb.reshape(batch_size, -1, self.d_model)
        )
        
        return geo_context.reshape(batch_size, seq_len, num_candidates, self.d_model)
    
    def _compute_distances(self, coords1: torch.Tensor, coords2: torch.Tensor) -> torch.Tensor:
        """Compute haversine distance between coordinates"""
        # Simplified euclidean distance (you should use haversine for real lat/lon)
        coords1 = coords1.unsqueeze(2)  # [batch, seq, 1, 2]
        coords2 = coords2.unsqueeze(1)  # [batch, 1, candidates, 2]
        return torch.norm(coords1 - coords2, dim=-1)


class UserContextEncoder(nn.Module):
    """Encode user preferences and patterns"""
    
    def __init__(self, d_model: int, num_users: int, num_categories: int):
        super().__init__()
        
        # User embedding
        self.user_embedding = nn.Embedding(num_users, d_model)
        
        # User preference modeling
        self.preference_encoder = nn.Sequential(
            nn.Linear(num_categories, d_model),
            nn.ReLU(),
            nn.Linear(d_model, d_model)
        )
        
        # Temporal pattern encoder
        self.temporal_pattern_encoder = nn.LSTM(
            d_model, d_model // 2, bidirectional=True, batch_first=True
        )
        
    def forward(self, user_ids: torch.Tensor, 
                user_preferences: torch.Tensor,
                historical_patterns: Optional[torch.Tensor] = None) -> torch.Tensor:
        
        # Get user embeddings
        user_emb = self.user_embedding(user_ids)
        
        # Encode preferences
        pref_emb = self.preference_encoder(user_preferences)
        
        # Combine user and preference embeddings
        context = user_emb + pref_emb
        
        # Add historical patterns if available
        if historical_patterns is not None:
            pattern_emb, _ = self.temporal_pattern_encoder(historical_patterns)
            context = context + pattern_emb.mean(dim=1)
            
        return context.unsqueeze(1)  # Add sequence dimension


class POITransformerWithCrossAttention(nn.Module):
    """Enhanced POI Transformer with Cross-Attention for next POI recommendation"""
    
    def __init__(self, config: Dict):
        super().__init__()
        self.config = config
        
        # Multi-modal embedding
        self.embedding = MultiModalEmbedding(config)
        
        # Temporal positional encoding
        self.pos_encoding = TemporalPositionalEncoding(config['d_model'])
        
        # User context encoder
        self.user_encoder = UserContextEncoder(
            config['d_model'],
            config['num_users'],
            config['num_categories']
        )
        
        # Geographical attention
        self.geo_attention = GeographicalAttention(
            config['d_model'],
            config.get('max_distance', 50.0)
        )
        
        # Transformer blocks with cross-attention
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                d_model=config['d_model'],
                nhead=config['nhead'],
                dim_feedforward=config['dim_feedforward'],
                dropout=config['dropout'],
                use_cross_attention=(i % 2 == 0)  # Alternate cross-attention layers
            ) for i in range(config['num_layers'])
        ])
        
        # Candidate POI encoder
        self.candidate_encoder = nn.Linear(config['d_model'], config['d_model'])
        
        # Output heads
        self.poi_scorer = nn.Sequential(
            nn.Linear(config['d_model'] * 2, config['d_model']),
            nn.ReLU(),
            nn.Dropout(config['dropout']),
            nn.Linear(config['d_model'], 1)
        )
        
        self.category_prediction_head = nn.Linear(config['d_model'], config['num_categories'])
        
        self.time_regression_head = nn.Sequential(
            nn.Linear(config['d_model'], config['d_model'] // 2),
            nn.ReLU(),
            nn.Linear(config['d_model'] // 2, 1),
            nn.Softplus()
        )
        
        # Dropout
        self.dropout = nn.Dropout(config['dropout'])
        
    def forward(self, batch_data: Dict, return_attention: bool = False) -> Dict:
        # Historical trajectory
        poi_ids = batch_data['poi_ids']
        categories = batch_data['categories']
        timestamps = batch_data['timestamps']
        days = batch_data['days']
        seasons = batch_data['seasons']
        regions = batch_data['regions']
        coords = batch_data['coords']
        
        # User context
        user_ids = batch_data.get('user_ids')
        user_preferences = batch_data.get('user_preferences')
        
        # Candidate POIs for ranking
        candidate_poi_ids = batch_data.get('candidate_poi_ids')
        candidate_coords = batch_data.get('candidate_coords')
        
        batch_size, seq_len = poi_ids.shape
        
        # Get embeddings for historical trajectory
        trajectory_emb = self.embedding(
            poi_ids, categories, timestamps, days, seasons, regions, coords
        )
        
        # Add positional encoding
        trajectory_emb = self.pos_encoding(trajectory_emb, timestamps, days)
        trajectory_emb = self.dropout(trajectory_emb)
        
        # Get user context
        user_context = None
        if user_ids is not None:
            user_context = self.user_encoder(user_ids, user_preferences)
        
        # Get geographical context
        geo_context = None
        if candidate_coords is not None:
            geo_context = self.geo_attention(
                trajectory_emb, coords, candidate_coords
            ).mean(dim=2)  # Aggregate over candidates
        
        # Create attention mask (optional: use causal or bidirectional)
        mask = None
        if self.config.get('use_causal_mask', False):
            mask = self.generate_causal_mask(seq_len).to(trajectory_emb.device)
        
        # Store attention weights
        attention_weights = []
        
        # Apply transformer blocks
        x = trajectory_emb
        for block in self.transformer_blocks:
            outputs = block(
                x, 
                user_context=user_context,
                geo_context=geo_context,
                mask=mask,
                return_attention=return_attention
            )
            x = outputs['hidden']
            
            if return_attention:
                attention_weights.append(outputs)
        
        # Get sequence representation (last hidden state or pooled)
        if self.config.get('pooling', 'last') == 'last':
            sequence_repr = x[:, -1, :]  # Last position
        else:
            sequence_repr = x.mean(dim=1)  # Mean pooling
        
        # Score candidate POIs
        poi_scores = None
        if candidate_poi_ids is not None:
            # Encode candidate POIs
            candidate_emb = self.embedding.poi_embedding(candidate_poi_ids)
            candidate_emb = self.candidate_encoder(candidate_emb)
            
            # Compute similarity scores
            batch_size, num_candidates, _ = candidate_emb.shape
            sequence_repr_expanded = sequence_repr.unsqueeze(1).expand(-1, num_candidates, -1)
            
            # Concatenate trajectory and candidate representations
            combined = torch.cat([sequence_repr_expanded, candidate_emb], dim=-1)
            poi_scores = self.poi_scorer(combined).squeeze(-1)
        
        # Category prediction
        category_logits = self.category_prediction_head(x)
        
        # Time to next POI
        time_to_next = self.time_regression_head(x)
        
        outputs = {
            'poi_scores': poi_scores,  # Scores for candidate POIs
            'category_logits': category_logits,
            'time_to_next': time_to_next,
            'sequence_representation': sequence_repr
        }
        
        if return_attention:
            outputs['attention_weights'] = attention_weights
        
        return outputs
    
    def generate_causal_mask(self, size: int) -> torch.Tensor:
        """Generate causal mask for autoregressive attention"""
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        return mask.masked_fill(mask == 1, float('-inf'))
    
    def compute_loss(self, outputs: Dict, targets: Dict) -> Dict[str, torch.Tensor]:
        """Compute multi-task loss"""
        losses = {}
        
        # POI ranking loss (if candidates provided)
        if outputs['poi_scores'] is not None and 'target_poi_idx' in targets:
            target_idx = targets['target_poi_idx']
            poi_loss = F.cross_entropy(outputs['poi_scores'], target_idx)
            losses['poi_loss'] = poi_loss
        
        # Category loss
        if 'target_categories' in targets:
            cat_loss = F.cross_entropy(
                outputs['category_logits'].reshape(-1, self.config['num_categories']),
                targets['target_categories'].reshape(-1)
            )
            losses['category_loss'] = cat_loss
        
        # Time regression loss
        if 'target_times' in targets:
            time_loss = F.mse_loss(
                outputs['time_to_next'].squeeze(-1),
                targets['target_times']
            )
            losses['time_loss'] = time_loss
        
        # Total loss with weights
        total_loss = sum(
            self.config.get(f'{k}_weight', 1.0) * v 
            for k, v in losses.items()
        )
        losses['total_loss'] = total_loss
        
        return losses
