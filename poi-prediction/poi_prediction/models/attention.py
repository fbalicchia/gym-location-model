"""
Cross-attention mechanisms for POI prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Dict

from ..config.model_config import ModelConfig


class CrossAttentionBlock(nn.Module):
    """Cross-attention block for attending between sequences and context"""
    
    def __init__(self, d_model: int, nhead: int, dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.nhead = nhead
        
        # Multi-head cross-attention
        self.cross_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        
        # Layer normalization and dropout
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, 
                query: torch.Tensor, 
                key: torch.Tensor, 
                value: torch.Tensor,
                key_padding_mask: Optional[torch.Tensor] = None,
                attn_mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass for cross-attention
        
        Args:
            query: (batch_size, tgt_len, d_model)
            key: (batch_size, src_len, d_model)  
            value: (batch_size, src_len, d_model)
            key_padding_mask: (batch_size, src_len)
            attn_mask: (tgt_len, src_len)
            
        Returns:
            output: (batch_size, tgt_len, d_model)
            attn_weights: (batch_size, nhead, tgt_len, src_len)
        """
        # Cross-attention
        attn_output, attn_weights = self.cross_attn(
            query, key, value, 
            key_padding_mask=key_padding_mask,
            attn_mask=attn_mask,
            need_weights=True
        )
        
        # Residual connection and layer norm
        output = self.norm(query + self.dropout(attn_output))
        
        return output, attn_weights


class GeographicalAttention(nn.Module):
    """Geographical attention module for spatial context"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        self.d_model = config.d_model
        self.max_distance = config.max_distance
        self.nhead = config.nhead
        
        # Distance encoding network
        self.distance_encoder = nn.Sequential(
            nn.Linear(1, config.d_model // 4),
            nn.LayerNorm(config.d_model // 4),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model // 4, config.d_model)
        )
        
        # Spatial attention
        self.spatial_attention = nn.MultiheadAttention(
            config.d_model, 
            num_heads=min(4, config.nhead), 
            dropout=config.dropout,
            batch_first=True
        )
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(config.d_model, config.d_model),
            nn.LayerNorm(config.d_model),
            nn.Dropout(config.dropout)
        )
        
    def forward(self, 
                trajectory_embeddings: torch.Tensor,
                trajectory_coords: torch.Tensor,
                candidate_coords: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute geographical attention
        
        Args:
            trajectory_embeddings: (batch_size, seq_len, d_model)
            trajectory_coords: (batch_size, seq_len, 2)
            candidate_coords: (batch_size, num_candidates, 2) - optional
            
        Returns:
            geo_context: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, d_model = trajectory_embeddings.shape
        
        if candidate_coords is not None:
            # Cross-location attention with candidates
            num_candidates = candidate_coords.shape[1]
            
            # Compute distances between trajectory points and candidates
            distances = self._compute_distances(trajectory_coords, candidate_coords)
            # (batch_size, seq_len, num_candidates)
            
            # Encode distances
            distance_features = self.distance_encoder(distances.unsqueeze(-1))
            # (batch_size, seq_len, num_candidates, d_model)
            
            # Reshape for attention
            query = trajectory_embeddings  # (batch_size, seq_len, d_model)
            key = distance_features.view(batch_size, seq_len * num_candidates, d_model)
            value = key
            
            # Apply spatial attention
            geo_attended, _ = self.spatial_attention(query, key, value)
            
        else:
            # Self-attention within trajectory based on spatial proximity
            # Compute pairwise distances within trajectory
            distances = self._compute_pairwise_distances(trajectory_coords)
            # (batch_size, seq_len, seq_len)
            
            # Create distance-based attention bias
            distance_bias = self._distance_to_bias(distances)
            # (batch_size, seq_len, seq_len)
            
            # Apply self-attention without distance bias for now
            # TODO: Implement proper batched attention masks
            geo_attended, _ = self.spatial_attention(
                trajectory_embeddings, 
                trajectory_embeddings, 
                trajectory_embeddings
            )
        
        # Final projection
        geo_context = self.output_proj(geo_attended)
        
        return geo_context
    
    def _compute_distances(self, coords1: torch.Tensor, coords2: torch.Tensor) -> torch.Tensor:
        """
        Compute distances between two sets of coordinates
        
        Args:
            coords1: (batch_size, len1, 2)
            coords2: (batch_size, len2, 2)
            
        Returns:
            distances: (batch_size, len1, len2)
        """
        # Expand dimensions for broadcasting
        coords1_expanded = coords1.unsqueeze(2)  # (batch_size, len1, 1, 2)
        coords2_expanded = coords2.unsqueeze(1)  # (batch_size, 1, len2, 2)
        
        # Compute Euclidean distance (simplified - in practice use Haversine for lat/lon)
        distances = torch.norm(coords1_expanded - coords2_expanded, dim=-1)
        
        # Normalize by max distance
        distances = torch.clamp(distances / self.max_distance, 0, 1)
        
        return distances
    
    def _compute_pairwise_distances(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Compute pairwise distances within a trajectory
        
        Args:
            coords: (batch_size, seq_len, 2)
            
        Returns:
            distances: (batch_size, seq_len, seq_len)
        """
        return self._compute_distances(coords, coords)
    
    def _distance_to_bias(self, distances: torch.Tensor) -> torch.Tensor:
        """
        Convert distances to attention bias (closer = higher attention)
        
        Args:
            distances: (batch_size, seq_len, seq_len)
            
        Returns:
            bias: (batch_size, seq_len, seq_len)
        """
        # Invert distances so closer locations get higher weights
        # Use negative log to create attention bias
        bias = -torch.log(distances + 1e-8)
        
        # Mask out very distant locations
        mask = distances > 0.8  # Very distant locations
        bias = bias.masked_fill(mask, float('-inf'))
        
        return bias


class MultiScaleCrossAttention(nn.Module):
    """Multi-scale cross-attention for different types of context"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        
        self.d_model = config.d_model
        self.nhead = config.nhead
        
        # Different attention heads for different contexts
        self.user_cross_attn = CrossAttentionBlock(
            config.d_model, 
            config.nhead // 2,  # Use fewer heads for user context
            config.dropout
        )
        
        self.geo_cross_attn = GeographicalAttention(config)
        
        # Temporal cross-attention for time-based patterns
        self.temporal_cross_attn = CrossAttentionBlock(
            config.d_model,
            config.nhead // 2,
            config.dropout
        )
        
        # Context fusion network
        self.context_fusion = nn.Sequential(
            nn.Linear(config.d_model * 3, config.d_model * 2),
            nn.LayerNorm(config.d_model * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.d_model * 2, config.d_model),
            nn.LayerNorm(config.d_model)
        )
        
        # Gating mechanism to control context influence
        self.context_gate = nn.Sequential(
            nn.Linear(config.d_model * 3, 3),
            nn.Softmax(dim=-1)
        )
        
    def forward(self, 
                trajectory_emb: torch.Tensor,
                trajectory_coords: torch.Tensor,
                user_context: Optional[torch.Tensor] = None,
                temporal_context: Optional[torch.Tensor] = None,
                candidate_coords: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Multi-scale cross-attention forward pass
        
        Args:
            trajectory_emb: (batch_size, seq_len, d_model)
            trajectory_coords: (batch_size, seq_len, 2)
            user_context: (batch_size, 1, d_model)
            temporal_context: (batch_size, time_len, d_model)
            candidate_coords: (batch_size, num_candidates, 2)
            
        Returns:
            Dictionary with attended features and attention weights
        """
        batch_size, seq_len, d_model = trajectory_emb.shape
        
        # Initialize contexts
        user_attended = trajectory_emb
        geo_attended = trajectory_emb
        temporal_attended = trajectory_emb
        
        attention_weights = {}
        
        # User context cross-attention
        if user_context is not None:
            user_attended, user_attn = self.user_cross_attn(
                trajectory_emb, user_context, user_context
            )
            attention_weights['user_attention'] = user_attn
        
        # Geographical cross-attention
        geo_attended = self.geo_cross_attn(
            trajectory_emb, trajectory_coords, candidate_coords
        )
        
        # Temporal context cross-attention
        if temporal_context is not None:
            temporal_attended, temporal_attn = self.temporal_cross_attn(
                trajectory_emb, temporal_context, temporal_context
            )
            attention_weights['temporal_attention'] = temporal_attn
        
        # Combine all contexts
        all_contexts = torch.stack([user_attended, geo_attended, temporal_attended], dim=-1)
        # (batch_size, seq_len, d_model, 3)
        
        # Compute context gates
        context_input = torch.cat([user_attended, geo_attended, temporal_attended], dim=-1)
        context_weights = self.context_gate(context_input)  # (batch_size, seq_len, 3)
        
        # Weighted combination of contexts
        weighted_context = torch.sum(
            all_contexts * context_weights.unsqueeze(-2), 
            dim=-1
        )  # (batch_size, seq_len, d_model)
        
        # Final fusion
        fused_input = torch.cat([user_attended, geo_attended, temporal_attended], dim=-1)
        final_output = self.context_fusion(fused_input)
        
        # Residual connection with original trajectory
        output = final_output + trajectory_emb
        
        return {
            'output': output,
            'user_attended': user_attended,
            'geo_attended': geo_attended,
            'temporal_attended': temporal_attended,
            'attention_weights': attention_weights,
            'context_weights': context_weights
        }
