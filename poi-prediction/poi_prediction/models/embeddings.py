"""
Multi-modal embedding layers for POI prediction
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Dict, Optional, Tuple

from ..config.model_config import ModelConfig


class MultiModalEmbedding(nn.Module):
    """Enhanced multi-modal embedding with proper fusion"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        d_model = config.d_model
        
        # ============ Modality-specific embeddings ============
        self.poi_embedding = nn.Embedding(config.num_pois, config.poi_embed_dim, padding_idx=0)
        self.category_embedding = nn.Embedding(config.num_categories, config.cat_embed_dim)
        self.time_embedding = nn.Embedding(24, config.time_embed_dim)  # Hour of day
        self.day_embedding = nn.Embedding(7, config.day_embed_dim)  # Day of week
        self.season_embedding = nn.Embedding(4, config.season_embed_dim)  # Seasons
        self.region_embedding = nn.Embedding(config.num_regions, config.region_embed_dim)
        
        # Geographical embedding (continuous coordinates)
        self.geo_projection = nn.Sequential(
            nn.Linear(2, config.geo_embed_dim),
            nn.LayerNorm(config.geo_embed_dim),
            nn.ReLU()
        )
        
        # Duration embedding (continuous)
        self.duration_projection = nn.Sequential(
            nn.Linear(1, config.geo_embed_dim // 2),
            nn.LayerNorm(config.geo_embed_dim // 2),
            nn.ReLU()
        )
        
        # ============ Modality-specific projections to common space ============
        self.poi_proj = nn.Sequential(
            nn.Linear(config.poi_embed_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(config.dropout)
        )
        
        self.cat_proj = nn.Sequential(
            nn.Linear(config.cat_embed_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(config.dropout)
        )
        
        # Combine temporal features
        temporal_dim = (config.time_embed_dim + config.day_embed_dim + 
                       config.season_embed_dim)
        self.temporal_proj = nn.Sequential(
            nn.Linear(temporal_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(config.dropout)
        )
        
        # Combine spatial features
        spatial_dim = config.region_embed_dim + config.geo_embed_dim + config.geo_embed_dim // 2
        self.spatial_proj = nn.Sequential(
            nn.Linear(spatial_dim, d_model),
            nn.LayerNorm(d_model),
            nn.Dropout(config.dropout)
        )
        
        # ============ Learnable modality weights ============
        self.modality_weights = nn.Parameter(torch.ones(4))  # 4 modality groups
        
        # ============ Non-linear fusion network ============
        self.fusion_network = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
        
        # Initialize embeddings
        self._init_embeddings()
    
    def _init_embeddings(self):
        """Initialize embedding layers"""
        # Xavier initialization for embedding layers
        nn.init.xavier_uniform_(self.poi_embedding.weight)
        nn.init.xavier_uniform_(self.category_embedding.weight)
        nn.init.xavier_uniform_(self.time_embedding.weight)
        nn.init.xavier_uniform_(self.day_embedding.weight)
        nn.init.xavier_uniform_(self.season_embedding.weight)
        nn.init.xavier_uniform_(self.region_embedding.weight)
        
        # Set padding token to zero
        if hasattr(self.poi_embedding, 'padding_idx'):
            nn.init.constant_(self.poi_embedding.weight[0], 0)
    
    def forward(self, 
                poi_ids: torch.Tensor,
                categories: torch.Tensor,
                hours: torch.Tensor,
                days: torch.Tensor,
                seasons: torch.Tensor,
                regions: torch.Tensor,
                coordinates: torch.Tensor,
                durations: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through multi-modal embedding
        
        Args:
            poi_ids: (batch_size, seq_len)
            categories: (batch_size, seq_len)
            hours: (batch_size, seq_len)
            days: (batch_size, seq_len)
            seasons: (batch_size, seq_len)
            regions: (batch_size, seq_len)
            coordinates: (batch_size, seq_len, 2)
            durations: (batch_size, seq_len)
        
        Returns:
            fused_embeddings: (batch_size, seq_len, d_model)
        """
        # ============ Get individual embeddings ============
        poi_emb = self.poi_embedding(poi_ids)
        cat_emb = self.category_embedding(categories)
        time_emb = self.time_embedding(hours)
        day_emb = self.day_embedding(days)
        season_emb = self.season_embedding(seasons)
        region_emb = self.region_embedding(regions)
        
        # Process continuous features
        geo_emb = self.geo_projection(coordinates)
        duration_emb = self.duration_projection(durations.unsqueeze(-1))
        
        # ============ Group and project modalities ============
        # POI identity features
        poi_features = self.poi_proj(poi_emb)
        
        # Categorical features
        cat_features = self.cat_proj(cat_emb)
        
        # Temporal features (combined)
        temporal_combined = torch.cat([time_emb, day_emb, season_emb], dim=-1)
        temporal_features = self.temporal_proj(temporal_combined)
        
        # Spatial features (combined)
        spatial_combined = torch.cat([region_emb, geo_emb, duration_emb], dim=-1)
        spatial_features = self.spatial_proj(spatial_combined)
        
        # ============ Apply learnable modality weights ============
        weights = torch.softmax(self.modality_weights, dim=0)
        
        # Weighted combination of modalities
        fused = (weights[0] * poi_features + 
                weights[1] * cat_features + 
                weights[2] * temporal_features + 
                weights[3] * spatial_features)
        
        # ============ Non-linear fusion ============
        fused = self.fusion_network(fused)
        
        return fused


class TemporalPositionalEncoding(nn.Module):
    """Temporal-aware positional encoding for sequences"""
    
    def __init__(self, d_model: int, max_len: int = 1000):
        super().__init__()
        self.d_model = d_model
        
        # Standard sinusoidal positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
        
        # Learnable temporal encodings for cyclical patterns
        self.hour_encoding = nn.Parameter(torch.randn(24, d_model // 4))
        self.day_encoding = nn.Parameter(torch.randn(7, d_model // 4))
        
        # Dropout
        self.dropout = nn.Dropout(0.1)
        
    def forward(self, 
                x: torch.Tensor, 
                hours: Optional[torch.Tensor] = None, 
                days: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Add positional encoding to input embeddings
        
        Args:
            x: (batch_size, seq_len, d_model)
            hours: (batch_size, seq_len) - optional hour information
            days: (batch_size, seq_len) - optional day information
            
        Returns:
            x_with_pos: (batch_size, seq_len, d_model)
        """
        seq_len = x.size(1)
        
        # Add standard positional encoding
        x = x + self.pe[:, :seq_len]
        
        # Add temporal encodings if provided
        if hours is not None and days is not None:
            batch_size, seq_len = hours.shape
            
            # Get temporal encodings
            hour_enc = self.hour_encoding[hours]  # (batch, seq, d_model//4)
            day_enc = self.day_encoding[days]    # (batch, seq, d_model//4)
            
            # Pad to match d_model
            remaining_dim = self.d_model - hour_enc.size(-1) - day_enc.size(-1)
            if remaining_dim > 0:
                zeros = torch.zeros(batch_size, seq_len, remaining_dim, 
                                  device=x.device, dtype=x.dtype)
                temporal_enc = torch.cat([hour_enc, day_enc, zeros], dim=-1)
            else:
                temporal_enc = torch.cat([hour_enc, day_enc], dim=-1)
                temporal_enc = temporal_enc[:, :, :self.d_model]
            
            x = x + temporal_enc
        
        return self.dropout(x)


class UserContextEncoder(nn.Module):
    """Encode user preferences and patterns into context vectors"""
    
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        d_model = config.d_model
        
        # User embedding
        self.user_embedding = nn.Embedding(config.num_users, d_model, padding_idx=0)
        
        # User preference modeling (category preferences)
        self.preference_encoder = nn.Sequential(
            nn.Linear(config.num_categories, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(config.dropout),
            nn.Linear(d_model, d_model)
        )
        
        # Context fusion
        self.context_fusion = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(config.dropout)
        )
        
        # Initialize embeddings
        nn.init.xavier_uniform_(self.user_embedding.weight)
        if hasattr(self.user_embedding, 'padding_idx'):
            nn.init.constant_(self.user_embedding.weight[0], 0)
    
    def forward(self, 
                user_ids: torch.Tensor, 
                user_preferences: torch.Tensor) -> torch.Tensor:
        """
        Encode user context
        
        Args:
            user_ids: (batch_size,)
            user_preferences: (batch_size, num_categories)
            
        Returns:
            user_context: (batch_size, 1, d_model)
        """
        # Get user embeddings
        user_emb = self.user_embedding(user_ids)  # (batch_size, d_model)
        
        # Encode preferences
        pref_emb = self.preference_encoder(user_preferences)  # (batch_size, d_model)
        
        # Fuse user and preference information
        combined = torch.cat([user_emb, pref_emb], dim=-1)  # (batch_size, d_model*2)
        context = self.context_fusion(combined)  # (batch_size, d_model)
        
        # Add sequence dimension for compatibility with cross-attention
        return context.unsqueeze(1)  # (batch_size, 1, d_model)
