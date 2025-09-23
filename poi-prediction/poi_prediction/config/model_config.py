"""
Model configuration for POI Transformer with Cross-Attention
"""

from dataclasses import dataclass
from typing import Optional, Dict, Any
from pathlib import Path


@dataclass
class ModelConfig:
    """Configuration for POI Transformer Model"""
    
    # Model architecture
    d_model: int = 128
    nhead: int = 8
    num_layers: int = 4
    dim_feedforward: int = 256
    dropout: float = 0.1
    max_sequence_length: int = 50
    
    # Data dimensions
    num_pois: int = 1000
    num_categories: int = 20
    num_users: int = 10000
    num_regions: int = 50
    
    # Embedding dimensions
    poi_embed_dim: int = 64
    cat_embed_dim: int = 32
    time_embed_dim: int = 32
    day_embed_dim: int = 16
    season_embed_dim: int = 16
    region_embed_dim: int = 32
    geo_embed_dim: int = 32
    
    # Cross-attention settings
    use_cross_attention: bool = True
    cross_attention_every_n_layers: int = 2
    
    # Geographical attention
    max_distance: float = 50.0  # km
    
    # Model behavior
    use_causal_mask: bool = False  # Set to True for autoregressive
    pooling: str = "last"  # "last" or "mean"
    
    # Loss weights
    poi_loss_weight: float = 1.0
    category_loss_weight: float = 0.5
    time_loss_weight: float = 0.3


@dataclass 
class TrainingConfig:
    """Training configuration"""
    
    # Optimization
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 32
    num_epochs: int = 100
    warmup_steps: int = 1000
    
    # Validation
    val_split: float = 0.2
    patience: int = 10
    
    # Logging
    log_interval: int = 100
    val_interval: int = 1000
    save_interval: int = 5000
    
    # Paths
    data_path: Optional[Path] = None
    output_dir: Path = Path("./outputs")
    checkpoint_dir: Path = Path("./checkpoints")
    
    # Device
    device: str = "auto"  # auto, cpu, cuda, mps
    
    # Distributed training
    use_ddp: bool = False
    
    # Wandb logging
    use_wandb: bool = False
    wandb_project: str = "poi-prediction"
    wandb_entity: Optional[str] = None


@dataclass
class DataConfig:
    """Data processing configuration"""
    
    # Trajectory processing
    min_sequence_length: int = 5
    max_sequence_length: int = 50
    sequence_stride: int = 1
    
    # Feature engineering
    normalize_coordinates: bool = True
    coordinate_bounds: tuple = ((-180, 180), (-90, 90))  # (lng, lat)
    
    # Time features
    time_bins: int = 24  # hours
    
    # Data augmentation
    add_noise: bool = False
    noise_std: float = 0.01
    
    # Candidate sampling
    num_negative_samples: int = 100
    candidate_sampling_strategy: str = "random"  # random, popularity, geographic


def get_config() -> Dict[str, Any]:
    """Get default configuration"""
    return {
        "model": ModelConfig(),
        "training": TrainingConfig(),
        "data": DataConfig()
    }
