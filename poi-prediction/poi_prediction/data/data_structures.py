"""
Data structures for POI prediction
"""

from dataclasses import dataclass
from typing import List, Optional, Dict, Any
import torch
from enum import Enum


class POICategory(Enum):
    """POI Categories"""
    FOOD = 0
    WORK = 1
    FITNESS = 2
    SHOPPING = 3
    ENTERTAINMENT = 4
    PERSONAL = 5
    TRANSPORT = 6
    EDUCATION = 8
    HEALTHCARE = 9
    ACCOMMODATION = 10
    TRAVEL = 11
    SERVICES = 12
    OTHERS = 13


@dataclass
class POI:
    """Point of Interest data structure"""
    poi_id: int
    name: str
    category: int
    latitude: float
    longitude: float
    region_id: int
    popularity_score: float = 0.0
    metadata: Optional[Dict[str, Any]] = None


@dataclass
class Visit:
    """Visit data structure"""
    poi_id: int
    user_id: int
    timestamp: int  # Unix timestamp
    hour_of_day: int  # 0-23
    day_of_week: int  # 0-6 (Monday=0)
    season: int  # 0-3 (Spring=0)
    duration_minutes: float
    check_in_type: str = "regular"  # regular, business, social
    

@dataclass
class UserProfile:
    """User profile data structure"""
    user_id: int
    age_group: int  # 0-6 (e.g., 18-25, 26-35, ...)
    occupation_category: int  # 0-9
    preferred_categories: List[int]  # List of preferred POI categories
    activity_level: float  # 0-1 (low to high)
    location_diversity: float  # 0-1 (low to high)
    temporal_regularity: float  # 0-1 (irregular to very regular)


@dataclass
class TrajectorySequence:
    """A sequence of POI visits for a user"""
    user_id: int
    visits: List[Visit]
    sequence_id: str
    start_time: int
    end_time: int
    
    def __len__(self) -> int:
        return len(self.visits)
    
    def get_poi_ids(self) -> List[int]:
        return [visit.poi_id for visit in self.visits]
    
    def get_categories(self) -> List[int]:
        # This would need POI database lookup in practice
        return [0] * len(self.visits)  # Placeholder


@dataclass
class BatchData:
    """Batch of training data"""
    
    # Historical trajectory features
    poi_ids: torch.Tensor  # (batch_size, seq_len)
    categories: torch.Tensor  # (batch_size, seq_len)
    timestamps: torch.Tensor  # (batch_size, seq_len)
    hours: torch.Tensor  # (batch_size, seq_len)
    days: torch.Tensor  # (batch_size, seq_len)
    seasons: torch.Tensor  # (batch_size, seq_len)
    regions: torch.Tensor  # (batch_size, seq_len)
    coordinates: torch.Tensor  # (batch_size, seq_len, 2)
    durations: torch.Tensor  # (batch_size, seq_len)
    
    # User context
    user_ids: torch.Tensor  # (batch_size,)
    user_preferences: torch.Tensor  # (batch_size, num_categories)
    
    # Sequence metadata
    sequence_lengths: torch.Tensor  # (batch_size,)
    attention_mask: torch.Tensor  # (batch_size, seq_len)
    
    # Candidate POIs for ranking (optional)
    candidate_poi_ids: Optional[torch.Tensor] = None  # (batch_size, num_candidates)
    candidate_coordinates: Optional[torch.Tensor] = None  # (batch_size, num_candidates, 2)
    
    def to(self, device: torch.device) -> 'BatchData':
        """Move batch to device"""
        new_batch = BatchData(
            poi_ids=self.poi_ids.to(device),
            categories=self.categories.to(device),
            timestamps=self.timestamps.to(device),
            hours=self.hours.to(device),
            days=self.days.to(device),
            seasons=self.seasons.to(device),
            regions=self.regions.to(device),
            coordinates=self.coordinates.to(device),
            durations=self.durations.to(device),
            user_ids=self.user_ids.to(device),
            user_preferences=self.user_preferences.to(device),
            sequence_lengths=self.sequence_lengths.to(device),
            attention_mask=self.attention_mask.to(device)
        )
        
        if self.candidate_poi_ids is not None:
            new_batch.candidate_poi_ids = self.candidate_poi_ids.to(device)
        if self.candidate_coordinates is not None:
            new_batch.candidate_coordinates = self.candidate_coordinates.to(device)
            
        return new_batch


@dataclass
class ModelTargets:
    """Training targets"""
    
    # Next POI prediction
    next_poi_ids: torch.Tensor  # (batch_size, seq_len)
    next_categories: torch.Tensor  # (batch_size, seq_len)
    next_durations: torch.Tensor  # (batch_size, seq_len)
    
    # For candidate ranking
    target_poi_indices: Optional[torch.Tensor] = None  # (batch_size,)
    
    def to(self, device: torch.device) -> 'ModelTargets':
        """Move targets to device"""
        new_targets = ModelTargets(
            next_poi_ids=self.next_poi_ids.to(device),
            next_categories=self.next_categories.to(device),
            next_durations=self.next_durations.to(device)
        )
        
        if self.target_poi_indices is not None:
            new_targets.target_poi_indices = self.target_poi_indices.to(device)
            
        return new_targets


@dataclass
class ModelOutput:
    """Model output structure"""
    
    # Predictions
    poi_scores: Optional[torch.Tensor] = None  # (batch_size, num_candidates)
    category_logits: Optional[torch.Tensor] = None  # (batch_size, seq_len, num_categories)
    duration_predictions: Optional[torch.Tensor] = None  # (batch_size, seq_len)
    
    # Representations
    sequence_representation: Optional[torch.Tensor] = None  # (batch_size, d_model)
    hidden_states: Optional[torch.Tensor] = None  # (batch_size, seq_len, d_model)
    
    # Attention weights
    attention_weights: Optional[List[Dict[str, torch.Tensor]]] = None
    
    # Loss components
    losses: Optional[Dict[str, torch.Tensor]] = None
