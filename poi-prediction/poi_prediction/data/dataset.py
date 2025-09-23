"""
Dataset and DataLoader for POI prediction
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from typing import List, Dict, Tuple, Optional
import random

from ..config.model_config import ModelConfig, DataConfig
from .data_structures import TrajectorySequence, BatchData, ModelTargets
from .synthetic_generator import AdvancedPOIDataGenerator


class POIDataset(Dataset):
    """Dataset for POI trajectory sequences"""
    
    def __init__(self, 
                 sequences: List[TrajectorySequence],
                 generator: AdvancedPOIDataGenerator,
                 config: DataConfig,
                 model_config: ModelConfig,
                 mode: str = 'train'):
        """
        Initialize POI Dataset
        
        Args:
            sequences: List of trajectory sequences
            generator: Data generator for POI information
            config: Data configuration
            model_config: Model configuration
            mode: 'train', 'val', or 'test'
        """
        self.sequences = sequences
        self.generator = generator
        self.config = config
        self.model_config = model_config
        self.mode = mode
        
        # Process sequences into subsequences
        self.processed_sequences = self._process_sequences()
        
        # Cache user preferences
        self._cache_user_preferences()
        
    def _process_sequences(self) -> List[Dict]:
        """Process sequences into fixed-length subsequences"""
        processed = []
        
        for seq in self.sequences:
            if len(seq.visits) < self.config.min_sequence_length:
                continue
                
            # Create sliding windows over the sequence
            max_len = min(len(seq.visits), self.config.max_sequence_length + 1)  # +1 for target
            
            for start_idx in range(0, len(seq.visits) - self.config.min_sequence_length + 1, 
                                 self.config.sequence_stride):
                end_idx = min(start_idx + max_len, len(seq.visits))
                
                if end_idx - start_idx >= self.config.min_sequence_length:
                    processed.append({
                        'sequence': seq,
                        'start_idx': start_idx,
                        'end_idx': end_idx,
                        'length': end_idx - start_idx
                    })
        
        return processed
    
    def _cache_user_preferences(self):
        """Cache user preferences for faster access"""
        self.user_preference_cache = {}
        
        for user_id, profile in self.generator.user_profiles.items():
            # Create one-hot encoded preference vector
            prefs = torch.zeros(self.model_config.num_categories)
            for cat in profile.preferred_categories:
                if cat < self.model_config.num_categories:
                    prefs[cat] = 1.0
            self.user_preference_cache[user_id] = prefs
    
    def __len__(self) -> int:
        return len(self.processed_sequences)
    
    def __getitem__(self, idx: int) -> Tuple[Dict, Dict]:
        """Get a single sample"""
        proc_seq = self.processed_sequences[idx]
        sequence = proc_seq['sequence']
        start_idx = proc_seq['start_idx']
        end_idx = proc_seq['end_idx']
        
        # Extract visits in the window
        visits = sequence.visits[start_idx:end_idx]
        actual_length = len(visits)
        
        # Prepare input features (all visits except last)
        input_visits = visits[:-1]
        input_length = len(input_visits)
        
        # Prepare targets (next visit for each input position)
        target_visits = visits[1:]
        
        # Initialize tensors with padding
        max_seq_len = self.config.max_sequence_length
        
        # Input features
        poi_ids = torch.zeros(max_seq_len, dtype=torch.long)
        categories = torch.zeros(max_seq_len, dtype=torch.long)
        timestamps = torch.zeros(max_seq_len, dtype=torch.long)
        hours = torch.zeros(max_seq_len, dtype=torch.long)
        days = torch.zeros(max_seq_len, dtype=torch.long)
        seasons = torch.zeros(max_seq_len, dtype=torch.long)
        regions = torch.zeros(max_seq_len, dtype=torch.long)
        coordinates = torch.zeros(max_seq_len, 2, dtype=torch.float32)
        durations = torch.zeros(max_seq_len, dtype=torch.float32)
        
        # Target features
        next_poi_ids = torch.full((max_seq_len,), -100, dtype=torch.long)  # -100 for padding
        next_categories = torch.full((max_seq_len,), -100, dtype=torch.long)
        next_durations = torch.full((max_seq_len,), -1.0, dtype=torch.float32)
        
        # Fill in actual data
        for i, visit in enumerate(input_visits):
            if i >= max_seq_len:
                break
                
            poi_features = self.generator.get_poi_features(visit.poi_id)
            
            poi_ids[i] = visit.poi_id
            categories[i] = poi_features.get('category', 0)
            timestamps[i] = visit.timestamp
            hours[i] = visit.hour_of_day
            days[i] = visit.day_of_week
            seasons[i] = visit.season
            regions[i] = poi_features.get('region_id', 0)
            coordinates[i] = torch.tensor([poi_features.get('latitude', 0.0), 
                                         poi_features.get('longitude', 0.0)])
            durations[i] = visit.duration_minutes / 60.0  # Convert to hours
        
        # Fill in targets
        for i, visit in enumerate(target_visits):
            if i >= max_seq_len:
                break
                
            poi_features = self.generator.get_poi_features(visit.poi_id)
            next_poi_ids[i] = visit.poi_id
            next_categories[i] = poi_features.get('category', 0)
            next_durations[i] = visit.duration_minutes / 60.0
        
        # Create attention mask (1 for real tokens, 0 for padding)
        attention_mask = torch.zeros(max_seq_len, dtype=torch.bool)
        attention_mask[:input_length] = 1
        
        # Get user preferences
        user_id = sequence.user_id
        user_preferences = self.user_preference_cache.get(
            user_id, 
            torch.zeros(self.model_config.num_categories)
        )
        
        # Get candidate POIs for ranking (if in training mode)
        candidate_poi_ids = None
        candidate_coordinates = None
        target_poi_idx = None
        
        if self.mode == 'train' and len(target_visits) > 0:
            # Get candidates including the ground truth
            last_visit = input_visits[-1]
            last_poi_features = self.generator.get_poi_features(last_visit.poi_id)
            current_location = (last_poi_features.get('latitude', 0.0), 
                              last_poi_features.get('longitude', 0.0))
            
            candidates = self.generator.get_candidate_pois(
                user_id, current_location, self.config.num_negative_samples
            )
            
            # Ensure ground truth is in candidates
            ground_truth_poi = target_visits[0].poi_id
            if ground_truth_poi not in candidates:
                candidates = candidates[:-1] + [ground_truth_poi]
            
            # Shuffle candidates
            random.shuffle(candidates)
            target_poi_idx = candidates.index(ground_truth_poi)
            
            # Get candidate features
            candidate_poi_ids = torch.tensor(candidates, dtype=torch.long)
            candidate_coords = []
            for cand_id in candidates:
                cand_features = self.generator.get_poi_features(cand_id)
                candidate_coords.append([
                    cand_features.get('latitude', 0.0),
                    cand_features.get('longitude', 0.0)
                ])
            candidate_coordinates = torch.tensor(candidate_coords, dtype=torch.float32)
            target_poi_idx = torch.tensor(target_poi_idx, dtype=torch.long)
        
        # Normalize coordinates if requested
        if self.config.normalize_coordinates:
            coordinates = self._normalize_coordinates(coordinates)
            if candidate_coordinates is not None:
                candidate_coordinates = self._normalize_coordinates(candidate_coordinates)
        
        # Create batch data
        batch_data = {
            'poi_ids': poi_ids,
            'categories': categories,
            'timestamps': timestamps,
            'hours': hours,
            'days': days,
            'seasons': seasons,
            'regions': regions,
            'coordinates': coordinates,
            'durations': durations,
            'user_ids': torch.tensor(user_id, dtype=torch.long),
            'user_preferences': user_preferences,
            'candidate_poi_ids': candidate_poi_ids,
            'candidate_coordinates': candidate_coordinates,
            'sequence_lengths': torch.tensor(input_length, dtype=torch.long),
            'attention_mask': attention_mask
        }
        
        # Create targets
        targets = {
            'next_poi_ids': next_poi_ids,
            'next_categories': next_categories,
            'next_durations': next_durations,
            'target_poi_indices': target_poi_idx
        }
        
        return batch_data, targets
    
    def _normalize_coordinates(self, coords: torch.Tensor) -> torch.Tensor:
        """Normalize coordinates to [-1, 1] range"""
        if coords.dim() == 1:
            coords = coords.unsqueeze(0)
        
        # Get bounds
        lng_bounds, lat_bounds = self.config.coordinate_bounds
        
        # Normalize longitude
        coords[..., 0] = 2 * (coords[..., 0] - lng_bounds[0]) / (lng_bounds[1] - lng_bounds[0]) - 1
        
        # Normalize latitude  
        coords[..., 1] = 2 * (coords[..., 1] - lat_bounds[0]) / (lat_bounds[1] - lat_bounds[0]) - 1
        
        return coords


def collate_fn(batch: List[Tuple[Dict, Dict]]) -> Tuple[BatchData, ModelTargets]:
    """Custom collate function for batching"""
    batch_data_list, targets_list = zip(*batch)
    
    # Stack all tensor fields
    batch_data = BatchData(
        poi_ids=torch.stack([item['poi_ids'] for item in batch_data_list]),
        categories=torch.stack([item['categories'] for item in batch_data_list]),
        timestamps=torch.stack([item['timestamps'] for item in batch_data_list]),
        hours=torch.stack([item['hours'] for item in batch_data_list]),
        days=torch.stack([item['days'] for item in batch_data_list]),
        seasons=torch.stack([item['seasons'] for item in batch_data_list]),
        regions=torch.stack([item['regions'] for item in batch_data_list]),
        coordinates=torch.stack([item['coordinates'] for item in batch_data_list]),
        durations=torch.stack([item['durations'] for item in batch_data_list]),
        user_ids=torch.stack([item['user_ids'] for item in batch_data_list]),
        user_preferences=torch.stack([item['user_preferences'] for item in batch_data_list]),
        sequence_lengths=torch.stack([item['sequence_lengths'] for item in batch_data_list]),
        attention_mask=torch.stack([item['attention_mask'] for item in batch_data_list])
    )
    
    # Handle optional candidate data
    candidate_poi_ids = None
    candidate_coordinates = None
    target_poi_indices = None
    
    if batch_data_list[0]['candidate_poi_ids'] is not None:
        candidate_poi_ids = torch.stack([item['candidate_poi_ids'] for item in batch_data_list])
        candidate_coordinates = torch.stack([item['candidate_coordinates'] for item in batch_data_list])
        target_poi_indices = torch.stack([item['target_poi_indices'] for item in targets_list])
    
    batch_data.candidate_poi_ids = candidate_poi_ids
    batch_data.candidate_coordinates = candidate_coordinates
    
    # Stack targets
    targets = ModelTargets(
        next_poi_ids=torch.stack([item['next_poi_ids'] for item in targets_list]),
        next_categories=torch.stack([item['next_categories'] for item in targets_list]),
        next_durations=torch.stack([item['next_durations'] for item in targets_list]),
        target_poi_indices=target_poi_indices
    )
    
    return batch_data, targets


def create_data_loaders(config: DataConfig, 
                       model_config: ModelConfig,
                       batch_size: int = 32,
                       num_workers: int = 4) -> Tuple[DataLoader, DataLoader, AdvancedPOIDataGenerator]:
    """Create train and validation data loaders"""
    
    # Create data generator
    generator = AdvancedPOIDataGenerator(num_users=model_config.num_users)
    
    # Generate synthetic dataset
    all_sequences = generator.generate_dataset(
        num_sequences=2000, 
        min_length=config.min_sequence_length,
        max_length=config.max_sequence_length + 5  
    )
    
    # Split into train/validation
    random.shuffle(all_sequences)
    split_idx = int(len(all_sequences) * (1 - 0.2))  # 80/20 split
    train_sequences = all_sequences[:split_idx]
    val_sequences = all_sequences[split_idx:]
    
    print(f"Created {len(train_sequences)} training sequences and {len(val_sequences)} validation sequences")
    
    # Create datasets
    train_dataset = POIDataset(
        train_sequences, generator, config, model_config, mode='train'
    )
    val_dataset = POIDataset(
        val_sequences, generator, config, model_config, mode='val'
    )
    
    print(f"Train dataset size: {len(train_dataset)}")
    print(f"Validation dataset size: {len(val_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=True
    )
    
    return train_loader, val_loader, generator
