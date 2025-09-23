"""
Synthetic POI data generator based on the advanced notebook implementation
"""

import torch
import numpy as np
import random
import math
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict

from .data_structures import POI, Visit, UserProfile, TrajectorySequence


"""
	1.	id → unique numeric identifier of the POI.
	2.	name → human-readable name of the POI.
	3.	category → type or semantic category of the POI (e.g., work, food).
	4.	latitude → geographic latitude coordinate.
	5.	longitude → geographic longitude coordinate.
	6.	zone/region_id → numeric label of the area or cluster the POI belongs to.
	7.	importance/weight → relevance score or user preference strength for the POI.
"""


class AdvancedPOIDataGenerator:
    """Generate rich synthetic POI data with realistic patterns"""
    
    def __init__(self, num_users: int = 1000, seed: int = 42):
        self.num_users = num_users
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        # POI categories mapping to enum values
        self.categories = {
            'food': 0,
            'work': 1,
            'fitness': 2,
            'shopping': 3,
            'entertainment': 4,
            'personal': 5,
            'transport': 6,
            'education': 7,
            'healthcare': 8,
            'accommodation': 9,
            'travel': 10,
            'services': 11,
            'others': 12
        }
        
        # Create POI database with realistic locations (NYC area)
        self.poi_database = self._create_poi_database()
        
        # Generate user profiles
        self.user_profiles = self._generate_user_profiles()
        
        # Temporal patterns for different POI types
        self.periodic_patterns = {
            'gym': {'frequency': 'bi-weekly', 'preferred_times': ['morning', 'evening']},
            'barber': {'frequency': 'monthly', 'preferred_times': ['afternoon']},
            'supermarket': {'frequency': 'weekly', 'preferred_times': ['afternoon', 'evening']},
            'cinema': {'frequency': 'weekly', 'preferred_times': ['evening', 'night']},
            'office': {'frequency': 'daily', 'preferred_times': ['morning']},
            'restaurant': {'frequency': 'frequent', 'preferred_times': ['lunch', 'dinner']},
        }
        
        # Long-range dependency patterns
        self.dependency_patterns = {
            'coffee_shop': ['office'],  # Coffee before work
            'gym': ['home', 'office'],  # Gym after work or from home
            'restaurant': ['cinema', 'shopping_mall'],  # Dinner and entertainment
            'pharmacy': ['home', 'hospital'],  # Medicine then home
            'gas_station': ['highway', 'travel'],  # Travel patterns
        }
        
        self.num_pois = len(self.poi_database)
        self.num_categories = len(self.categories)
        self.num_regions = 10  # NYC boroughs/areas
        
    def _create_poi_database(self) -> Dict[int, POI]:
        """Create a realistic POI database"""
        pois = {}
        
        # NYC coordinates (roughly)
        base_lat, base_lon = 40.7128, -74.0060
        
        poi_data = [
            # Essential places
            (0, 'home', 'personal', 40.7128, -74.0060, 0, 1.0),
            (1, 'office', 'work', 40.7580, -73.9855, 1, 0.9),
            (2, 'starbucks', 'food', 40.7489, -73.9680, 1, 0.8),
            (3, 'gym', 'fitness', 40.7614, -73.9776, 1, 0.7),
            (4, 'whole_foods', 'shopping', 40.7424, -74.0055, 2, 0.7),
            
            # Restaurants
            (5, 'italian_restaurant', 'food', 40.7431, -73.9897, 2, 0.6),
            (6, 'sushi_bar', 'food', 40.7516, -73.9755, 1, 0.6),
            (7, 'pizza_place', 'food', 40.7505, -73.9780, 1, 0.8),
            (8, 'chinese_restaurant', 'food', 40.7390, -73.9900, 2, 0.5),
            
            # Entertainment
            (9, 'cinema', 'entertainment', 40.7580, -73.9855, 1, 0.6),
            (10, 'central_park', 'entertainment', 40.7829, -73.9654, 3, 0.9),
            (11, 'museum', 'entertainment', 40.7794, -73.9632, 3, 0.5),
            (12, 'bar', 'entertainment', 40.7400, -73.9900, 2, 0.6),
            
            # Services
            (13, 'barber', 'personal', 40.7480, -73.9870, 0, 0.4),
            (14, 'bank', 'personal', 40.7505, -73.9934, 1, 0.7),
            (15, 'pharmacy', 'healthcare', 40.7480, -73.9870, 0, 0.8),
            (16, 'hospital', 'healthcare', 40.7350, -73.9900, 2, 0.9),
            
            # Transport
            (17, 'subway_station', 'transport', 40.7527, -73.9772, 1, 1.0),
            (18, 'bus_stop', 'transport', 40.7400, -73.9800, 1, 0.9),
            (19, 'taxi_stand', 'transport', 40.7580, -73.9855, 1, 0.8),
            
            # Shopping
            (20, 'clothing_store', 'shopping', 40.7550, -73.9840, 1, 0.6),
            (21, 'electronics_store', 'shopping', 40.7450, -73.9850, 1, 0.5),
            (22, 'bookstore', 'shopping', 40.7520, -73.9800, 1, 0.4),
            
            # Extended locations for diversity
            (23, 'library', 'education', 40.7532, -73.9822, 3, 0.7),
            (24, 'school', 'education', 40.7600, -73.9700, 4, 0.8),
            (25, 'hotel', 'accommodation', 40.7590, -73.9840, 1, 0.5),
        ]
        
        for poi_id, name, category_name, lat, lon, region, popularity in poi_data:
            pois[poi_id] = POI(
                poi_id=poi_id,
                name=name,
                category=self.categories[category_name],
                latitude=lat,
                longitude=lon,
                region_id=region,
                popularity_score=popularity
            )
        
        return pois
    
    def _generate_user_profiles(self) -> Dict[int, UserProfile]:
        """Generate diverse user profiles"""
        profiles = {}
        
        for user_id in range(self.num_users):
            # Random demographics
            age_group = random.randint(0, 6)  # 7 age groups
            occupation = random.randint(0, 9)  # 10 occupation categories
            
            # Generate preferences based on age and occupation
            preferred_categories = self._generate_preferences(age_group, occupation)
            
            # Activity characteristics
            activity_level = random.uniform(0.3, 1.0)
            location_diversity = random.uniform(0.2, 0.9)
            temporal_regularity = random.uniform(0.4, 0.9)
            
            profiles[user_id] = UserProfile(
                user_id=user_id,
                age_group=age_group,
                occupation_category=occupation,
                preferred_categories=preferred_categories,
                activity_level=activity_level,
                location_diversity=location_diversity,
                temporal_regularity=temporal_regularity
            )
        
        return profiles
    
    def _generate_preferences(self, age_group: int, occupation: int) -> List[int]:
        """Generate category preferences based on demographics"""
        base_preferences = []
        
        # Age-based preferences
        if age_group <= 2:  # Young adults (18-35)
            base_preferences.extend([self.categories['entertainment'], 
                                   self.categories['fitness'],
                                   self.categories['food']])
        elif age_group <= 4:  # Middle-aged (36-55)
            base_preferences.extend([self.categories['work'], 
                                   self.categories['shopping'],
                                   self.categories['healthcare']])
        else:  # Older adults (55+)
            base_preferences.extend([self.categories['healthcare'], 
                                   self.categories['personal'],
                                   self.categories['entertainment']])
        
        # Occupation-based preferences
        if occupation in [0, 1, 2]:  # Office workers
            base_preferences.append(self.categories['work'])
        elif occupation in [3, 4]:  # Service workers
            base_preferences.append(self.categories['transport'])
        elif occupation in [5, 6]:  # Creative/education
            base_preferences.append(self.categories['education'])
        
        # Add some randomness
        all_categories = list(self.categories.values())
        additional = random.sample(all_categories, random.randint(1, 3))
        base_preferences.extend(additional)
        
        return list(set(base_preferences))  # Remove duplicates
    
    def calculate_distance(self, poi1: POI, poi2: POI) -> float:
        """Calculate approximate distance between two POIs (km)"""
        # Simplified distance calculation (should use Haversine for real lat/lon)
        lat_diff = poi1.latitude - poi2.latitude
        lon_diff = poi1.longitude - poi2.longitude
        # Rough conversion to km for NYC area
        return math.sqrt(lat_diff**2 + lon_diff**2) * 111  # 1 degree ≈ 111 km
    
    def generate_user_trajectory(self, 
                                user_id: int, 
                                length: int = 20,
                                time_span_days: int = 7) -> TrajectorySequence:
        """Generate a realistic trajectory for a specific user"""
        if user_id not in self.user_profiles:
            raise ValueError(f"User {user_id} not found in profiles")
        
        user_profile = self.user_profiles[user_id]
        visits = []
        
        # Initialize trajectory state
        current_hour = random.randint(7, 10)  # Start morning
        current_day = random.randint(0, 6)
        current_season = random.randint(0, 3)
        base_timestamp = random.randint(1640995200, 1672531200)  # 2022 timestamps
        
        # Start from home
        current_poi_id = 0
        visit_history = defaultdict(list)
        
        for step in range(length):
            # Create visit
            timestamp = base_timestamp + (step * 3600)  # Hour intervals
            
            visit = Visit(
                poi_id=current_poi_id,
                user_id=user_id,
                timestamp=timestamp,
                hour_of_day=current_hour,
                day_of_week=current_day,
                season=current_season,
                duration_minutes=random.uniform(30, 180),  # 30 min to 3 hours
                check_in_type=random.choice(['regular', 'business', 'social'])
            )
            
            visits.append(visit)
            visit_history[self.poi_database[current_poi_id].name].append(step)
            
            # Determine next POI based on user profile and patterns
            next_poi_id = self._select_next_poi_for_user(
                user_profile,
                current_poi_id,
                current_hour,
                current_day,
                visit_history,
                step
            )
            
            # Update time (simplified progression)
            duration_hours = visit.duration_minutes / 60
            current_hour = int((current_hour + duration_hours + 1) % 24)
            
            if current_hour < 7:  # New day started
                current_day = (current_day + 1) % 7
            
            current_poi_id = next_poi_id
        
        sequence_id = f"user_{user_id}_seq_{random.randint(1000, 9999)}"
        
        return TrajectorySequence(
            user_id=user_id,
            visits=visits,
            sequence_id=sequence_id,
            start_time=visits[0].timestamp,
            end_time=visits[-1].timestamp
        )
    
    def _select_next_poi_for_user(self, 
                                 user_profile: UserProfile,
                                 current_poi_id: int, 
                                 hour: int, 
                                 day: int,
                                 history: Dict, 
                                 step: int) -> int:
        """Select next POI based on user profile and context"""
        current_poi = self.poi_database[current_poi_id]
        candidates = []
        weights = []
        
        for poi_id, poi in self.poi_database.items():
            if poi_id == current_poi_id:
                continue
            
            weight = 1.0
            
            # User preference factor
            if poi.category in user_profile.preferred_categories:
                weight *= 2.0
            
            # Distance factor (closer POIs preferred, but varies by user diversity)
            distance = self.calculate_distance(current_poi, poi)
            distance_factor = max(0.1, 1.0 - (distance / 10.0))  # 10km normalization
            weight *= distance_factor ** (2 - user_profile.location_diversity)
            
            # Time appropriateness
            weight *= self._get_time_appropriateness(poi, hour, day)
            
            # Popularity factor
            weight *= (0.5 + poi.popularity_score)
            
            # Periodic patterns (user regularity affects this)
            if poi.name in self.periodic_patterns:
                pattern_weight = self._get_pattern_weight(poi.name, history, step)
                weight *= (1 + pattern_weight * user_profile.temporal_regularity)
            
            # Long-range dependencies
            if current_poi.name in self.dependency_patterns.get(poi.name, []):
                weight *= 1.5
            
            # Activity level affects exploration
            if poi_id not in [v.poi_id for v in history.values() for v in []]:
                weight *= (0.5 + user_profile.activity_level)
            
            candidates.append(poi_id)
            weights.append(weight)
        
        # Normalize and sample
        if not weights:
            return random.choice(list(self.poi_database.keys()))
        
        total_weight = sum(weights)
        if total_weight == 0:
            return random.choice(candidates)
        
        weights = [w / total_weight for w in weights]
        return random.choices(candidates, weights=weights)[0]
    
    def _get_time_appropriateness(self, poi: POI, hour: int, day: int) -> float:
        """Get time-based appropriateness score for a POI"""
        multiplier = 1.0
        
        # Category-based time preferences
        if poi.category == self.categories['food']:
            if 7 <= hour < 10 or 12 <= hour < 14 or 18 <= hour < 21:
                multiplier *= 2.0
        elif poi.category == self.categories['work']:
            if 8 <= hour < 18 and day < 5:  # Weekdays
                multiplier *= 3.0
            else:
                multiplier *= 0.3
        elif poi.category == self.categories['entertainment']:
            if hour >= 17 or day >= 5:  # Evening or weekend
                multiplier *= 2.0
        elif poi.category == self.categories['fitness']:
            if 6 <= hour < 10 or 17 <= hour < 21:
                multiplier *= 2.0
        elif poi.category == self.categories['shopping']:
            if 10 <= hour < 20:
                multiplier *= 1.5
        
        return multiplier
    
    def _get_pattern_weight(self, poi_name: str, history: Dict, current_step: int) -> float:
        """Get weight based on periodic patterns"""
        if poi_name not in history or not history[poi_name]:
            return 1.0  # First visit
        
        pattern = self.periodic_patterns[poi_name]
        last_visit = history[poi_name][-1]
        steps_since = current_step - last_visit
        
        if pattern['frequency'] == 'daily':
            return max(0, 1.0 - steps_since / 24)  # Once per day
        elif pattern['frequency'] == 'weekly':
            return max(0, 1.0 - abs(steps_since - 168) / 24)  # Once per week
        elif pattern['frequency'] == 'bi-weekly':
            return max(0, 1.0 - abs(steps_since - 336) / 48)  # Every two weeks
        elif pattern['frequency'] == 'monthly':
            return max(0, 1.0 - abs(steps_since - 720) / 168)  # Once per month
        
        return 0.5
    
    def generate_dataset(self, 
                        num_sequences: int = 1000,
                        min_length: int = 10,
                        max_length: int = 30) -> List[TrajectorySequence]:
        """Generate a dataset of trajectory sequences"""
        sequences = []
        
        for _ in range(num_sequences):
            # Select random user
            user_id = random.randint(0, self.num_users - 1)
            
            # Random sequence length
            length = random.randint(min_length, max_length)
            
            # Generate trajectory
            try:
                sequence = self.generate_user_trajectory(user_id, length)
                sequences.append(sequence)
            except Exception as e:
                print(f"Error generating sequence for user {user_id}: {e}")
                continue
        
        return sequences
    
    def get_poi_features(self, poi_id: int) -> Dict:
        """Get features for a specific POI"""
        if poi_id not in self.poi_database:
            return {}
        
        poi = self.poi_database[poi_id]
        return {
            'poi_id': poi.poi_id,
            'category': poi.category,
            'latitude': poi.latitude,
            'longitude': poi.longitude,
            'region_id': poi.region_id,
            'popularity_score': poi.popularity_score
        }
    
    def get_candidate_pois(self, 
                          user_id: int,
                          current_location: Tuple[float, float],
                          num_candidates: int = 50) -> List[int]:
        """Get candidate POIs for a user at a given location"""
        if user_id not in self.user_profiles:
            return list(self.poi_database.keys())[:num_candidates]
        
        user_profile = self.user_profiles[user_id]
        lat, lon = current_location
        
        # Score all POIs for this user and location
        poi_scores = []
        for poi_id, poi in self.poi_database.items():
            # Distance factor
            distance = math.sqrt((poi.latitude - lat)**2 + (poi.longitude - lon)**2) * 111
            distance_score = max(0.1, 1.0 - distance / 10.0)
            
            # User preference factor
            preference_score = 2.0 if poi.category in user_profile.preferred_categories else 1.0
            
            # Popularity factor
            popularity_score = poi.popularity_score
            
            total_score = distance_score * preference_score * popularity_score
            poi_scores.append((poi_id, total_score))
        
        # Sort by score and return top candidates
        poi_scores.sort(key=lambda x: x[1], reverse=True)
        return [poi_id for poi_id, _ in poi_scores[:num_candidates]]
