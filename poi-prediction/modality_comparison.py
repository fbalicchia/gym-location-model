"""
Comparison: Simple Concatenation vs Learnable Weighted Fusion
"""

import torch
import torch.nn as nn
import numpy as np

def demonstrate_modality_fusion():
    """
    Demonstrate the difference between concatenation and learnable weighted fusion
    """
    
    # Simulate 4 modality features (each with d_model=128 dimensions)
    batch_size, seq_len, d_model = 2, 5, 128
    
    # Example modality features (already projected to d_model dimensions)
    poi_features = torch.randn(batch_size, seq_len, d_model)
    cat_features = torch.randn(batch_size, seq_len, d_model) 
    temporal_features = torch.randn(batch_size, seq_len, d_model)
    spatial_features = torch.randn(batch_size, seq_len, d_model)
    
    print("="*80)
    print("MODALITY FUSION COMPARISON")
    print("="*80)
    
    # Method 1: Simple Concatenation (traditional approach)
    print("\n1. SIMPLE CONCATENATION APPROACH:")
    print("-" * 40)
    
    concatenated = torch.cat([
        poi_features, cat_features, temporal_features, spatial_features
    ], dim=-1)
    
    print(f"Individual modality shapes: {poi_features.shape}")
    print(f"Concatenated shape: {concatenated.shape}")
    print(f"Total dimensions: {d_model * 4} = {concatenated.shape[-1]}")
    
    # Requires projection back to d_model
    concat_proj = nn.Linear(d_model * 4, d_model)
    concat_result = concat_proj(concatenated)
    print(f"After projection back to d_model: {concat_result.shape}")
    
    # Method 2: Learnable Weighted Fusion (this paper's approach)
    print("\n2. LEARNABLE WEIGHTED FUSION APPROACH:")
    print("-" * 40)
    
    # Learnable weights (initialized to 1s, will be learned during training)
    modality_weights = nn.Parameter(torch.ones(4))
    print(f"Initial weights: {modality_weights.data}")
    
    # Apply softmax to get normalized weights
    weights = torch.softmax(modality_weights, dim=0)
    print(f"Normalized weights: {weights.data}")
    
    # Weighted combination
    fused = (weights[0] * poi_features + 
             weights[1] * cat_features + 
             weights[2] * temporal_features + 
             weights[3] * spatial_features)
    
    print(f"Fused shape: {fused.shape}")
    print(f"Maintains d_model dimensions: {fused.shape[-1]} = {d_model}")
    
    # Method 3: Show what happens after training (simulate learned weights)
    print("\n3. AFTER TRAINING (SIMULATED LEARNED WEIGHTS):")
    print("-" * 40)
    
    # Simulate weights that might be learned for POI prediction
    # POI identity and temporal info might be more important
    learned_weights = torch.tensor([0.4, 0.2, 0.3, 0.1])  # POI, Cat, Temporal, Spatial
    print(f"Learned weights: {learned_weights}")
    print("Interpretation:")
    print(f"  - POI identity: {learned_weights[0]:.1%} importance")
    print(f"  - Category: {learned_weights[1]:.1%} importance") 
    print(f"  - Temporal: {learned_weights[2]:.1%} importance")
    print(f"  - Spatial: {learned_weights[3]:.1%} importance")
    
    fused_learned = (learned_weights[0] * poi_features + 
                     learned_weights[1] * cat_features + 
                     learned_weights[2] * temporal_features + 
                     learned_weights[3] * spatial_features)
    
    print(f"Fused with learned weights shape: {fused_learned.shape}")
    
    # Compare computational efficiency
    print("\n4. COMPUTATIONAL COMPARISON:")
    print("-" * 40)
    
    # Parameters count
    concat_params = d_model * 4 * d_model  # Linear layer parameters
    weighted_params = 4  # Just 4 learnable weights
    
    print(f"Concatenation method parameters: {concat_params:,}")
    print(f"Weighted fusion method parameters: {weighted_params}")
    print(f"Parameter reduction: {concat_params / weighted_params:.0f}x fewer parameters")
    
    # Memory usage
    concat_memory = batch_size * seq_len * d_model * 4  # Intermediate concatenated tensor
    weighted_memory = batch_size * seq_len * d_model  # No intermediate expansion
    
    print(f"Concatenation intermediate memory: {concat_memory:,} elements")
    print(f"Weighted fusion memory: {weighted_memory:,} elements")
    print(f"Memory reduction: {concat_memory / weighted_memory:.0f}x less memory")
    
    print("\n5. KEY ADVANTAGES OF LEARNABLE WEIGHTED FUSION:")
    print("-" * 40)
    print("✓ Learns optimal importance of each modality automatically")
    print("✓ Adapts weights based on the specific prediction task")
    print("✓ No parameter explosion (4 weights vs 65,536 parameters)")
    print("✓ No memory overhead from concatenation")
    print("✓ Maintains fixed d_model dimensionality")
    print("✓ Allows interpretability of modality importance")
    
    return fused_learned, weights

def visualize_weight_evolution():
    """
    Visualize how modality weights might evolve during training
    """
    # Simulate weight evolution during training
    epochs = 100
    
    # Different scenarios
    scenarios = {
        "Morning Commute": np.array([
            [0.25, 0.25, 0.25, 0.25],  # Start equal
            [0.2, 0.15, 0.5, 0.15],    # Temporal becomes important
            [0.15, 0.1, 0.6, 0.15],    # Even more temporal focus
            [0.1, 0.05, 0.7, 0.15]     # Final: time matters most
        ]),
        "Leisure Weekend": np.array([
            [0.25, 0.25, 0.25, 0.25],  # Start equal  
            [0.4, 0.3, 0.15, 0.15],    # POI and category matter more
            [0.5, 0.35, 0.1, 0.05],    # Focus on what/where
            [0.6, 0.3, 0.05, 0.05]     # Final: POI identity dominates
        ]),
        "Business Trip": np.array([
            [0.25, 0.25, 0.25, 0.25],  # Start equal
            [0.15, 0.4, 0.2, 0.25],    # Category becomes important
            [0.1, 0.5, 0.15, 0.25],    # Business categories
            [0.05, 0.6, 0.1, 0.25]     # Final: category-driven
        ])
    }
    
    print("\n6. WEIGHT EVOLUTION SCENARIOS:")
    print("-" * 40)
    
    for scenario_name, weights in scenarios.items():
        print(f"\n{scenario_name}:")
        print("  Initial → Final weights:")
        print(f"  POI:      {weights[0,0]:.2f} → {weights[-1,0]:.2f}")
        print(f"  Category: {weights[0,1]:.2f} → {weights[-1,1]:.2f}")  
        print(f"  Temporal: {weights[0,2]:.2f} → {weights[-1,2]:.2f}")
        print(f"  Spatial:  {weights[0,3]:.2f} → {weights[-1,3]:.2f}")

if __name__ == "__main__":
    # Run the demonstration
    fused_result, final_weights = demonstrate_modality_fusion()
    visualize_weight_evolution()
    
    print("\n" + "="*80)
    print("MATHEMATICAL FORMULATION:")
    print("="*80)
    print("\nSimple Concatenation:")
    print("  fused = Linear_proj([E_poi; E_cat; E_temp; E_spatial])")
    print("  Parameters: O(d_model²)")
    print("  Memory: O(4 × d_model)")
    
    print("\nLearnable Weighted Fusion:")
    print("  α = softmax([w₁, w₂, w₃, w₄])")
    print("  fused = α₁×E_poi + α₂×E_cat + α₃×E_temp + α₄×E_spatial")
    print("  Parameters: O(1)")
    print("  Memory: O(d_model)")
    print("\nwhere α sums to 1 and is learned end-to-end!")
