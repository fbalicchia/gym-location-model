import torch
import numpy as np
from pathlib import Path

from poi_prediction.config.model_config import ModelConfig, TrainingConfig, DataConfig
from poi_prediction.models.poi_transformer import POITransformerWithCrossAttention
from poi_prediction.data.dataset import create_data_loaders
from poi_prediction.data.synthetic_generator import AdvancedPOIDataGenerator
from poi_prediction.training.trainer import POITrainer, get_device


def run_demo():
    """Run a quick demo of the POI transformer"""
    print("🚀 POI Transformer with Cross-Attention Demo")
    print("=" * 50)
    
    # Configuration for demo (smaller scale)
    model_config = ModelConfig(
        d_model=64,  # Smaller for demo
        nhead=4,
        num_layers=2,
        dim_feedforward=128,
        dropout=0.1,
        max_sequence_length=20,
        num_pois=26,  # Will be updated based on generator
        num_categories=13,
        num_users=100,  # Smaller for demo
        use_cross_attention=True,
        cross_attention_every_n_layers=1  # Use cross-attention in every layer
    )
    
    training_config = TrainingConfig(
        learning_rate=1e-3,
        batch_size=16,  # Smaller batch size
        num_epochs=5,   # Just 5 epochs for demo
        patience=3,
        device='auto',
        output_dir=Path('./poi_prediction_outputs'),
        checkpoint_dir=Path('./poi_prediction_checkpoints'),
        use_wandb=False  # Disable wandb for demo
    )
    
    data_config = DataConfig(
        min_sequence_length=5,
        max_sequence_length=20,
        num_negative_samples=20  # Smaller for demo
    )
    
    print(f"📊 Model Configuration:")
    print(f"   - Model dimension: {model_config.d_model}")
    print(f"   - Attention heads: {model_config.nhead}")
    print(f"   - Transformer layers: {model_config.num_layers}")
    print(f"   - Cross-attention enabled: {model_config.use_cross_attention}")
    print(f"   - Max sequence length: {model_config.max_sequence_length}")
    
    # Get device
    device = get_device(training_config)
    
    # Create data loaders
    print(f"\n📁 Creating synthetic dataset...")
    train_loader, val_loader, generator = create_data_loaders(
        data_config, 
        model_config,
        batch_size=training_config.batch_size,
        num_workers=0  # Single-threaded for demo
    )
    
    # Update model config with actual data statistics
    model_config.num_pois = len(generator.poi_database)
    model_config.num_categories = len(generator.categories)
    
    print(f"   - Number of POIs: {model_config.num_pois}")
    print(f"   - Number of categories: {model_config.num_categories}")
    print(f"   - Training samples: {len(train_loader.dataset)}")
    print(f"   - Validation samples: {len(val_loader.dataset)}")
    
    # Create model
    print(f"\n🧠 Creating POI Transformer model...")
    model = POITransformerWithCrossAttention(model_config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"   - Model parameters: {num_params:,}")
    
    # Test model with a sample batch
    print(f"\n🔍 Testing model with sample batch...")
    sample_batch, sample_targets = next(iter(train_loader))
    sample_batch = sample_batch.to(device)
    sample_targets = sample_targets.to(device)
    
    model.to(device)
    model.eval()
    
    with torch.no_grad():
        outputs = model(sample_batch, return_attention=True)
        losses = model.compute_loss(outputs, sample_targets)
    
    print(f"   - Sample batch size: {sample_batch.poi_ids.shape}")
    print(f"   - POI scores shape: {outputs.poi_scores.shape if outputs.poi_scores is not None else 'None'}")
    print(f"   - Category logits shape: {outputs.category_logits.shape if outputs.category_logits is not None else 'None'}")
    print(f"   - Duration predictions shape: {outputs.duration_predictions.shape if outputs.duration_predictions is not None else 'None'}")
    print(f"   - Total loss: {losses['total_loss'].item():.4f}")
    
    # Test prediction functionality
    print(f"\n🎯 Testing prediction functionality...")
    predictions = model.predict_next_poi(sample_batch, top_k=5)
    
    if 'top_poi_indices' in predictions:
        print(f"   - Top POI predictions shape: {predictions['top_poi_indices'].shape}")
    if 'top_category_indices' in predictions:
        print(f"   - Top category predictions shape: {predictions['top_category_indices'].shape}")
    if 'predicted_durations' in predictions:
        print(f"   - Duration predictions shape: {predictions['predicted_durations'].shape}")
    
    # Analyze attention patterns
    if outputs.attention_weights:
        print(f"\n👁️ Analyzing attention patterns...")
        print(f"   - Number of transformer layers with attention: {len(outputs.attention_weights)}")
        
        for i, layer_attn in enumerate(outputs.attention_weights):
            if 'self_attention' in layer_attn and layer_attn['self_attention'] is not None:
                attn_shape = layer_attn['self_attention'].shape
                print(f"   - Layer {i} self-attention shape: {attn_shape}")
            
            if 'user_attention' in layer_attn and layer_attn['user_attention'] is not None:
                user_attn_shape = layer_attn['user_attention'].shape
                print(f"   - Layer {i} user cross-attention shape: {user_attn_shape}")
    
    # Demo some user and POI information
    print(f"\n🏢 Sample POI Information:")
    for poi_id in [0, 1, 2, 3, 4]:
        poi_features = generator.get_poi_features(poi_id)
        poi_obj = generator.poi_database[poi_id]
        print(f"   - {poi_obj.name}: category={poi_obj.category}, region={poi_obj.region_id}, "
              f"coords=({poi_obj.latitude:.3f}, {poi_obj.longitude:.3f})")
    
    print(f"\n👤 Sample User Profiles:")
    for user_id in range(min(3, len(generator.user_profiles))):
        profile = generator.user_profiles[user_id]
        print(f"   - User {user_id}: age_group={profile.age_group}, "
              f"preferred_categories={profile.preferred_categories[:3]}...")
    
    # Quick training demo (optional)
    run_training = input(f"\n🎓 Run quick training demo? (y/n): ").lower().strip() == 'y'
    
    if run_training:
        print(f"\n🎓 Running quick training demo...")
        trainer = POITrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=training_config,
            model_config=model_config,
            device=device
        )
        
        try:
            history = trainer.train()
            print(f"✅ Training completed!")
            print(f"   - Final train loss: {history['train']['loss'][-1]:.4f}")
            print(f"   - Final val loss: {history['val']['loss'][-1]:.4f}")
            
        except KeyboardInterrupt:
            print(f"\n⏹️ Training interrupted by user")
        except Exception as e:
            print(f"\n❌ Training error: {e}")
    
    print(f"\n🎉 Demo completed!")
    print(f"\nKey Features Demonstrated:")
    print(f"✓ Multi-modal embeddings (POI, category, time, location, region)")
    print(f"✓ Cross-attention mechanisms (user context, geographical context)")
    print(f"✓ Multi-task learning (POI ranking, category prediction, time regression)")
    print(f"✓ Synthetic data generation with realistic patterns")
    print(f"✓ Attention pattern analysis")
    print(f"✓ Training pipeline with comprehensive metrics")


if __name__ == '__main__':
    run_demo()
