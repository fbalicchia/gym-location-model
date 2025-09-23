"""
Command line interface for POI prediction training and inference
"""

import argparse
import torch
from pathlib import Path
import json
import sys

from .config.model_config import ModelConfig, TrainingConfig, DataConfig, get_config
from .models.poi_transformer import POITransformerWithCrossAttention
from .data.dataset import create_data_loaders
from .training.trainer import POITrainer, get_device
from .training.metrics import evaluate_model_predictions


def train_cli():
    """Main training CLI entry point"""
    parser = argparse.ArgumentParser(description='Train POI Transformer with Cross-Attention')
    
    # Model arguments
    parser.add_argument('--d_model', type=int, default=128, help='Model dimension')
    parser.add_argument('--nhead', type=int, default=8, help='Number of attention heads')
    parser.add_argument('--num_layers', type=int, default=4, help='Number of transformer layers')
    parser.add_argument('--dropout', type=float, default=0.1, help='Dropout rate')
    
    # Training arguments
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--learning_rate', type=float, default=1e-3, help='Learning rate')
    parser.add_argument('--num_epochs', type=int, default=50, help='Number of epochs')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Device to use for training')
    
    # Data arguments
    parser.add_argument('--num_users', type=int, default=1000, help='Number of synthetic users')
    parser.add_argument('--max_seq_length', type=int, default=30, help='Maximum sequence length')
    
    # Logging and output
    parser.add_argument('--output_dir', type=str, default='./outputs', help='Output directory')
    parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='Checkpoint directory')
    parser.add_argument('--use_wandb', action='store_true', help='Use Weights & Biases logging')
    parser.add_argument('--wandb_project', type=str, default='poi-prediction', help='W&B project name')
    
    # Configuration file
    parser.add_argument('--config', type=str, help='Path to configuration JSON file')
    
    args = parser.parse_args()
    
    # Load configuration
    if args.config and Path(args.config).exists():
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        # Override with command line arguments
        for key, value in vars(args).items():
            if value is not None:
                config_dict[key] = value
    else:
        config_dict = vars(args)
    
    # Create configuration objects
    config = get_config()
    
    # Update model config
    model_config = config['model']
    model_config.d_model = config_dict.get('d_model', model_config.d_model)
    model_config.nhead = config_dict.get('nhead', model_config.nhead)
    model_config.num_layers = config_dict.get('num_layers', model_config.num_layers)
    model_config.dropout = config_dict.get('dropout', model_config.dropout)
    model_config.num_users = config_dict.get('num_users', model_config.num_users)
    model_config.max_sequence_length = config_dict.get('max_seq_length', model_config.max_sequence_length)
    
    # Update training config
    training_config = config['training']
    training_config.batch_size = config_dict.get('batch_size', training_config.batch_size)
    training_config.learning_rate = config_dict.get('learning_rate', training_config.learning_rate)
    training_config.num_epochs = config_dict.get('num_epochs', training_config.num_epochs)
    training_config.device = config_dict.get('device', training_config.device)
    training_config.output_dir = Path(config_dict.get('output_dir', training_config.output_dir))
    training_config.checkpoint_dir = Path(config_dict.get('checkpoint_dir', training_config.checkpoint_dir))
    training_config.use_wandb = config_dict.get('use_wandb', training_config.use_wandb)
    training_config.wandb_project = config_dict.get('wandb_project', training_config.wandb_project)
    
    # Update data config
    data_config = config['data']
    data_config.max_sequence_length = config_dict.get('max_seq_length', data_config.max_sequence_length)
    
    print("=== POI Transformer Training ===")
    print(f"Model dimension: {model_config.d_model}")
    print(f"Attention heads: {model_config.nhead}")
    print(f"Transformer layers: {model_config.num_layers}")
    print(f"Max sequence length: {model_config.max_sequence_length}")
    print(f"Batch size: {training_config.batch_size}")
    print(f"Learning rate: {training_config.learning_rate}")
    print(f"Number of epochs: {training_config.num_epochs}")
    print(f"Using cross-attention: {model_config.use_cross_attention}")
    
    # Get device
    device = get_device(training_config)
    
    # Create data loaders
    print("\nCreating data loaders...")
    train_loader, val_loader, generator = create_data_loaders(
        data_config, 
        model_config,
        batch_size=training_config.batch_size,
        num_workers=4
    )
    
    # Update model config with actual data statistics
    model_config.num_pois = len(generator.poi_database)
    model_config.num_categories = len(generator.categories)
    
    print(f"Number of POIs: {model_config.num_pois}")
    print(f"Number of categories: {model_config.num_categories}")
    
    # Create model
    print("\nCreating model...")
    model = POITransformerWithCrossAttention(model_config)
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {num_params:,}")
    
    # Create trainer
    print("\nInitializing trainer...")
    trainer = POITrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=training_config,
        model_config=model_config,
        device=device
    )
    
    # Save configuration
    config_save_path = training_config.output_dir / 'config.json'
    with open(config_save_path, 'w') as f:
        json.dump({
            'model_config': model_config.__dict__,
            'training_config': training_config.__dict__,
            'data_config': data_config.__dict__
        }, f, indent=2, default=str)
    print(f"Configuration saved to: {config_save_path}")
    
    # Start training
    print("\nStarting training...")
    try:
        history = trainer.train()
        print("✅ Training completed successfully!")
        
        # Save training history
        history_path = training_config.output_dir / 'training_history.json'
        with open(history_path, 'w') as f:
            json.dump(history, f, indent=2)
        print(f"Training history saved to: {history_path}")
        
    except KeyboardInterrupt:
        print("\n❌ Training interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Training failed with error: {e}")
        sys.exit(1)


def predict_cli():
    """CLI for making predictions with trained model"""
    parser = argparse.ArgumentParser(description='Make predictions with trained POI Transformer')
    
    parser.add_argument('--checkpoint', type=str, required=True, 
                       help='Path to trained model checkpoint')
    parser.add_argument('--config', type=str, 
                       help='Path to model configuration (if not in checkpoint)')
    parser.add_argument('--output_dir', type=str, default='./predictions',
                       help='Directory to save predictions')
    parser.add_argument('--num_samples', type=int, default=100,
                       help='Number of samples to predict')
    parser.add_argument('--top_k', type=int, default=10,
                       help='Number of top predictions to return')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda', 'mps'],
                       help='Device to use for inference')
    
    args = parser.parse_args()
    
    print("=== POI Transformer Prediction ===")
    
    # Load checkpoint
    print(f"Loading checkpoint from: {args.checkpoint}")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # Load configuration
    if 'model_config' in checkpoint:
        model_config_dict = checkpoint['model_config']
        model_config = ModelConfig(**model_config_dict)
    elif args.config:
        with open(args.config, 'r') as f:
            config_dict = json.load(f)
        model_config = ModelConfig(**config_dict['model_config'])
    else:
        raise ValueError("Model configuration not found in checkpoint and not provided")
    
    # Get device
    device = torch.device(args.device if args.device != 'auto' else 
                         ('cuda' if torch.cuda.is_available() else 
                          'mps' if torch.backends.mps.is_available() else 'cpu'))
    print(f"Using device: {device}")
    
    # Create model and load weights
    print("Creating model...")
    model = POITransformerWithCrossAttention(model_config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create data loader for prediction
    print("Creating data loader...")
    data_config = DataConfig()
    train_loader, val_loader, generator = create_data_loaders(
        data_config, model_config, batch_size=32, num_workers=2
    )
    
    # Generate predictions
    print(f"Generating predictions for {args.num_samples} samples...")
    predictions = evaluate_model_predictions(
        model, val_loader, device, top_k=args.top_k
    )
    
    # Save predictions
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    import numpy as np
    predictions_file = output_dir / 'predictions.npz'
    np.savez(predictions_file, **predictions)
    print(f"Predictions saved to: {predictions_file}")
    
    # Print summary statistics
    print("\n=== Prediction Summary ===")
    if len(predictions['poi_predictions']) > 0:
        print(f"POI predictions: {len(predictions['poi_predictions'])} samples")
    if len(predictions['category_predictions']) > 0:
        print(f"Category predictions: {len(predictions['category_predictions'])} samples")
    if len(predictions['duration_predictions']) > 0:
        print(f"Duration predictions: {len(predictions['duration_predictions'])} samples")
        mean_duration = np.mean(predictions['duration_predictions'])
        print(f"Mean predicted duration: {mean_duration:.2f} hours")
    
    print("✅ Prediction completed successfully!")


def main():
    """Main entry point - route to appropriate CLI"""
    if len(sys.argv) > 1 and sys.argv[1] == 'predict':
        sys.argv.pop(1)  # Remove 'predict' from args
        predict_cli()
    else:
        train_cli()


if __name__ == '__main__':
    main()
