"""
Training pipeline for POI Transformer
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR
import wandb
from tqdm import tqdm
import os
import json
from typing import Dict, Optional, Tuple
import numpy as np
from pathlib import Path

from ..config.model_config import ModelConfig, TrainingConfig
from ..models.poi_transformer import POITransformerWithCrossAttention
from ..data.data_structures import BatchData, ModelTargets, ModelOutput
from .metrics import POIMetrics


class POITrainer:
    """Trainer for POI Transformer with Cross-Attention"""
    
    def __init__(self, 
                 model: POITransformerWithCrossAttention,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 config: TrainingConfig,
                 model_config: ModelConfig,
                 device: torch.device):
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.model_config = model_config
        self.device = device
        
        # Move model to device
        self.model.to(device)
        
        # Setup optimizer
        self.optimizer = self._setup_optimizer()
        
        # Setup learning rate scheduler
        self.scheduler = self._setup_scheduler()
        
        # Setup metrics
        self.metrics = POIMetrics(model_config.num_categories)
        
        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        
        # Setup logging
        self._setup_logging()
        
        # Create output directories
        self.config.output_dir.mkdir(parents=True, exist_ok=True)
        self.config.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    def _setup_optimizer(self) -> optim.Optimizer:
        """Setup optimizer with different learning rates for different components"""
        # Separate parameters for different learning rates
        embedding_params = []
        transformer_params = []
        output_params = []
        
        for name, param in self.model.named_parameters():
            if 'embedding' in name or 'pos_encoding' in name:
                embedding_params.append(param)
            elif 'transformer_blocks' in name:
                transformer_params.append(param)
            else:
                output_params.append(param)
        
        # Use different learning rates
        optimizer = optim.AdamW([
            {'params': embedding_params, 'lr': self.config.learning_rate * 0.5},
            {'params': transformer_params, 'lr': self.config.learning_rate},
            {'params': output_params, 'lr': self.config.learning_rate * 1.5}
        ], weight_decay=self.config.weight_decay)
        
        return optimizer
    
    def _setup_scheduler(self) -> optim.lr_scheduler._LRScheduler:
        """Setup learning rate scheduler"""
        # Warmup scheduler
        warmup_scheduler = LinearLR(
            self.optimizer,
            start_factor=0.1,
            total_iters=self.config.warmup_steps
        )
        
        # Main scheduler
        cosine_scheduler = CosineAnnealingLR(
            self.optimizer,
            T_max=self.config.num_epochs - self.config.warmup_steps // len(self.train_loader),
            eta_min=self.config.learning_rate * 0.01
        )
        
        # Combine schedulers
        scheduler = SequentialLR(
            self.optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[self.config.warmup_steps]
        )
        
        return scheduler
    
    def _setup_logging(self):
        """Setup logging with wandb if enabled"""
        if self.config.use_wandb:
            wandb.init(
                project=self.config.wandb_project,
                entity=self.config.wandb_entity,
                config={
                    'model_config': self.model_config.__dict__,
                    'training_config': self.config.__dict__
                }
            )
    
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch"""
        self.model.train()
        epoch_losses = {'total': 0.0, 'poi': 0.0, 'category': 0.0, 'time': 0.0}
        num_batches = len(self.train_loader)
        
        progress_bar = tqdm(self.train_loader, desc=f'Epoch {self.current_epoch}')
        
        for batch_idx, (batch_data, targets) in enumerate(progress_bar):
            # Move data to device
            batch_data = batch_data.to(self.device)
            targets = targets.to(self.device)
            
            # Forward pass
            outputs = self.model(batch_data)
            
            # Compute losses
            losses = self.model.compute_loss(outputs, targets)
            
            # Backward pass
            self.optimizer.zero_grad()
            losses['total_loss'].backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Update parameters
            self.optimizer.step()
            
            # Update learning rate
            if self.global_step < self.config.warmup_steps:
                self.scheduler.step()
            
            # Update metrics
            for key, value in losses.items():
                if key.endswith('_loss'):
                    key_short = key.replace('_loss', '')
                    if key_short in epoch_losses:
                        epoch_losses[key_short] += value.item()
            
            # Log batch metrics
            if self.global_step % self.config.log_interval == 0:
                self._log_batch_metrics(losses, batch_idx, num_batches)
            
            # Validation
            if self.global_step % self.config.val_interval == 0:
                val_metrics = self.validate()
                self._log_validation_metrics(val_metrics)
                self.model.train()  # Return to training mode
            
            # Save checkpoint
            if self.global_step % self.config.save_interval == 0:
                self._save_checkpoint(f'checkpoint_step_{self.global_step}.pt')
            
            self.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                'loss': f"{losses['total_loss'].item():.4f}",
                'lr': f"{self.optimizer.param_groups[0]['lr']:.2e}"
            })
        
        # Average losses over epoch
        for key in epoch_losses:
            epoch_losses[key] /= num_batches
        
        return epoch_losses
    
    def validate(self) -> Dict[str, float]:
        """Validate the model"""
        self.model.eval()
        val_losses = {'total': 0.0, 'poi': 0.0, 'category': 0.0, 'time': 0.0}
        all_predictions = []
        all_targets = []
        
        with torch.no_grad():
            for batch_data, targets in tqdm(self.val_loader, desc='Validation'):
                # Move data to device
                batch_data = batch_data.to(self.device)
                targets = targets.to(self.device)
                
                # Forward pass
                outputs = self.model(batch_data)
                
                # Compute losses
                losses = self.model.compute_loss(outputs, targets)
                
                # Accumulate losses
                for key, value in losses.items():
                    if key.endswith('_loss'):
                        key_short = key.replace('_loss', '')
                        if key_short in val_losses:
                            val_losses[key_short] += value.item()
                
                # Collect predictions for metrics
                all_predictions.append(outputs)
                all_targets.append(targets)
        
        # Average losses
        num_batches = len(self.val_loader)
        for key in val_losses:
            val_losses[key] /= num_batches
        
        # Compute detailed metrics
        detailed_metrics = self.metrics.compute_metrics(all_predictions, all_targets)
        val_losses.update(detailed_metrics)
        
        return val_losses
    
    def train(self) -> Dict[str, list]:
        """Main training loop"""
        print(f"Starting training for {self.config.num_epochs} epochs")
        print(f"Training on device: {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters()):,}")
        
        train_history = {'loss': [], 'poi_loss': [], 'category_loss': [], 'time_loss': []}
        val_history = {'loss': [], 'poi_loss': [], 'category_loss': [], 'time_loss': []}
        
        for epoch in range(self.config.num_epochs):
            self.current_epoch = epoch
            
            # Train epoch
            train_metrics = self.train_epoch()
            train_history['loss'].append(train_metrics['total'])
            train_history['poi_loss'].append(train_metrics.get('poi', 0.0))
            train_history['category_loss'].append(train_metrics.get('category', 0.0))
            train_history['time_loss'].append(train_metrics.get('time', 0.0))
            
            # Validate
            val_metrics = self.validate()
            val_history['loss'].append(val_metrics['total'])
            val_history['poi_loss'].append(val_metrics.get('poi', 0.0))
            val_history['category_loss'].append(val_metrics.get('category', 0.0))
            val_history['time_loss'].append(val_metrics.get('time', 0.0))
            
            # Update scheduler (for epochs after warmup)
            if self.global_step >= self.config.warmup_steps:
                self.scheduler.step()
            
            # Log epoch metrics
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            print(f"Train Loss: {train_metrics['total']:.4f}")
            print(f"Val Loss: {val_metrics['total']:.4f}")
            
            # Early stopping check
            if val_metrics['total'] < self.best_val_loss:
                self.best_val_loss = val_metrics['total']
                self.patience_counter = 0
                self._save_checkpoint('best_model.pt')
            else:
                self.patience_counter += 1
                
            if self.patience_counter >= self.config.patience:
                print(f"Early stopping triggered after {epoch + 1} epochs")
                break
            
            # Log to wandb
            if self.config.use_wandb:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_metrics['total'],
                    'val_loss': val_metrics['total'],
                    'learning_rate': self.optimizer.param_groups[0]['lr']
                })
        
        # Save final model
        self._save_checkpoint('final_model.pt')
        
        return {'train': train_history, 'val': val_history}
    
    def _log_batch_metrics(self, losses: Dict[str, torch.Tensor], batch_idx: int, num_batches: int):
        """Log batch-level metrics"""
        if self.config.use_wandb:
            log_dict = {
                f'batch_{k}': v.item() if isinstance(v, torch.Tensor) else v 
                for k, v in losses.items()
            }
            log_dict['step'] = self.global_step
            log_dict['batch_progress'] = batch_idx / num_batches
            wandb.log(log_dict)
    
    def _log_validation_metrics(self, metrics: Dict[str, float]):
        """Log validation metrics"""
        if self.config.use_wandb:
            log_dict = {f'val_{k}': v for k, v in metrics.items()}
            log_dict['step'] = self.global_step
            wandb.log(log_dict)
    
    def _save_checkpoint(self, filename: str):
        """Save model checkpoint"""
        checkpoint = {
            'epoch': self.current_epoch,
            'global_step': self.global_step,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'model_config': self.model_config.__dict__,
            'training_config': self.config.__dict__
        }
        
        filepath = self.config.checkpoint_dir / filename
        torch.save(checkpoint, filepath)
        print(f"Checkpoint saved: {filepath}")
    
    def load_checkpoint(self, filepath: str) -> bool:
        """Load model checkpoint"""
        try:
            checkpoint = torch.load(filepath, map_location=self.device)
            
            self.model.load_state_dict(checkpoint['model_state_dict'])
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            
            self.current_epoch = checkpoint['epoch']
            self.global_step = checkpoint['global_step']
            self.best_val_loss = checkpoint['best_val_loss']
            
            print(f"Checkpoint loaded from {filepath}")
            return True
            
        except Exception as e:
            print(f"Failed to load checkpoint from {filepath}: {e}")
            return False


def get_device(config: TrainingConfig) -> torch.device:
    """Get the appropriate device for training"""
    if config.device == "auto":
        if torch.cuda.is_available():
            device = torch.device("cuda")
        elif torch.backends.mps.is_available():
            device = torch.device("mps")
        else:
            device = torch.device("cpu")
    else:
        device = torch.device(config.device)
    
    print(f"Using device: {device}")
    return device
