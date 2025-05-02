"""Trainer for the Enhanced MAGNET model."""
import os
import time
import torch
from tqdm import tqdm
import numpy as np
from training.loss import compute_total_loss

class Trainer:
    """Trainer for the Enhanced MAGNET model."""
    def __init__(self, model, config, train_loader, val_loader=None):
        """
        Initialize the trainer.
        
        Args:
            model: EnhancedMAGNET model
            config: Configuration object
            train_loader: DataLoader for training data
            val_loader: DataLoader for validation data
        """
        self.model = model
        self.config = config
        self.train_loader = train_loader
        self.val_loader = val_loader
        
        # Move model to device
        self.model = self.model.to(config.device)
        
        # Set up optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
        
        # Set up learning rate scheduler with warmup
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer,
            lr_lambda=lambda step: min(1.0, step / config.warmup_steps)
        )
        
        # Initialize temperature for Gumbel-Sigmoid
        self.temperature = config.initial_temperature
        
        # Create output directory if it doesn't exist
        os.makedirs(config.output_dir, exist_ok=True)
        
        # Initialize tracking variables
        self.global_step = 0
        self.best_val_loss = float('inf')
        
    def train(self, num_epochs=None):
        """
        Train the model.
        
        Args:
            num_epochs: Number of epochs to train for (defaults to config value)
        """
        if num_epochs is None:
            num_epochs = self.config.num_epochs
            
        # Training loop
        for epoch in range(num_epochs):
            print(f"Epoch {epoch+1}/{num_epochs}")
            
            # Train for one epoch
            train_metrics = self.train_epoch()
            
            # Validate model
            if self.val_loader is not None:
                val_metrics = self.validate()
                
                # Save model if it's the best so far
                if val_metrics['total_loss'] < self.best_val_loss:
                    self.best_val_loss = val_metrics['total_loss']
                    self.save_model(os.path.join(self.config.output_dir, "best_model.pt"))
                    print(f"New best model saved with validation loss: {val_metrics['total_loss']:.4f}")
            
            # Save model checkpoint
            self.save_model(os.path.join(self.config.output_dir, f"checkpoint_epoch_{epoch+1}.pt"))
            
            # Print epoch summary
            print(f"Epoch {epoch+1} completed:")
            print(f"  Training loss: {train_metrics['total_loss']:.4f}")
            if self.val_loader is not None:
                print(f"  Validation loss: {val_metrics['total_loss']:.4f}")
            print(f"  Current LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            print(f"  Gumbel temperature: {self.temperature:.4f}")
    
    def train_epoch(self):
        """
        Train model for one epoch.
        
        Returns:
            Dictionary of training metrics
        """
        self.model.train()
        
        epoch_losses = {
            'total_loss': 0.0,
            'lm_loss': 0.0,
            'boundary_loss': 0.0,
            'morph_loss': 0.0
        }
        
        start_time = time.time()
        num_batches = len(self.train_loader)
        
        for batch_idx, batch in enumerate(tqdm(self.train_loader, desc="Training")):
            # Move batch to device
            batch = {k: v.to(self.config.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            self.optimizer.zero_grad()
            model_outputs = self.model(batch, temperature=self.temperature)
            
            # Compute loss
            losses = compute_total_loss(model_outputs, batch, self.config)
            total_loss = losses['total_loss']
            
            # Backward pass
            total_loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip_grad_norm)
            
            # Update parameters
            self.optimizer.step()
            self.scheduler.step()
            
            # Update temperature for Gumbel-Sigmoid
            self.temperature = max(
                self.config.min_temperature,
                self.temperature * self.config.annealing_factor
            )
            
            # Update tracking variables
            self.global_step += 1
            
            # Accumulate losses
            for k in epoch_losses:
                epoch_losses[k] += losses[k].item()
            
            # Log progress
            if self.global_step % self.config.log_steps == 0:
                lr = self.optimizer.param_groups[0]['lr']
                print(f"Step {self.global_step}: total_loss={total_loss.item():.4f}, "
                      f"lm_loss={losses['lm_loss'].item():.4f}, "
                      f"boundary_loss={losses['boundary_loss'].item():.4f}, "
                      f"morph_loss={losses['morph_loss'].item():.4f}, "
                      f"lr={lr:.2e}, temp={self.temperature:.4f}")
            
            # Save checkpoint
            if self.global_step % self.config.save_steps == 0:
                self.save_model(os.path.join(self.config.output_dir, f"step_{self.global_step}.pt"))
        
        # Calculate average losses
        for k in epoch_losses:
            epoch_losses[k] /= num_batches
        
        # Print epoch summary
        elapsed = time.time() - start_time
        print(f"Epoch completed in {elapsed:.2f}s. "
              f"Avg loss: {epoch_losses['total_loss']:.4f}")
        
        return epoch_losses
    
    def validate(self):
        """
        Validate the model on the validation set.
        
        Returns:
            Dictionary of validation metrics
        """
        self.model.eval()
        
        val_losses = {
            'total_loss': 0.0,
            'lm_loss': 0.0,
            'boundary_loss': 0.0,
            'morph_loss': 0.0
        }
        
        num_batches = len(self.val_loader)
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="Validating"):
                # Move batch to device
                batch = {k: v.to(self.config.device) if isinstance(v, torch.Tensor) else v 
                        for k, v in batch.items()}
                
                # Forward pass
                model_outputs = self.model(batch, temperature=self.temperature)
                
                # Compute loss
                losses = compute_total_loss(model_outputs, batch, self.config)
                
                # Accumulate losses
                for k in val_losses:
                    val_losses[k] += losses[k].item()
        
        # Calculate average losses
        for k in val_losses:
            val_losses[k] /= num_batches
        
        # Print validation summary
        print(f"Validation results: "
              f"total_loss={val_losses['total_loss']:.4f}, "
              f"lm_loss={val_losses['lm_loss']:.4f}, "
              f"boundary_loss={val_losses['boundary_loss']:.4f}, "
              f"morph_loss={val_losses['morph_loss']:.4f}")
        
        return val_losses
    
    def save_model(self, path):
        """Save model checkpoint to file."""
        checkpoint = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'global_step': self.global_step,
            'temperature': self.temperature,
            'config': self.config
        }
        torch.save(checkpoint, path)
        print(f"Model saved to {path}")
    
    def load_model(self, path):
        """Load model checkpoint from file."""
        checkpoint = torch.load(path, map_location=self.config.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        self.global_step = checkpoint['global_step']
        self.temperature = checkpoint['temperature']
        
        print(f"Model loaded from {path} at step {self.global_step}")   