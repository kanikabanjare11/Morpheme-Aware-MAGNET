"""Evaluation script for the Enhanced MAGNET model."""
import argparse
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
import numpy as np

from config import Config
from data import create_dataloaders
from model import magnet
from tools.morfessor_util import MorfessorWrapper, detect_language
from utils import boundary_f1_score, byte_accuracy

def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate Enhanced MAGNET model")
    parser.add_argument("--model_path", type=str, required=True, help="/root/Morpheme-Aware MAGNET/model_path")
    parser.add_argument("--output_dir", type=str, default="evaluation_results", help="/root/Morpheme-Aware MAGNET/output")
    parser.add_argument("--batch_size", type=int, default=None, help="32")
    parser.add_argument("--device", type=str, default=None, help="Device for evaluation (cuda or cpu)")
    return parser.parse_args()

def evaluate_model(model, data_loader, config):
    """
    Evaluate model on a dataset.
    
    Args:
        model: EnhancedMAGNET model
        data_loader: DataLoader for evaluation
        config: Configuration object
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    
    total_byte_acc = 0.0
    total_boundary_metrics = {'precision': 0.0, 'recall': 0.0, 'f1': 0.0}
    total_loss = 0.0
    
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Evaluating"):
            # Move batch to device
            batch = {k: v.to(config.device) if isinstance(v, torch.Tensor) else v 
                    for k, v in batch.items()}
            
            # Forward pass
            outputs = model(batch, temperature=0.1)  # Low temperature for more discrete boundaries
            
            # Get predictions
            logits = outputs['logits']
            boundary_probs = outputs['boundary_probs']
            loss_mask = outputs['loss_mask']
            
            # Shift targets for next-byte prediction
            shifted_target = torch.zeros_like(batch['input_bytes'])
            shifted_target[:, :-1] = batch['input_bytes'][:, 1:]
            
            # Calculate metrics
            byte_acc = byte_accuracy(logits, batch['input_bytes'], loss_mask)
            boundary_metrics = boundary_f1_score(boundary_probs, batch['gold_boundaries'], loss_mask)
            
            # Compute loss (for reference)
            lm_loss = F.cross_entropy(
                logits.reshape(-1, 256), 
                shifted_target.reshape(-1), 
                reduction='none'
            )
            lm_loss = (lm_loss.reshape(loss_mask.shape) * loss_mask).sum() / loss_mask.sum().clamp(min=1)
            
            # Accumulate metrics
            total_byte_acc += byte_acc.item()
            for key in total_boundary_metrics:
                total_boundary_metrics[key] += boundary_metrics[key]
            total_loss += lm_loss.item()
    
    # Calculate average metrics
    num_batches = len(data_loader)
    avg_byte_acc = total_byte_acc / num_batches
    avg_boundary_metrics = {k: v / num_batches for k, v in total_boundary_metrics.items()}
    avg_loss = total_loss / num_batches
    
    return {
        'byte_accuracy': avg_byte_acc,
        'boundary_precision': avg_boundary_metrics['precision'],
        'boundary_recall': avg_boundary_metrics['recall'],
        'boundary_f1': avg_boundary_metrics['f1'],
        'loss': avg_loss
    }

def main():
    # Parse command-line arguments
    args = parse_args()
    
    # Load configuration
    config = Config()
    if args.batch_size:
        config.batch_size = args.batch_size
    if args.device:
        config.device = args.device
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create data loaders
    _, _, test_loader, morfessor_wrapper = create_dataloaders(config)
    
    # Load trained model
    checkpoint = torch.load(args.model_path, map_location=config.device)
    model = EnhancedMAGNET(config)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(config.device)
    
    print(f"Loaded model from {args.model_path}")
    
    # Evaluate model
    metrics = evaluate_model(model, test_loader, config)
    
    # Print and save metrics
    print("\nEvaluation results:")
    print(f"Byte accuracy: {metrics['byte_accuracy']:.4f}")
    print(f"Boundary precision: {metrics['boundary_precision']:.4f}")
    print(f"Boundary recall: {metrics['boundary_recall']:.4f}")
    print(f"Boundary F1: {metrics['boundary_f1']:.4f}")
    print(f"Loss: {metrics['loss']:.4f}")
    
    # Save metrics to file
    with open(os.path.join(args.output_dir, "evaluation_results.txt"), "w") as f:
        for key, value in metrics.items():
            f.write(f"{key}: {value:.6f}\n")
    
    print(f"Results saved to {os.path.join(args.output_dir, 'evaluation_results.txt')}")

if __name__ == "__main__":
    main()