"""Utility functions for the Enhanced MAGNET model."""
import torch
import numpy as np
import random
import os
from typing import Dict, List, Tuple

def set_seed(seed):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def byte_accuracy(predictions, targets, mask=None):
    """
    Calculate accuracy of byte predictions.
    
    Args:
        predictions: Predicted logits of shape [batch_size, seq_len, 256]
        targets: Target bytes of shape [batch_size, seq_len]
        mask: Optional mask of shape [batch_size, seq_len]
        
    Returns:
        Accuracy (0.0-1.0)
    """
    # Get predicted byte indices
    pred_indices = torch.argmax(predictions, dim=-1)
    
    # Shift targets to match next-byte prediction
    shifted_targets = torch.zeros_like(targets)
    shifted_targets[:, :-1] = targets[:, 1:]
    
    # Calculate correct predictions
    correct = (pred_indices == shifted_targets)
    
    # Apply mask if provided
    if mask is not None:
        correct = correct * mask
        return correct.sum().float() / mask.sum().clamp(min=1)
    
    return correct.float().mean()

def boundary_f1_score(pred_boundaries, gold_boundaries, mask=None):
    """
    Calculate F1 score for boundary prediction.
    
    Args:
        pred_boundaries: Predicted boundary probabilities [batch_size, seq_len, 1]
        gold_boundaries: Gold boundary labels [batch_size, seq_len]
        mask: Optional mask for valid positions [batch_size, seq_len]
        
    Returns:
        Dictionary containing precision, recall, and F1
    """
    # Convert probabilities to binary predictions (threshold = 0.5)
    pred_boundaries = (pred_boundaries.squeeze(-1) > 0.5).float()
    
    # Apply mask if provided
    if mask is not None:
        pred_boundaries = pred_boundaries * mask
        gold_boundaries = gold_boundaries * mask
    
    # Calculate true positives, false positives, false negatives
    true_positives = (pred_boundaries * gold_boundaries).sum()
    false_positives = (pred_boundaries * (1 - gold_boundaries)).sum()
    false_negatives = ((1 - pred_boundaries) * gold_boundaries).sum()
    
    # Calculate precision, recall, F1
    precision = true_positives / (true_positives + false_positives).clamp(min=1e-8)
    recall = true_positives / (true_positives + false_negatives).clamp(min=1e-8)
    f1 = 2 * precision * recall / (precision + recall).clamp(min=1e-8)
    
    return {
        'precision': precision.item(),
        'recall': recall.item(),
        'f1': f1.item()
    }