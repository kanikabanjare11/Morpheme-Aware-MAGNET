"""Enhanced MAGNET model for Indian languages with morphological priors."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.hourglass_transformer import HourglassTransformer
from tools.morfessor_util import detect_language

class EnhancedMAGNET(nn.Module):
    """
    Enhanced MAGNET model with morpheme-aware compression and boundary supervision.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Hourglass Transformer backbone
        self.transformer = HourglassTransformer(config)
        
    def forward(self, batch, temperature=1.0):
        """
        Forward pass through the MAGNET model.
        
        Args:
            batch: Dictionary containing:
                input_bytes: Byte sequence [batch_size, seq_len]
                attention_mask: Mask for valid positions
                lang: List of language codes
            temperature: Temperature for Gumbel-Sigmoid
            
        Returns:
            Dictionary containing model outputs and losses
        """
        input_bytes = batch['input_bytes']
        attention_mask = batch['attention_mask']
        langs = batch['lang']
        
        # Forward pass through transformer
        outputs = self.transformer(
            input_bytes=input_bytes,
            attention_mask=attention_mask,
            langs=langs,
            temperature=temperature
        )
        
        # Extract outputs
        logits = outputs['logits']
        boundary_probs = outputs['boundary_probs']
        loss_mask = outputs['loss_mask']
        
        return {
            'logits': logits,                   # Next-byte prediction
            'boundary_probs': boundary_probs,   # Boundary probabilities
            'loss_mask': loss_mask              # Mask for loss calculation
        }

    def compute_language_modeling_loss(self, logits, target_bytes, loss_mask):
        """
        Compute language modeling loss (next-byte prediction).
        
        Args:
            logits: Model logits [batch_size, seq_len, 256]
            target_bytes: Target byte values [batch_size, seq_len]
            loss_mask: Mask for valid positions [batch_size, seq_len]
            
        Returns:
            Language modeling loss
        """
        # Shift target_bytes to get next-byte prediction targets
        # Input: [b, c, d, ...], Target: [c, d, ..., EOS]
        shifted_target = torch.zeros_like(target_bytes)
        shifted_target[:, :-1] = target_bytes[:, 1:]
        
        # Reshape for cross-entropy
        batch_size, seq_len = target_bytes.shape
        logits_flat = logits.view(-1, 256)
        targets_flat = shifted_target.view(-1)
        mask_flat = loss_mask.view(-1)
        
        # Compute loss only on valid positions
        loss = F.cross_entropy(
            logits_flat, 
            targets_flat, 
            reduction='none'
        )
        
        # Apply mask and compute mean
        loss = (loss * mask_flat).sum() / mask_flat.sum().clamp(min=1)
        
        return loss
    
    def compute_boundary_compression_loss(self, boundary_probs, compression_ratios, loss_mask):
        """
        Compute boundary compression loss to match morpheme-aware compression ratio.
        
        Args:
            boundary_probs: Predicted boundary probabilities [batch_size, seq_len, 1]
            compression_ratios: Target compression ratios [batch_size]
            loss_mask: Mask for valid positions [batch_size, seq_len]
            
        Returns:
            Boundary compression loss
        """
        batch_size, seq_len, _ = boundary_probs.shape
        
        # Count predicted boundaries
        # Sum boundary probabilities for valid positions
        valid_counts = loss_mask.sum(dim=1)
        boundary_sums = (boundary_probs.squeeze(-1) * loss_mask).sum(dim=1)
        
        # Compute actual compression ratios (valid_chars / num_boundaries)
        actual_ratios = valid_counts / boundary_sums.clamp(min=1)
        
        # Compute MSE between actual and target ratios
        loss = F.mse_loss(actual_ratios, compression_ratios)
        
        return loss
    
    def boundary_entropy_loss(self, boundary_probs, loss_mask):
        eps = 1e-8
        entropy = -(
            boundary_probs*(boundary_probs + eps).log() +(1 - boundary_probs) * (1 - boundary_probs + eps).log()
        )
        entropy = (entropy.squeeze(-1) * loss_mask).sum() / loss_mask.sum().clamp(min=1)
        return entropy

    def fuzzy_boundary_loss(self, boundary_probs, morfessor_boundaries, loss_mask, window=2):
        # morfessor_boundaries: binary tensor [batch, seq_len], 1 at morpheme boundary
        proto_targets = torch.zeros_like(boundary_probs.squeeze(-1))
        for w in range(-window, window + 1):
            shifted = torch.roll(morfessor_boundaries, shifts=w, dims=1)
            proto_targets = torch.max(proto_targets, shifted)
        loss = F.binary_cross_entropy(boundary_probs.squeeze(-1), proto_targets, reduction='none')
        loss = (loss * loss_mask).sum() / loss_mask.sum().clamp(min=1)
        return loss

    def compute_binomial_boundary_loss(self, boundary_probs, prior, loss_mask):
        """
        Compute binomial loss for boundary prediction regularization.
    
    Args:
        boundary_probs: Predicted boundary probabilities [batch_size, seq_len, 1]
        prior: Target probability of segment boundary
        loss_mask: Mask for valid positions [batch_size, seq_len]
        
    Returns:
        Binomial log probability loss
    """
        # Count non-padding positions
        total_count = loss_mask.sum(dim=1, keepdim=True).float()
    
        # Count predicted boundaries in non-padding positions
        sum_preds = (boundary_probs.squeeze(-1) * loss_mask).sum(dim=1, keepdim=True)
    
        # Create binomial distribution with target prior
        binomial = torch.distributions.binomial.Binomial(
            total_count,
            probs=torch.tensor([prior], device=boundary_probs.device)
        )
    
        # Compute negative log probability (loss)
        loss = -binomial.log_prob(sum_preds).mean()
    
        return loss

    def compute_compression_stats(self, hard_boundaries, attention_mask):
        """
    Compute compression statistics for logging and monitoring.
    
    Args:
        hard_boundaries: Binary boundary predictions [batch_size, seq_len, 1]
        attention_mask: Mask for valid positions [batch_size, seq_len]
        
    Returns:
        Dictionary with compression statistics
        """
        # Mask out padding
        mask = attention_mask.eq(1)
        masked_hard_boundaries = hard_boundaries.squeeze(-1) * mask.float()
    
        # Count non-padded positions
        num_positions = mask.sum(dim=1).float()
    
        # Count boundaries
        sum_boundaries = masked_hard_boundaries.sum(dim=1)
    
        # Compute compression rate (non-padded / boundaries)
        compression_rate = (num_positions / sum_boundaries.clamp(min=1)).mean().item()
    
        # Compute probability of being a boundary
        p_ones = (sum_boundaries / num_positions.clamp(min=1)).mean().item()
    
        # Average segment length
        avg_segment_len = 1 / p_ones if p_ones > 0 else float('inf')
    
        return {
            'compression_rate': compression_rate,
            'p_ones': p_ones,
            'avg_segment_len': avg_segment_len
        }
    
    # def compute_morphological_boundary_loss(self, boundary_probs, gold_boundaries, loss_mask):
    #     """
    #     Compute morphological boundary supervision loss.
        
    #     Args:
    #         boundary_probs: Predicted boundary probabilities [batch_size, seq_len, 1]
    #         gold_boundaries: Gold boundary labels [batch_size, seq_len]
    #         loss_mask: Mask for valid positions [batch_size, seq_len]
            
    #     Returns:
    #         Morphological boundary loss
    #     """

    #     # Compute BCE loss between predicted and gold boundaries
    #     boundary_loss = F.binary_cross_entropy_with_logits(
    #         boundary_probs.squeeze(-1),
    #         gold_boundaries,
    #         reduction='none'
    #     )
        
    #     # Apply mask and compute mean
    #     boundary_loss = (boundary_loss * loss_mask).sum() / loss_mask.sum().clamp(min=1)
        
    #     return boundary_loss