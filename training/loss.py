"""Loss functions for the Enhanced MAGNET model (IMPROVED)."""
import torch
import torch.nn.functional as F


def compute_total_loss(model_outputs, batch, config):
    """
    Compute the total loss from model outputs, MAGNET-style.
    Returns a dict of all relevant loss components.
    
    Args:
        model_outputs: Dict containing:
            - 'logits': Language model token prediction logits [B, L, V]
            - 'boundary_logits': Raw boundary prediction logits [B, L, 1] or [B, L]
            - 'boundary_probs': Sigmoid-activated boundary probabilities [B, L, 1] or [B, L]
            - 'loss_mask': Mask for valid token positions [B, L]
        batch: Dict containing:
            - 'input_bytes': Input token ids [B, L]
            - 'gold_boundaries': Optional gold boundary labels [B, L]
            - 'compression_ratio': Optional target compression ratios [B]
            - 'lang': Optional language identifiers for each example
        config: Model configuration with loss weights and settings
    
    Returns:
        Dict of loss components and the total weighted loss
    """
    # Model outputs
    logits = model_outputs['logits']  # [B, L, V]
    
    # Get boundary logits and probs (ensuring consistent shapes)
    boundary_logits = model_outputs.get('boundary_logits', None)
    boundary_probs = model_outputs['boundary_probs']  # [B, L, 1] or [B, L]
    
    # Make sure boundary_probs is [B, L] for consistent handling
    if boundary_probs.dim() > 2:
        boundary_probs = boundary_probs.squeeze(-1)  # [B, L]
    
    loss_mask = model_outputs['loss_mask']  # [B, L]
    
    # Batch data
    input_bytes = batch['input_bytes']  # [B, L]
    gold_boundaries = batch.get('gold_boundaries', None)
    compression_ratios = batch.get('compression_ratio', None)
    
    # 1. Language Modeling Loss
    lm_loss = compute_language_modeling_loss(logits, input_bytes, loss_mask)
    
    # 2. Get language-specific priors
    langs = batch.get('lang', [None] * input_bytes.size(0))
    language_priors = getattr(config, 'language_priors', {})
    
    if compression_ratios is not None:
        # If user-supplied, convert avg segment length to boundary probability
        priors = 1.0 / (compression_ratios.clamp(min=1e-8))
    else:
        priors = torch.tensor([
            language_priors.get(lang, 0.1) for lang in langs
        ], device=boundary_probs.device)
    
    # Ensure priors are in valid probability range
    priors = torch.clamp(priors, 0.0, 1.0)
    
    # 3. Boundary Count Loss (soft/MSE)
    boundary_loss = compute_boundary_count_loss(boundary_probs, priors, loss_mask)
    
    # 4. Entropy Loss (promoting more decisive boundary/non-boundary decisions)
    entropy_loss = compute_entropy_loss(boundary_probs, loss_mask)
    
    # 5. Morphological Loss (if gold boundaries available)
    morph_loss = torch.tensor(0.0, device=logits.device)
    if gold_boundaries is not None:
        if hasattr(config, 'use_fuzzy_morphology') and config.use_fuzzy_morphology:
            window = getattr(config, 'morph_window', 2)
            morph_loss = compute_fuzzy_boundary_loss(
                boundary_probs, gold_boundaries, loss_mask, window
            )
        else:
            # Use logits if available for BCE with Logits Loss, otherwise use probs
            if boundary_logits is not None:
                # Ensure consistent shape
                if boundary_logits.dim() > 2:
                    boundary_logits = boundary_logits.squeeze(-1)  # [B, L]
                morph_loss = compute_supervised_boundary_loss_with_logits(
                    boundary_logits, gold_boundaries, loss_mask
                )
            else:
                morph_loss = compute_supervised_boundary_loss(
                    boundary_probs, gold_boundaries, loss_mask
                )
    
    # 6. Loss weighting (these can be annealed in main loop/trainer)
    lambda_lm = getattr(config, 'lambda_lm', 1.0)
    lambda_boundary = getattr(config, 'lambda_boundary', 0.1)
    lambda_morph = getattr(config, 'lambda_morph', 0.05)
    lambda_entropy = getattr(config, 'lambda_entropy', 0.01)

    # Print weights for debugging
    if hasattr(config, 'verbose') and config.verbose and \
       hasattr(config, 'global_step') and config.global_step % 100 == 0:
        print(f"Loss weights: lm={lambda_lm}, boundary={lambda_boundary}, "
              f"morph={lambda_morph}, entropy={lambda_entropy}")
              
    # Normalize boundary loss to prevent extreme values
    scaled_boundary_loss = boundary_loss
    boundary_scale_factor = getattr(config, 'boundary_scale_factor', None)
    if boundary_scale_factor is not None:
        scaled_boundary_loss = boundary_loss / boundary_scale_factor
    
    total_loss = (
        lambda_lm * lm_loss +
        lambda_boundary * scaled_boundary_loss +
        lambda_morph * morph_loss -
        lambda_entropy * entropy_loss
    )
    
    # Print loss components' scale for debugging
    if hasattr(config, 'verbose') and config.verbose and \
       hasattr(config, 'global_step') and config.global_step % 100 == 0:
        print(f"Loss components: lm={lm_loss.item():.4f}, "
              f"boundary={boundary_loss.item():.4f}, "
              f"scaled boundary={scaled_boundary_loss.item():.4f}, "
              f"morph={morph_loss.item():.4f}, "
              f"entropy={entropy_loss.item():.4f}")
    
    return {
        'total_loss': total_loss,
        'lm_loss': lm_loss,
        'boundary_loss': boundary_loss,
        'morph_loss': morph_loss,
        'entropy_loss': entropy_loss
    }


def compute_language_modeling_loss(logits, input_bytes, loss_mask):
    """
    Compute masked language modeling loss.
    
    Args:
        logits: [B, L, V] prediction logits
        input_bytes: [B, L] input token ids
        loss_mask: [B, L] mask for valid positions
        
    Returns:
        Average token-level cross entropy loss
    """
    # Shift targets for next-token prediction
    shifted_target = torch.zeros_like(input_bytes)
    shifted_target[:, :-1] = input_bytes[:, 1:]
    
    # Flatten tensors for loss calculation
    logits_flat = logits.reshape(-1, logits.size(-1))  # [B*L, V]
    targets_flat = shifted_target.reshape(-1)  # [B*L]
    mask_flat = loss_mask.reshape(-1)  # [B*L]
    
    # Compute token-level loss
    loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')
    
    # Apply mask and average
    loss = (loss * mask_flat).sum() / mask_flat.sum().clamp(min=1)
    return loss


def compute_boundary_count_loss(boundary_probs, priors, loss_mask):
    """
    Improved (soft) boundary loss: MSE between sum of predicted boundaries and expected number.
    This is easier to learn than a hard binomial NLL.
    
    Args:
        boundary_probs: [B, L] boundary probabilities (sigmoid output)
        priors: [B] expected ratio of boundary tokens
        loss_mask: [B, L] mask for valid positions
        
    Returns:
        MSE loss between predicted and expected boundary counts
    """
    # Ensure consistent shapes
    assert boundary_probs.dim() == 2, f"Expected 2D tensor, got {boundary_probs.dim()}D"
    assert loss_mask.dim() == 2, f"Expected 2D tensor for mask, got {loss_mask.dim()}D"
    
    # Get token counts and predicted boundary sums per batch
    total_count = loss_mask.sum(dim=1).float()  # [B]
    sum_preds = (boundary_probs * loss_mask).sum(dim=1)  # [B]
    
    # Compute expected boundary count based on prior probabilities
    expected = total_count * priors.to(boundary_probs.device)  # [B]
    
    # Compute MSE between predicted and expected counts
    loss = ((sum_preds - expected) ** 2).mean()
    
    return loss


def compute_supervised_boundary_loss_with_logits(boundary_logits, gold_boundaries, loss_mask):
    """
    Compute token-level boundary supervision loss using raw logits.
    
    Args:
        boundary_logits: [B, L] raw boundary logits
        gold_boundaries: [B, L] gold boundary labels (0/1)
        loss_mask: [B, L] mask for valid positions
        
    Returns:
        Average BCE loss for boundary prediction
    """
    # Use BCEWithLogitsLoss with raw logits
    loss = F.binary_cross_entropy_with_logits(
        boundary_logits, 
        gold_boundaries.float(), 
        reduction='none'
    )
    
    # Apply mask and average
    loss = (loss * loss_mask).sum() / loss_mask.sum().clamp(min=1)
    return loss


def compute_supervised_boundary_loss(boundary_probs, gold_boundaries, loss_mask):
    """
    Compute token-level boundary supervision loss using probabilities.
    
    Args:
        boundary_probs: [B, L] boundary probabilities (sigmoid output)
        gold_boundaries: [B, L] gold boundary labels (0/1)
        loss_mask: [B, L] mask for valid positions
        
    Returns:
        Average BCE loss for boundary prediction
    """
    # Use BCE with pre-computed probabilities
    loss = F.binary_cross_entropy(
        boundary_probs, 
        gold_boundaries.float(), 
        reduction='none'
    )
    
    # Apply mask and average
    loss = (loss * loss_mask).sum() / loss_mask.sum().clamp(min=1)
    return loss


def compute_fuzzy_boundary_loss(boundary_probs, gold_boundaries, loss_mask, window=2):
    """
    Compute fuzzy boundary loss that allows matches within a window.
    
    Args:
        boundary_probs: [B, L] boundary probabilities (sigmoid output)
        gold_boundaries: [B, L] gold boundary labels (0/1)
        loss_mask: [B, L] mask for valid positions
        window: Size of the window around gold boundaries to consider as valid
        
    Returns:
        Average BCE loss with fuzzy boundary matching
    """
    # Create fuzzy targets by considering window around gold boundaries
    proto_targets = torch.zeros_like(boundary_probs)
    for w in range(-window, window + 1):
        shifted = torch.roll(gold_boundaries, shifts=w, dims=1)
        proto_targets = torch.max(proto_targets, shifted)
    
    # Compute BCE loss with fuzzy targets
    loss = F.binary_cross_entropy(
        boundary_probs, 
        proto_targets.float(), 
        reduction='none'
    )
    
    # Apply mask and average
    loss = (loss * loss_mask).sum() / loss_mask.sum().clamp(min=1)
    return loss


def compute_entropy_loss(boundary_probs, loss_mask):
    """
    Compute entropy loss to encourage more decisive boundary predictions.
    
    Args:
        boundary_probs: [B, L] boundary probabilities (sigmoid output)
        loss_mask: [B, L] mask for valid positions
        
    Returns:
        Average entropy of boundary predictions
    """
    # Numerical stability epsilon
    eps = 1e-8
    
    # Compute binary entropy
    entropy = -(
        boundary_probs * (boundary_probs + eps).log() +
        (1 - boundary_probs) * (1 - boundary_probs + eps).log()
    )
    
    # Apply mask and average
    entropy = (entropy * loss_mask).sum() / loss_mask.sum().clamp(min=1)
    return entropy


def compute_compression_stats(boundary_probs, loss_mask):
    """
    Compute compression-related statistics from boundary predictions.
    
    Args:
        boundary_probs: [B, L] boundary probabilities (sigmoid output)
        loss_mask: [B, L] mask for valid positions
        
    Returns:
        Dict of compression statistics
    """
    # Binarize probabilities
    hard_boundaries = (boundary_probs > 0.5).float()
    
    # Apply mask
    masked_boundaries = hard_boundaries * loss_mask
    
    # Count valid positions and boundaries
    num_positions = loss_mask.sum(dim=1).float()  # [B]
    sum_boundaries = masked_boundaries.sum(dim=1)  # [B]
    
    # Identify valid samples (those with at least one boundary)
    valid_samples = (sum_boundaries > 0).float()
    
    # Compute compression rate for valid samples
    if valid_samples.sum() > 0:
        valid_positions = (num_positions * valid_samples).sum()
        valid_boundaries = (sum_boundaries * valid_samples).sum()
        compression_rate = (valid_positions / valid_boundaries).item()
    else:
        compression_rate = float('inf')
    
    # Compute overall boundary probability
    total_positions = num_positions.sum().item()
    if total_positions > 0:
        p_ones = sum_boundaries.sum().item() / total_positions
    else:
        p_ones = 0.0
    
    # Compute average segment length
    avg_segment_len = 1 / p_ones if p_ones > 0 else float('inf')
    
    return {
        'compression_rate': compression_rate,
        'p_ones': p_ones,
        'avg_segment_len': avg_segment_len
    }