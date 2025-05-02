"""Hourglass Transformer architecture for efficient long-sequence modeling."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from model.layers import TransformerEncoderLayer, PositionalEncoding
from model.boundary_predictor import BoundaryPredictor

class HourglassTransformer(nn.Module):
    """
    Hourglass Transformer with downsampling and upsampling structure.
    """
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Embedding layer for byte inputs (0-255)
        self.embedding = nn.Embedding(256, config.d_model)
        self.positional_encoding = PositionalEncoding(config.d_model, config.dropout, config.max_seq_length)
        
        # Pre-bottleneck encoder layers
        self.pre_encoder = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.nhead,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout
            ) for _ in range(config.pre_encoder_layers)
        ])
        
        self.boundary_predictors = nn.ModuleDict({
            lang: BoundaryPredictor(
                d_model=config.d_model,
                d_inner=config.dim_feedforward,
                activation_function=getattr(config, 'activation_function', 'gelu'),
                temp=getattr(config, 'initial_temperature', 1.0),
                prior=getattr(config, 'language_priors', {}).get(lang, 0.1),
                bp_type=getattr(config, 'bp_type', 'gumbel')
            )
            for lang in config.supported_languages
        })
        
        # Middle (bottleneck) encoder layers
        self.middle_encoder = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.nhead,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout
            ) for _ in range(config.middle_encoder_layers)
        ])
        
        # Post-bottleneck encoder layers (after upsampling)
        self.post_encoder = nn.ModuleList([
            TransformerEncoderLayer(
                d_model=config.d_model,
                nhead=config.nhead,
                dim_feedforward=config.dim_feedforward,
                dropout=config.dropout
            ) for _ in range(config.post_encoder_layers)
        ])
        
        # Upsampling layer to expand bottlenecked sequence
        self.upsample = nn.Linear(config.d_model, config.d_model * config.pooling_factor)
        
        # Output prediction head (for next byte prediction)
        self.output_head = nn.Linear(config.d_model, 256)  # 256 = byte vocabulary size
        
        # Language embedding for script-aware processing
        self.lang_embeddings = nn.Embedding(len(config.supported_languages), config.d_model)
        
    def gumbel_sigmoid(self, logits, temperature=1.0, hard=False):
        """
        Apply the Gumbel-Sigmoid for differentiable binary sampling.
        
        Args:
            logits: Raw logits
            temperature: Temperature parameter controlling sample discreteness
            hard: Whether to use straight-through estimator for hard samples
            
        Returns:
            Differentiable binary samples
        """
        probs = torch.sigmoid(logits)
        if hard:
            # For inference or when we need discrete samples
            bernoulli = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(
                temperature=temperature,
                probs=probs,
            )
            soft_boundaries = bernoulli.rsample()
        
            # Straight-through estimator for discrete boundaries
            hard_boundaries = (soft_boundaries > 0.5).float()
            boundaries = hard_boundaries - soft_boundaries.detach() + soft_boundaries
            return boundaries
        else:
            # For training with soft samples
            bernoulli = torch.distributions.relaxed_bernoulli.RelaxedBernoulli(
                temperature=temperature, 
                probs=probs
            )
            return bernoulli.rsample()
    
    def pooling_with_boundaries(self, x, boundaries, attention_mask, pooling_factor):
        """
        Pool sequence based on predicted boundaries.
        
        Args:
            x: Hidden states [batch_size, seq_len, d_model]
            boundaries: Boundary probabilities [batch_size, seq_len, 1]
            attention_mask: Mask for valid positions [batch_size, seq_len]
            pooling_factor: Desired compression factor
            
        Returns:
            Pooled sequence and attention mask
        """
        batch_size, seq_len, d_model = x.shape
        device = x.device
        
        # For adaptive boundary-based pooling
        boundaries = boundaries.squeeze(-1)  # [batch_size, seq_len]
        
        # Create pooled representations
        pooled_outputs = []
        pooled_masks = []
        
        for i in range(batch_size):
            # Get valid sequence (non-padding)
            valid_len = attention_mask[i].sum().int().item()
            if valid_len == 0:
                # Handle empty sequence
                empty_pooled = torch.zeros(seq_len // pooling_factor, d_model, device=device)
                empty_mask = torch.zeros(seq_len // pooling_factor, device=device)
                pooled_outputs.append(empty_pooled)
                pooled_masks.append(empty_mask)
                continue
                
            # Extract valid sequence and boundaries
            seq = x[i, :valid_len]
            seq_boundaries = boundaries[i, :valid_len]
            
            # Ensure we have enough boundaries for pooling
            # Target: At least valid_len / pooling_factor boundaries
            target_boundaries = valid_len // pooling_factor
            
            # Get boundary indices (where probability > 0.5)
            boundary_indices = (seq_boundaries > 0.5).nonzero(as_tuple=True)[0]
            
            # If we don't have enough boundaries, add more based on scores
            if len(boundary_indices) < target_boundaries:
                # Get indices of highest boundary probabilities that aren't already boundaries
                non_boundary_scores = seq_boundaries.clone()
                non_boundary_scores[boundary_indices] = 0.0
                _, additional_indices = torch.topk(
                    non_boundary_scores, 
                    k=min(target_boundaries - len(boundary_indices), valid_len - len(boundary_indices))
                )
                boundary_indices = torch.cat([boundary_indices, additional_indices])
                boundary_indices = torch.sort(boundary_indices)[0]
            
            # If we have too many boundaries, keep the ones with highest probability
            elif len(boundary_indices) > target_boundaries:
                boundary_scores = seq_boundaries[boundary_indices]
                _, top_indices = torch.topk(boundary_scores, k=target_boundaries)
                boundary_indices = boundary_indices[top_indices]
                boundary_indices = torch.sort(boundary_indices)[0]
            
            # Add beginning and end for complete segmentation
            if len(boundary_indices) == 0 or boundary_indices[0] != 0:
                boundary_indices = torch.cat([torch.tensor([0], device=device), boundary_indices])
            if boundary_indices[-1] != valid_len - 1:
                boundary_indices = torch.cat([boundary_indices, torch.tensor([valid_len - 1], device=device)])
            
            # Segment and pool
            segments = []
            for j in range(len(boundary_indices) - 1):
                start, end = boundary_indices[j], boundary_indices[j+1]
                if end > start:  # Ensure non-empty segment
                    segment = seq[start:end+1]
                    # Mean pooling for segment
                    segments.append(segment.mean(dim=0))
            
            # Create pooled sequence (pad if needed)
            pooled_len = len(segments)
            pooled_seq = torch.stack(segments) if segments else torch.zeros(0, d_model, device=device)
            
            # Create new attention mask for pooled sequence
            pooled_attention = torch.ones(pooled_len, device=device)
            
            # Pad if needed
            target_pooled_len = seq_len // pooling_factor
            if pooled_len < target_pooled_len:
                padding = torch.zeros(target_pooled_len - pooled_len, d_model, device=device)
                pooled_seq = torch.cat([pooled_seq, padding], dim=0)
                padding_mask = torch.zeros(target_pooled_len - pooled_len, device=device)
                pooled_attention = torch.cat([pooled_attention, padding_mask], dim=0)
            elif pooled_len > target_pooled_len:
                # Truncate if too long (shouldn't happen often)
                pooled_seq = pooled_seq[:target_pooled_len]
                pooled_attention = pooled_attention[:target_pooled_len]
            
            pooled_outputs.append(pooled_seq)
            pooled_masks.append(pooled_attention)
        
        # Stack batch outputs
        pooled_hidden = torch.stack(pooled_outputs)
        pooled_mask = torch.stack(pooled_masks)
        
        return pooled_hidden, pooled_mask

    def forward(self, input_bytes, attention_mask, langs, temperature=1.0):
        """
        Forward pass through Hourglass Transformer.
        
        Args:
            input_bytes: Byte sequence [batch_size, seq_len]
            attention_mask: Mask for valid positions [batch_size, seq_len]
            langs: List of language codes for each sample
            temperature: Temperature for Gumbel-Sigmoid
            
        Returns:
            Dictionary containing:
                logits: Next-byte prediction logits
                boundary_probs: Predicted boundary probabilities
                loss_mask: Mask for valid positions in loss calculation
        """
        batch_size, seq_len = input_bytes.shape
        device = input_bytes.device
        
        # Embedding layer
        x = self.embedding(input_bytes)  # [batch_size, seq_len, d_model]
        x = self.positional_encoding(x)
        
        # Create extended attention mask for transformer
        # (1 = valid position, 0 = masked position)
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0
        
        # Add language embeddings
        lang_indices = torch.tensor(
            [self.config.supported_languages.index(lang) for lang in langs],
            device=device
        )
        lang_emb = self.lang_embeddings(lang_indices).unsqueeze(1)  # [batch_size, 1, d_model]
        x = x + lang_emb
        
        # Pre-bottleneck encoding
        for layer in self.pre_encoder:
            x = layer(x, src_key_padding_mask=(attention_mask == 0))
        
        # Predict boundaries
        boundary_logits = torch.zeros(batch_size, seq_len, 1, device=device)
        for i, lang in enumerate(langs):
            if lang in self.boundary_predictors:
                boundary_output = self.boundary_predictors[lang](x[i].unsqueeze(0))
                if isinstance(boundary_output, tuple):
                    boundary_logits[i] = boundary_output[0]
                else:
                    boundary_logits[i] = boundary_output
                # soft_boundaries = out[0]
                # boundary_logits[i] = soft_boundaries.squeeze(0)
        
        # Apply Gumbel-Sigmoid to get differentiable binary boundaries
        boundary_probs = self.gumbel_sigmoid(boundary_logits, temperature=temperature)
        
        # Pool sequences based on boundaries to create bottleneck
        pooled_x, pooled_mask = self.pooling_with_boundaries(
            x, boundary_probs, attention_mask, self.config.pooling_factor
        )
        
        # Create extended attention mask for pooled sequence
        pooled_attention_mask = pooled_mask.unsqueeze(1).unsqueeze(2)
        pooled_attention_mask = (1.0 - pooled_attention_mask) * -10000.0
        
        # Middle (bottleneck) encoding
        for layer in self.middle_encoder:
            pooled_x = layer(pooled_x, src_key_padding_mask=(pooled_mask == 0))

        # Upsample back to original sequence length
        batch_size, pooled_len, d_model = pooled_x.shape
        upsampled_x = self.upsample(pooled_x).view(
            batch_size, pooled_len, self.config.pooling_factor, d_model
        )
        # Reshape to [batch_size, seq_len, d_model] and handle any dimension mismatch
        upsampled_x = upsampled_x.reshape(batch_size, pooled_len * self.config.pooling_factor, d_model)
        upsampled_x = upsampled_x[:, :seq_len]
        # If upsampled sequence is shorter than original, pad it
        if upsampled_x.size(1) < seq_len:
            padding = torch.zeros(
                batch_size, seq_len - upsampled_x.size(1), d_model, 
                device=upsampled_x.device
            )
            upsampled_x = torch.cat([upsampled_x, padding], dim=1)
        
        # Post-bottleneck encoding
        for layer in self.post_encoder:
            upsampled_x = layer(upsampled_x, src_key_padding_mask=(attention_mask == 0))
        
        # Output layer for next-byte prediction
        logits = self.output_head(upsampled_x)
        
        return {
            'logits': logits,  # [batch_size, seq_len, 256]
            'boundary_probs': boundary_probs,  # [batch_size, seq_len, 1]
            'loss_mask': attention_mask  # [batch_size, seq_len]
        }