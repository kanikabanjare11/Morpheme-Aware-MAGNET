"""Basic building blocks for the Hourglass Transformer."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class PositionalEncoding(nn.Module):
    """Positional encoding for the transformer architecture."""
    
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        
        # Create positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        
        self.register_buffer('pe', pe)
        
    def forward(self, x):
        """
        Add positional encoding to input tensor.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            
        Returns:
            Output with positional encoding added
        """
        x = x + self.pe[:, :x.size(1), :]
        return self.dropout(x)

class TransformerEncoderLayer(nn.Module):
    """Custom Transformer encoder layer with pre-normalization."""
    
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout, batch_first= True)
        
        # Feed-forward network
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Dropout
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, src, src_mask=None, src_key_padding_mask=None):
        """
        Forward pass through encoder layer.
        
        Args:
            src: Input tensor [seq_len, batch_size, d_model] or [batch_size, seq_len, d_model]
            src_mask: Mask for self-attention
            src_key_padding_mask: Mask for padding
            
        Returns:
            Output tensor same shape as src
        """
        # Handle different input formats (batch first or not)
        batch_first = src.dim() == 3 and src.size(0) > src.size(1)
        if batch_first:
            # Convert to [seq_len, batch_size, d_model]
            src = src.transpose(0, 1)
        
        # Pre-normalization
        src2 = self.norm1(src)
        
        # Self-attention
        src2, _ = self.self_attn(
            src2, src2, src2, 
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask
        )
        src = src + self.dropout1(src2)
        
        # Feed-forward network with residual connection
        src2 = self.norm2(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src2))))
        src = src + self.dropout2(src2)
        
        # Convert back to batch-first if needed
        if batch_first:
            src = src.transpose(0, 1)
            
        return src