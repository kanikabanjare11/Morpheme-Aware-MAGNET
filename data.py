"""Data loading and preprocessing for Indic language text."""
import os
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tools.morfessor_util import MorfessorWrapper, detect_language

class IndicTextDataset(Dataset):
    """Dataset for Indic language text with morphological features."""
    
    def __init__(self, file_paths, morfessor_wrapper, script_ranges, max_length=1024):
        """
        Args:
            file_paths: Dictionary mapping language codes to file paths
            morfessor_wrapper: MorfessorWrapper instance for morphological analysis
            script_ranges: Dictionary of Unicode ranges for language detection
            max_length: Maximum sequence length in bytes
        """
        self.file_paths = file_paths
        self.morfessor_wrapper = morfessor_wrapper
        self.script_ranges = script_ranges
        self.max_length = max_length
        
        # Load all data
        self.samples = []
        for lang, path in file_paths.items():
            if not os.path.exists(path):
                print(f"Warning: File not found - {path}")
                continue
                
            with open(path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:  # Skip empty lines
                        self.samples.append((line, lang))
        
        print(f"Loaded {len(self.samples)} samples from {len(file_paths)} languages")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        text, lang = self.samples[idx]
        
        # Convert text to byte sequence
        bytes_seq = np.array(list(text.encode('utf-8')), dtype=np.int64)
        
        # Truncate if too long
        if len(bytes_seq) > self.max_length:
            bytes_seq = bytes_seq[:self.max_length]
        
        # Get morpheme boundaries from Morfessor
        morphemes, boundary_labels = self.morfessor_wrapper.get_morphemes_and_boundaries(text, lang)
        boundary_labels = np.array(boundary_labels, dtype=np.float32)
        
        # Truncate boundary labels if needed
        if len(boundary_labels) > self.max_length:
            boundary_labels = boundary_labels[:self.max_length]
        
        # Pad inputs to max_length
        attention_mask = np.zeros(self.max_length, dtype=np.float32)
        attention_mask[:len(bytes_seq)] = 1.0
        
        padded_bytes = np.zeros(self.max_length, dtype=np.int64)
        padded_bytes[:len(bytes_seq)] = bytes_seq
        
        padded_boundaries = np.zeros(self.max_length, dtype=np.float32)
        padded_boundaries[:len(boundary_labels)] = boundary_labels
        
        # Get compression ratio (bytes / morphemes)
        compression_ratio = len(bytes_seq) / max(1, len(morphemes))
        
        return {
            'input_bytes': torch.tensor(padded_bytes),
            'attention_mask': torch.tensor(attention_mask),
            'gold_boundaries': torch.tensor(padded_boundaries),
            'compression_ratio': torch.tensor(compression_ratio, dtype=torch.float32),
            'lang': lang,
            'seq_len': len(bytes_seq)
        }

def create_dataloaders(config):
    """Create data loaders for train, validation, and test sets."""
    # Initialize Morfessor wrapper
    morfessor_wrapper = MorfessorWrapper(config.morfessor_models)
    
    # File paths for each language and split
    train_files = {lang: os.path.join(config.data_path, f"{lang}_train.txt") 
                  for lang in config.supported_languages}
    
    val_files = {lang: os.path.join(config.data_path, f"{lang}_val.txt") 
                for lang in config.supported_languages}
    
    test_files = {lang: os.path.join(config.data_path, f"{lang}_test.txt") 
                 for lang in config.supported_languages}
    
    # Create datasets
    train_dataset = IndicTextDataset(
        train_files, morfessor_wrapper, config.script_ranges, config.max_seq_length
    )
    
    val_dataset = IndicTextDataset(
        val_files, morfessor_wrapper, config.script_ranges, config.max_seq_length
    )
    
    test_dataset = IndicTextDataset(
        test_files, morfessor_wrapper, config.script_ranges, config.max_seq_length
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    return train_loader, val_loader, test_loader, morfessor_wrapper