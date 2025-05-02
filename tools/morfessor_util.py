"""Utilities for working with Morfessor for morphological analysis."""
import morfessor
import numpy as np
from typing import Dict, List, Tuple, Union

class MorfessorWrapper:
    """Wrapper for Morfessor models with utilities for Indian languages."""
    
    def __init__(self, model_paths: Dict[str, str]):
        """
        Initialize Morfessor models for each language.
        
        Args:
            model_paths: Dictionary mapping language codes to paths of Morfessor models
        """
        self.models = {}
        for lang, path in model_paths.items():
            try:
                io = morfessor.MorfessorIO()
                self.models[lang] = io.read_binary_model_file(path)
                print(f"Loaded Morfessor model for {lang}")
            except Exception as e:
                print(f"Error loading Morfessor model for {lang}: {e}")
    
    def get_morphemes_and_boundaries(self, text: str, lang: str) -> Tuple[List[str], List[int]]:
        """
        Get morphemes and boundary labels for a text.
        
        Args:
            text: Input text
            lang: Language code
            
        Returns:
            Tuple of (morphemes, boundaries)
            - morphemes: List of individual morphemes
            - boundaries: List of 0/1 values where 1 indicates a morpheme boundary
        """
        if lang not in self.models:
            raise ValueError(f"No Morfessor model for language: {lang}")
        
        model = self.models[lang]
        tokens = text.strip().split()
        all_morphemes = []
        boundaries = []
        
        for token in tokens:
            if not token:  # Skip empty tokens
                continue
                
            # Segment with Morfessor
            morphemes, _ = model.viterbi_segment(token)
            all_morphemes.extend(morphemes)
            
            # Create boundary indicators (1 at morpheme boundary, 0 elsewhere)
            char_boundaries = [0] * len(token)
            pos = 0
            for morpheme in morphemes[:-1]:  # Skip last boundary
                pos += len(morpheme)
                if pos - 1 < len(char_boundaries):
                    char_boundaries[pos-1] = 1
            
            boundaries.extend(char_boundaries)
            
            # Add space boundary after each token except the last
            if token != tokens[-1]:
                boundaries.append(1)  # Space is a boundary
        
        return all_morphemes, boundaries

    def compute_compression_ratio(self, texts: List[str], lang: str) -> float:
        """
        Compute morpheme-based compression ratio for a batch of texts.
        
        Args:
            texts: List of input texts
            lang: Language code
            
        Returns:
            Average compression ratio (bytes/morphemes)
        """
        total_bytes = 0
        total_morphemes = 0
        
        for text in texts:
            morphemes, _ = self.get_morphemes_and_boundaries(text, lang)
            text_bytes = len(text.encode('utf-8'))
            
            total_bytes += text_bytes
            total_morphemes += len(morphemes)
        
        if total_morphemes == 0:
            return 1.0  # Default ratio
            
        return total_bytes / total_morphemes


def detect_language(text: str, script_ranges: Dict[str, List[Tuple[int, int]]]) -> str:
    """
    Detect the language of a text based on Unicode script ranges.
    
    Args:
        text: Input text
        script_ranges: Dictionary mapping language codes to Unicode range tuples
        
    Returns:
        Detected language code
    """
    if not text:
        return list(script_ranges.keys())[0]  # Default to first language
    
    # Count characters in each script range
    counts = {lang: 0 for lang in script_ranges}
    
    for char in text:
        code = ord(char)
        for lang, ranges in script_ranges.items():
            for start, end in ranges:
                if start <= code <= end:
                    counts[lang] += 1
                    break
    
    # Return language with highest count
    if sum(counts.values()) == 0:
        return list(script_ranges.keys())[0]  # Default to first language
        
    return max(counts.items(), key=lambda x: x[1])[0]