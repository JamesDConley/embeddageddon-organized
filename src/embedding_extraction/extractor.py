"""
Generic efficient embedding extractor using HuggingFace Hub downloads.

This module provides a unified extractor that works with any model type
for embedding extraction, downloading only the necessary files (index and 
specific safetensors) instead of the full model.
"""

from typing import Dict, List, Optional
import numpy as np

from .interface import EmbeddingModel
from .extractor_utils import (
    create_model_directories,
    load_tokenizer,
    download_and_load_embedding_weights,
    get_vocab_from_tokenizer,
    get_token_embedding,
    get_all_token_embeddings
)


class GenericEfficientEmbeddingModel(EmbeddingModel):
    """Generic efficient embedding extraction for any model using HF Hub downloads."""
    
    def __init__(self, model_name: str, model_dir: str = "models"):
        """Initialize the generic embedding extractor.
        
        Args:
            model_name: HuggingFace model name/path
            model_dir: Directory for storing model files
        """
        self.model_name = model_name
        self.model_dir = model_dir
        self.tokenizer = None
        self.embedding_weights = None
        
        # Create directories
        self.indexes_dir, self.tensors_dir = create_model_directories(model_dir)
        
        # Load tokenizer
        self.tokenizer = load_tokenizer(model_name)
        
        # Download and load embedding weights
        self.embedding_weights = download_and_load_embedding_weights(
            model_name, self.indexes_dir, self.tensors_dir
        )
    
    def get_vocab(self) -> List[str]:
        """Return all tokens in the vocabulary."""
        return get_vocab_from_tokenizer(self.tokenizer)
    
    def get_embedding(self, token: str) -> Optional[np.ndarray]:
        """Return the embedding for a specific token."""
        return get_token_embedding(token, self.tokenizer, self.embedding_weights)
    
    def get_all_embeddings(self) -> Dict[str, np.ndarray]:
        """Return a dict mapping each token to its embedding."""
        return get_all_token_embeddings(self.tokenizer, self.embedding_weights)
