"""
Common utility functions for efficient embedding extraction across different model types.

This module contains shared functionality used by all model-specific extractors,
including directory management, file downloading, index parsing, and embedding loading.
"""

import json
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import torch
from safetensors import safe_open
from transformers import AutoTokenizer
from huggingface_hub import hf_hub_download


def create_model_directories(model_dir: str) -> Tuple[Path, Path]:
    """Create and return the indexes and tensors directories for a model.
    
    Args:
        model_dir: Base directory for model storage
        
    Returns:
        Tuple of (indexes_dir, tensors_dir) paths
    """
    indexes_dir = Path(model_dir) / "indexes"
    tensors_dir = Path(model_dir) / "embedding_tensors"
    indexes_dir.mkdir(parents=True, exist_ok=True)
    tensors_dir.mkdir(parents=True, exist_ok=True)
    return indexes_dir, tensors_dir


def load_tokenizer(model_name: str) -> Optional[AutoTokenizer]:
    """Load a tokenizer for the specified model.
    
    Args:
        model_name: HuggingFace model name/path
        
    Returns:
        Loaded tokenizer or None if failed
    """
    print(f"  Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(
        model_name, 
        trust_remote_code=True
    )
    print(f"  Tokenizer loaded with vocab size: {len(tokenizer.get_vocab())}")
    return tokenizer


def download_model_index(model_name: str, indexes_dir: Path) -> Optional[Path]:
    """Download the model's safetensors index file.
    
    Args:
        model_name: HuggingFace model name/path
        indexes_dir: Directory to store index files
        
    Returns:
        Path to downloaded index file or None if failed
    """
    try:
        # Check if index file already exists
        local_model_dir = indexes_dir / model_name.replace("/", "--")
        expected_index_path = local_model_dir / "model.safetensors.index.json"
        
        if expected_index_path.exists():
            print(f"  Model index already exists at: {expected_index_path}")
            return expected_index_path
        
        print(f"  Downloading model index for {model_name}...")
        
        # Download the index file
        index_path = hf_hub_download(
            repo_id=model_name,
            filename="model.safetensors.index.json",
            local_dir=local_model_dir,
            local_dir_use_symlinks=False
        )
        
        print(f"  Model index downloaded to: {index_path}")
        return Path(index_path)
        
    except Exception as e:
        print(f"  Error downloading model index: {e}")
        raise e


def find_embedding_file(index_path: Path, embedding_key: str = "model.embed_tokens.weight") -> Optional[str]:
    """Find which safetensors file contains the embedding weights.
    
    Args:
        index_path: Path to the model index file
        embedding_key: Key name for embedding weights in the model
        
    Returns:
        Filename containing embeddings or None if not found
    """
    try:
        with open(index_path, 'r') as f:
            index_data = json.load(f)
        
        weight_map = index_data.get("weight_map", {})
        
        if embedding_key in weight_map:
            embedding_file = weight_map[embedding_key]
            print(f"  Found embedding layer '{embedding_key}' in file: {embedding_file}")
            return embedding_file
        
        print(f"  Error: Embedding key '{embedding_key}' not found in weight map.")
        print(f"  Available keys (first 10):")
        for key in sorted(weight_map.keys())[:10]:
            print(f"    {key}")
        return None
        
    except Exception as e:
        print(f"  Error reading index file: {e}")
        raise e


def download_embedding_file(model_name: str, embedding_filename: str, tensors_dir: Path) -> Optional[Path]:
    """Download the specific safetensors file containing embeddings.
    
    Args:
        model_name: HuggingFace model name/path
        embedding_filename: Name of the safetensors file to download
        tensors_dir: Directory to store tensor files
        
    Returns:
        Path to downloaded embedding file or None if failed
    """
    try:
        # Check if embedding file already exists
        local_model_dir = tensors_dir / model_name.replace("/", "--")
        expected_embedding_path = local_model_dir / embedding_filename
        
        if expected_embedding_path.exists():
            print(f"  Embedding file already exists at: {expected_embedding_path}")
            return expected_embedding_path
        
        print(f"  Downloading embedding file: {embedding_filename}...")
        
        # Download the specific safetensors file
        embedding_path = hf_hub_download(
            repo_id=model_name,
            filename=embedding_filename,
            local_dir=local_model_dir,
            local_dir_use_symlinks=False
        )
        
        print(f"  Embedding file downloaded to: {embedding_path}")
        return Path(embedding_path)
        
    except Exception as e:
        print(f"  Error downloading embedding file: {e}")
        raise e


def load_embedding_weights(embedding_file_path: Path, embedding_key: str = "model.embed_tokens.weight") -> Optional[np.ndarray]:
    """Load embedding weights from a safetensors file.
    
    Args:
        embedding_file_path: Path to the safetensors file
        embedding_key: Key name for embedding weights in the file
        
    Returns:
        Numpy array of embedding weights or None if failed
    """
    try:
        print(f"  Loading embedding weights from {embedding_file_path.name}...")
        
        with safe_open(embedding_file_path, framework="pt", device="cpu") as f:
            if embedding_key in f.keys():
                tensor = f.get_tensor(embedding_key)
                # Convert to float32 if needed
                if tensor.dtype == torch.bfloat16:
                    tensor = tensor.to(torch.float32)
                embedding_weights = tensor.numpy()
                print(f"  Loaded embedding weights: {embedding_weights.shape}")
                return embedding_weights
            
            print(f"  Error: Embedding tensor '{embedding_key}' not found.")
            print(f"  Available tensors (first 10):")
            for key in list(f.keys())[:10]:
                print(f"    {key}")
            return None
                
    except Exception as e:
        print(f"  Error loading embedding weights: {e}")
        raise e


def download_and_load_embedding_weights(
    model_name: str, 
    indexes_dir: Path, 
    tensors_dir: Path,
    embedding_key: str = "model.embed_tokens.weight"
) -> Optional[np.ndarray]:
    """Complete pipeline to download and load embedding weights for a model.
    
    Args:
        model_name: HuggingFace model name/path
        indexes_dir: Directory for storing index files
        tensors_dir: Directory for storing tensor files
        embedding_key: Key name for embedding weights
        
    Returns:
        Numpy array of embedding weights or None if failed
    """
    print(f"  Finding embedding weights for {model_name}...")
    
    # Download model index
    index_path = download_model_index(model_name, indexes_dir)
    if not index_path:
        return None
    
    # Find which file contains embeddings
    embedding_filename = find_embedding_file(index_path, embedding_key)
    if not embedding_filename:
        return None
    
    # Download the specific embedding file
    embedding_file_path = download_embedding_file(model_name, embedding_filename, tensors_dir)
    if not embedding_file_path:
        return None
    
    # Load the embedding weights
    return load_embedding_weights(embedding_file_path, embedding_key)


def get_vocab_from_tokenizer(tokenizer: Optional[AutoTokenizer]) -> List[str]:
    """Get vocabulary list from a tokenizer.
    
    Args:
        tokenizer: Loaded tokenizer
        
    Returns:
        List of vocabulary tokens
    """
    if not tokenizer:
        return []
    return list(tokenizer.get_vocab().keys())


def get_token_embedding(
    token: str, 
    tokenizer: Optional[AutoTokenizer], 
    embedding_weights: Optional[np.ndarray]
) -> Optional[np.ndarray]:
    """Get embedding for a specific token.
    
    Args:
        token: Token to get embedding for
        tokenizer: Loaded tokenizer
        embedding_weights: Numpy array of embedding weights
        
    Returns:
        Embedding vector for the token or None if not found
    """
    if not tokenizer or embedding_weights is None:
        return None
    
    try:
        token_id = tokenizer.convert_tokens_to_ids(token)
        if token_id is None or token_id < 0 or token_id >= len(embedding_weights):
            return None
        
        return embedding_weights[token_id]
    except (AssertionError, Exception) as e:
        # Handle cases where token cannot be converted (e.g., Mistral tokenizer issues)
        # This can happen with some tokenizers where vocab entries are not valid individual tokens
        print(f"    Warning: Could not convert token '{token}' to ID: {e}")
        return None


def get_all_token_embeddings(
    tokenizer: Optional[AutoTokenizer], 
    embedding_weights: Optional[np.ndarray]
) -> Dict[str, np.ndarray]:
    """Get embeddings for all tokens in the vocabulary.
    
    Args:
        tokenizer: Loaded tokenizer
        embedding_weights: Numpy array of embedding weights
        
    Returns:
        Dictionary mapping tokens to their embedding vectors
    """
    if not tokenizer or embedding_weights is None:
        return {}
    
    vocab = get_vocab_from_tokenizer(tokenizer)
    embeddings_dict = {}
    
    for token in vocab:
        embedding = get_token_embedding(token, tokenizer, embedding_weights)
        if embedding is not None:
            embeddings_dict[token] = embedding
    
    return embeddings_dict


def save_embeddings_pickle(embeddings_dict: Dict[str, np.ndarray], output_path: str) -> bool:
    """Save embeddings dictionary to pickle format (fast and efficient for numpy arrays).
    
    Args:
        embeddings_dict: Dictionary mapping tokens to embedding vectors
        output_path: Path to save the pickle file
        
    Returns:
        True if successful, False otherwise
    """
    try:
        print(f"  Saving embeddings to pickle format: {output_path}")
        with open(output_path, 'wb') as f:
            pickle.dump(embeddings_dict, f, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  Successfully saved {len(embeddings_dict)} embeddings to pickle")
        return True
    except Exception as e:
        print(f"  Error saving pickle file: {e}")
        return False


def load_embeddings_pickle(input_path: str) -> Dict[str, np.ndarray]:
    """Load embeddings dictionary from pickle format.
    
    Args:
        input_path: Path to the pickle file
        
    Returns:
        Dictionary mapping tokens to embedding vectors
    """
    try:
        print(f"  Loading embeddings from pickle format: {input_path}")
        with open(input_path, 'rb') as f:
            embeddings_dict = pickle.load(f)
        print(f"  Successfully loaded {len(embeddings_dict)} embeddings from pickle")
        return embeddings_dict
    except Exception as e:
        print(f"  Error loading pickle file: {e}")
        return {}
