#!/usr/bin/env python3
"""
Token Mapping Generation Script for Embeddageddon

This script generates token mappings by:
1. Loading LLaMA tokenizer vocabulary
2. Loading all embedding dictionaries from different models
3. For each LLaMA token, collecting embeddings from all available models
4. Running collected embeddings through a trained embedding model encoder
5. Saving bottleneck representations as new token mappings

Usage:
    python generate_token_mappings.py --encoder_model_path <path> --output_dir embeddageddon_embeddings
"""

import os
import pickle
import argparse
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional
import torch
from transformers import AutoTokenizer
from tqdm import tqdm
import logging

from autoencoder.model import EmbeddingAutoencoder

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_embedding_dicts(embedding_dicts_dir: str) -> Dict[str, Dict[str, np.ndarray]]:
    """Load all embedding dictionaries from the embedding_dicts directory.
    
    Args:
        embedding_dicts_dir: Path to directory containing embedding pickle files
        
    Returns:
        Dictionary mapping model names to their embedding dictionaries
    """
    embedding_dicts = {}
    embedding_dir = Path(embedding_dicts_dir)
    
    logger.info(f"Loading embedding dictionaries from {embedding_dir}")
    
    for pkl_file in embedding_dir.glob("*.pkl"):
        model_name = pkl_file.stem  # Remove .pkl extension
        logger.info(f"Loading {model_name}...")
        
        try:
            with open(pkl_file, 'rb') as f:
                emb_dict = pickle.load(f)
            embedding_dicts[model_name] = emb_dict
            logger.info(f"Loaded {len(emb_dict)} tokens from {model_name}")
        except Exception as e:
            logger.error(f"Failed to load {pkl_file}: {e}")
    
    logger.info(f"Successfully loaded {len(embedding_dicts)} embedding dictionaries")
    return embedding_dicts


def get_llama_tokenizer() -> AutoTokenizer:
    """Load LLaMA tokenizer to get the target vocabulary.
    
    Returns:
        Loaded LLaMA tokenizer
    """
    logger.info("Loading LLaMA tokenizer...")
    
    model_name = "NousResearch/Llama-3.2-1B"
    
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        logger.info(f"Successfully loaded tokenizer from {model_name}")
        logger.info(f"Vocabulary size: {len(tokenizer.get_vocab())}")
        return tokenizer
    except Exception as e:
        raise RuntimeError(f"Could not load tokenizer from {model_name}: {e}")


def collect_token_embeddings(token: str, embedding_dicts: Dict[str, Dict[str, np.ndarray]],
                           model_dims: Dict[str, int]) -> Optional[np.ndarray]:
    """Collect embeddings for a given token from all available models with consistent ordering.
    
    Args:
        token: Token to collect embeddings for
        embedding_dicts: Dictionary of model embeddings
        model_dims: Dictionary mapping model names to their embedding dimensions
        
    Returns:
        Concatenated embedding vector with consistent length or None if no embeddings found
    """
    # Create a consistent ordered list of model names
    model_names = sorted(embedding_dicts.keys())
    embeddings_parts = []
    
    for model_name in model_names:
        emb_dict = embedding_dicts[model_name]
        if token in emb_dict:
            # Use the actual embedding
            embeddings_parts.append(emb_dict[token])
        else:
            # Use zero padding with the correct dimension for this model
            dim = model_dims[model_name]
            embeddings_parts.append(np.zeros(dim, dtype=np.float32))
    
    if not embeddings_parts:
        return None
    
    return np.concatenate(embeddings_parts, axis=0)


def load_trained_encoder(encoder_path: str, input_dim: int, bottleneck_dim: int) -> Optional[EmbeddingAutoencoder]:
    """Load a pre-trained encoder model.
    
    Args:
        encoder_path: Path to the saved encoder model
        input_dim: Expected input dimension
        bottleneck_dim: Expected bottleneck dimension
        
    Returns:
        Loaded encoder model or None if not found
    """
    if not os.path.exists(encoder_path):
        logger.warning(f"Encoder model not found at {encoder_path}")
        return None
    
    encoder = EmbeddingAutoencoder(input_dim, bottleneck_dim)
    encoder.load_state_dict(torch.load(encoder_path, map_location='cpu')["model_state_dict"])
    encoder.eval()
    encoder.to("cuda")
    logger.info(f"Loaded encoder from {encoder_path}")
    return encoder


def process_tokens_to_mappings(
    tokenizer: AutoTokenizer,
    embedding_dicts: Dict[str, Dict[str, np.ndarray]],
    encoder: EmbeddingAutoencoder,
    output_dir: str
) -> Dict[str, np.ndarray]:
    """Process all tokens to create bottleneck embeddings.
    
    Args:
        tokenizer: LLaMA tokenizer
        embedding_dicts: Dictionary of model embeddings
        encoder: Encoder model to compress embeddings
        output_dir: Directory to save results
        
    Returns:
        Dictionary mapping tokens to bottleneck embeddings
    """
    vocab = list(tokenizer.get_vocab().keys())
    token_mappings = {}
    
    logger.info(f"Processing {len(vocab)} tokens...")
    
    # Calculate model dimensions for verification
    model_dims = {}
    total_input_dim = 0
    for model_name, emb_dict in embedding_dicts.items():
        if emb_dict:  # Check if dictionary is not empty
            first_token = next(iter(emb_dict.keys()))
            dim = emb_dict[first_token].shape[0]
            model_dims[model_name] = dim
            total_input_dim += dim
    
    logger.info(f"Model dimensions: {model_dims}")
    logger.info(f"Total input dimension: {total_input_dim}")
    
    # Process tokens in batches for efficiency
    batch_size = 1000
    processed_tokens = 0
    
    with torch.no_grad():
        for i in tqdm(range(0, len(vocab), batch_size), desc="Processing token batches"):
            batch_tokens = vocab[i:i + batch_size]
            batch_embeddings = []
            batch_token_names = []
            
            for token in batch_tokens:
                # Collect embeddings for this token
                concat_embedding = collect_token_embeddings(token, embedding_dicts, model_dims)
                
                if concat_embedding is not None:
                    batch_embeddings.append(concat_embedding)
                    batch_token_names.append(token)
            
            # Process batch through encoder if we have embeddings
            if batch_embeddings:
                batch_tensor = torch.from_numpy(np.stack(batch_embeddings)).float().to("cuda")
                bottleneck_embeddings = encoder.encode(batch_tensor).cpu().numpy()
                
                # Store results
                for token, bottleneck_emb in zip(batch_token_names, bottleneck_embeddings):
                    token_mappings[token] = bottleneck_emb
            
            processed_tokens += len(batch_tokens)
    
    logger.info(f"Successfully processed {len(token_mappings)} tokens out of {len(vocab)} total tokens")
    
    # Save metadata
    metadata = {
        "total_tokens": len(vocab),
        "processed_tokens": len(token_mappings),
        "model_dims": model_dims,
        "total_input_dim": total_input_dim,
        #"bottleneck_dim": encoder.encoder[2].out_features if hasattr(encoder.encoder[2], 'out_features') else None
    }
    os.makedirs(output_dir, exist_ok=True)
    metadata_path = os.path.join(output_dir, "metadata.json")
    import json
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    return token_mappings


def save_token_mappings(token_mappings: Dict[str, np.ndarray], output_dir: str, bottleneck_dims: List[int]):
    """Save token mappings for different bottleneck sizes based on MatFormer scale factors.
    
    Args:
        token_mappings: Dictionary mapping tokens to bottleneck embeddings
        output_dir: Directory to save the mappings
        bottleneck_dims: List of bottleneck dimensions corresponding to scale factors
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    scale_factors = [0.125, 0.25, 0.5, 1.0]
    scale_names = ['s', 'm', 'l', 'xl']
    
    # Get the full bottleneck dimension
    if token_mappings:
        full_dim = list(token_mappings.values())[0].shape[0]
    else:
        logger.error("No token mappings to save!")
        return
    
    logger.info(f"Saving token mappings with full dimension: {full_dim}")
    
    for scale_factor, scale_name in zip(scale_factors, scale_names):
        subset_dim = int(full_dim * scale_factor)
        subset_mappings = {}
        
        # Create subset by taking first subset_dim dimensions
        for token, embedding in token_mappings.items():
            subset_mappings[token] = embedding[:subset_dim]
        
        # Save subset mapping
        output_file = output_path / f"embeddageddon_embeddings_{scale_name}_{subset_dim}d.pkl"
        with open(output_file, 'wb') as f:
            pickle.dump(subset_mappings, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        logger.info(f"Saved {len(subset_mappings)} embeddings ({subset_dim}D) to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Generate token mappings for Embeddageddon LLM training")
    parser.add_argument("--embedding_dicts_dir", default="embeddageddon/embedding_dicts",
                       help="Directory containing embedding pickle files")
    parser.add_argument("--encoder_model_path", default=None,
                       help="Path to pre-trained encoder model")
    parser.add_argument("--output_dir", default="embeddageddon/embeddageddon_embeddings",
                       help="Output directory for token mappings")
    parser.add_argument("--bottleneck_dim", type=int, default=7168,
                       help="Bottleneck dimension for encoder")
    
    args = parser.parse_args()
    
    assert args.encoder_model_path is not None, "You must provide an autoencoder model path!"

    try:
        # Load embedding dictionaries
        embedding_dicts = load_embedding_dicts(args.embedding_dicts_dir)
        if not embedding_dicts:
            logger.error("No embedding dictionaries loaded!")
            return
        
        # Load LLaMA tokenizer
        tokenizer = get_llama_tokenizer()
        
        # Calculate total input dimension
        total_input_dim = 0
        for model_name, emb_dict in embedding_dicts.items():
            if emb_dict:
                first_token = next(iter(emb_dict.keys()))
                dim = emb_dict[first_token].shape[0]
                total_input_dim += dim
        
        # Load or create encoder
        
        encoder = load_trained_encoder(args.encoder_model_path, total_input_dim, args.bottleneck_dim)
        assert encoder is not None, "Error loading encoder!"


        # Process tokens to create mappings
        token_mappings = process_tokens_to_mappings(tokenizer, embedding_dicts, encoder, args.output_dir)
        
        # Save token mappings for different scale factors
        save_token_mappings(token_mappings, args.output_dir, [args.bottleneck_dim])
        
        logger.info("Token mapping generation completed successfully!")
        
    except Exception as e:
        logger.error(f"Token mapping generation failed: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
