#!/usr/bin/env python3

"""
Preprocess embedding dictionaries into efficient memory-mapped format.

This script converts the embedding dictionaries into a compact binary format
that can be streamed from disk without loading everything into memory.
Uses numpy's memmap for efficient disk-based storage.
"""

import argparse
import numpy as np
from pathlib import Path
import json
from tqdm import tqdm

from utils.embeddings import load_embedding_dicts
from utils.tokens import get_common_tokens


def preprocess_embeddings(embedding_dir: str, output_dir: str):
    """
    Preprocess embeddings into memory-mapped arrays.

    Args:
        embedding_dir: Directory containing embedding pickle files
        output_dir: Directory to save preprocessed data
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PREPROCESSING EMBEDDINGS")
    print("=" * 60)

    # Load embedding dictionaries
    print("Loading embedding dictionaries...")
    embedding_dicts = load_embedding_dicts(embedding_dir)

    # Get common tokens
    print("Finding common tokens...")
    common_tokens = get_common_tokens(embedding_dicts)
    print(f"Found {len(common_tokens)} unique tokens")

    # Get model names in consistent order
    model_names = sorted(embedding_dicts.keys())
    print(f"Models: {model_names}")

    # Calculate dimensions
    model_dims = {}
    for model_name in model_names:
        if model_name in embedding_dicts and embedding_dicts[model_name]:
            first_token = next(iter(embedding_dicts[model_name].keys()))
            model_dims[model_name] = len(embedding_dicts[model_name][first_token])
        else:
            model_dims[model_name] = 0

    total_dim = sum(model_dims.values())
    max_dim = max(model_dims.values()) if model_dims else 128
    bottleneck_dim = int(max_dim)

    print(f"Model dimensions: {model_dims}")
    print(f"Total input dimension: {total_dim}")
    print(f"Bottleneck dimension: {bottleneck_dim}")
    print("-" * 60)

    # Create memory-mapped arrays
    num_samples = len(common_tokens)

    print(f"Creating memory-mapped arrays for {num_samples} samples...")

    # Create memmap files
    inputs_path = output_path / "inputs.npy"
    masks_path = output_path / "masks.npy"

    # Initialize memmaps (float32 for memory efficiency)
    inputs_mmap = np.memmap(
        inputs_path,
        dtype='float32',
        mode='w+',
        shape=(num_samples, total_dim)
    )

    masks_mmap = np.memmap(
        masks_path,
        dtype='float32',
        mode='w+',
        shape=(num_samples, total_dim)
    )

    # Fill the arrays
    print("Processing tokens...")
    for idx, token in enumerate(tqdm(common_tokens)):
        input_vector = []
        mask_vector = []

        for model_name in model_names:
            model_dim = model_dims[model_name]

            if (model_name in embedding_dicts and
                token in embedding_dicts[model_name]):
                # Token exists in this model
                embedding = embedding_dicts[model_name][token]
                input_vector.extend(embedding)
                mask_vector.extend([1.0] * model_dim)
            else:
                # Token doesn't exist - zero pad
                input_vector.extend([0.0] * model_dim)
                mask_vector.extend([0.0] * model_dim)

        # Write to memmap
        inputs_mmap[idx] = input_vector
        masks_mmap[idx] = mask_vector

    # Flush to disk
    print("Flushing data to disk...")
    inputs_mmap.flush()
    masks_mmap.flush()

    # Save metadata
    metadata = {
        "num_samples": num_samples,
        "total_dim": total_dim,
        "bottleneck_dim": bottleneck_dim,
        "model_names": model_names,
        "model_dims": model_dims,
        "common_tokens": common_tokens
    }

    metadata_path = output_path / "metadata.json"
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)

    print("=" * 60)
    print("PREPROCESSING COMPLETE!")
    print(f"Output directory: {output_path}")
    print(f"Inputs: {inputs_path} ({inputs_mmap.nbytes / 1024**2:.2f} MB)")
    print(f"Masks: {masks_path} ({masks_mmap.nbytes / 1024**2:.2f} MB)")
    print(f"Metadata: {metadata_path}")
    print("=" * 60)


def main():
    parser = argparse.ArgumentParser(description="Preprocess embeddings for training")
    parser.add_argument("--embedding_dir", type=str, default="embedding_dicts",
                       help="Directory containing embedding pickle files")
    parser.add_argument("--output_dir", type=str, default="preprocessed_embeddings",
                       help="Directory to save preprocessed data")

    args = parser.parse_args()
    preprocess_embeddings(args.embedding_dir, args.output_dir)


if __name__ == "__main__":
    main()
