"""
Memory-mapped dataset for efficient streaming from disk.

This dataset class loads preprocessed embeddings from memory-mapped numpy arrays,
avoiding memory leaks and enabling training on datasets larger than RAM.
"""

from typing import Tuple
import torch
from torch.utils.data import Dataset
import numpy as np
import json
from pathlib import Path


class MemMapEmbeddingDataset(Dataset):
    """Memory-mapped dataset for multi-model token embeddings."""

    def __init__(self, preprocessed_dir: str):
        """
        Initialize the dataset from preprocessed files.

        Args:
            preprocessed_dir: Directory containing preprocessed memmap files
        """
        self.preprocessed_dir = Path(preprocessed_dir)

        # Load metadata
        metadata_path = self.preprocessed_dir / "metadata.json"
        with open(metadata_path, 'r') as f:
            metadata = json.load(f)

        self.num_samples = metadata["num_samples"]
        self.total_dim = metadata["total_dim"]
        self.bottleneck_dim = metadata["bottleneck_dim"]
        self.model_names = metadata["model_names"]
        self.model_dims = metadata["model_dims"]
        self.common_tokens = metadata["common_tokens"]

        # Open memory-mapped arrays (read-only mode)
        self.inputs_mmap = np.memmap(
            self.preprocessed_dir / "inputs.npy",
            dtype='float32',
            mode='r',
            shape=(self.num_samples, self.total_dim)
        )

        self.masks_mmap = np.memmap(
            self.preprocessed_dir / "masks.npy",
            dtype='float32',
            mode='r',
            shape=(self.num_samples, self.total_dim)
        )

        print(f"Loaded memory-mapped dataset from: {self.preprocessed_dir}")
        print(f"Samples: {self.num_samples}")
        print(f"Total dimension: {self.total_dim}")
        print(f"Bottleneck dimension: {self.bottleneck_dim}")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a training sample.

        Returns:
            input_tensor: Concatenated embeddings from all models
            target_tensor: Same as input_tensor (autoencoder target)
            mask_tensor: Binary mask indicating which parts contribute to loss
        """
        # Read from memmap (only loads this row from disk, not entire array)
        input_array = self.inputs_mmap[idx]
        mask_array = self.masks_mmap[idx]

        # Convert to PyTorch tensors (creates a copy, not a view)
        input_tensor = torch.from_numpy(input_array.copy())
        mask_tensor = torch.from_numpy(mask_array.copy())

        # For autoencoder, target is same as input
        target_tensor = input_tensor.clone()

        return input_tensor, target_tensor, mask_tensor
