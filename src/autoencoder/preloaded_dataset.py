"""
Preloaded dataset that loads all tensors into RAM for maximum performance.

This dataset class loads all preprocessed embeddings into memory during initialization,
then __getitem__ simply indexes into pre-allocated tensors. This eliminates all disk I/O
and tensor creation overhead during training for maximum GPU utilization.

Use this when you have sufficient RAM to hold the entire dataset.
"""

from typing import Tuple
import torch
from torch.utils.data import Dataset
import numpy as np
import json
from pathlib import Path
from tqdm import tqdm


class PreloadedEmbeddingDataset(Dataset):
    """Preloaded dataset with all tensors in RAM."""

    def __init__(self, preprocessed_dir: str):
        """
        Initialize the dataset by loading all tensors into RAM.

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

        print(f"Loading dataset from: {self.preprocessed_dir}")
        print(f"Samples: {self.num_samples}")
        print(f"Total dimension: {self.total_dim}")
        print(f"Bottleneck dimension: {self.bottleneck_dim}")

        # Open memory-mapped arrays
        inputs_mmap = np.memmap(
            self.preprocessed_dir / "inputs.npy",
            dtype='float32',
            mode='r',
            shape=(self.num_samples, self.total_dim)
        )

        masks_mmap = np.memmap(
            self.preprocessed_dir / "masks.npy",
            dtype='float32',
            mode='r',
            shape=(self.num_samples, self.total_dim)
        )

        # Preload all data into RAM as PyTorch tensors
        print("Preloading all data into RAM...")
        print(f"Expected memory usage: ~{(self.num_samples * self.total_dim * 4 * 2) / 1024**3:.2f} GB")

        # Load inputs (using tqdm for progress bar)
        print("Loading inputs...")
        self.inputs = torch.from_numpy(np.array(inputs_mmap, dtype=np.float32))

        # Load masks
        print("Loading masks...")
        self.masks = torch.from_numpy(np.array(masks_mmap, dtype=np.float32))

        # Targets are same as inputs for autoencoder
        print("Creating targets...")
        self.targets = self.inputs#.clone()

        print("Preloading complete! Dataset ready for training.")
        print(f"Actual memory usage: {(self.inputs.element_size() * self.inputs.nelement() * 3) / 1024**3:.2f} GB")

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a training sample (zero-copy indexing into preloaded tensors).

        Returns:
            input_tensor: Concatenated embeddings from all models
            target_tensor: Same as input_tensor (autoencoder target)
            mask_tensor: Binary mask indicating which parts contribute to loss
        """
        # Simple indexing - returns views, not copies (zero overhead)
        return self.inputs[idx], self.targets[idx], self.masks[idx]
