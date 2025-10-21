
from typing import Dict, List, Tuple
import torch
from torch.utils.data import Dataset


class MultiModelEmbeddingDataset(Dataset):
    """Dataset for multi-model token embeddings."""
    
    def __init__(self, embedding_dicts: Dict[str, Dict[str, List[float]]], 
                 common_tokens: List[str], model_names: List[str]):
        """
        Initialize the dataset.
        
        Args:
            embedding_dicts: Dict mapping model names to their embedding dictionaries
            common_tokens: List of tokens that appear in at least one model
            model_names: List of model names in consistent order
        """
        self.embedding_dicts = embedding_dicts
        self.common_tokens = common_tokens
        self.model_names = model_names
        
        # Calculate embedding dimensions for each model
        self.model_dims = {}
        for model_name in model_names:
            if model_name in embedding_dicts and embedding_dicts[model_name]:
                # Get dimension from first available embedding
                first_token = next(iter(embedding_dicts[model_name].keys()))
                self.model_dims[model_name] = len(embedding_dicts[model_name][first_token])
            else:
                self.model_dims[model_name] = 0
        
        # Calculate total input dimension
        self.total_dim = sum(self.model_dims.values())
        
        # Calculate bottleneck dimension (1.5x largest embedding dimension)
        max_dim = max(self.model_dims.values()) if self.model_dims else 128
        self.bottleneck_dim = int(max_dim)
        
        print(f"Model dimensions: {self.model_dims}")
        print(f"Total input dimension: {self.total_dim}")
        print(f"Bottleneck dimension: {self.bottleneck_dim}")
    
    def __len__(self):
        return len(self.common_tokens)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Get a training sample.
        
        Returns:
            input_tensor: Concatenated embeddings from all models (zero-padded for missing tokens)
            target_tensor: Same as input_tensor (autoencoder target)
            mask_tensor: Binary mask indicating which parts of the output should contribute to loss
        """
        token = self.common_tokens[idx]
        
        # Build concatenated input and mask
        input_parts = []
        mask_parts = []
        
        for model_name in self.model_names:
            model_dim = self.model_dims[model_name]
            
            if (model_name in self.embedding_dicts and 
                token in self.embedding_dicts[model_name]):
                # Token exists in this model
                embedding = self.embedding_dicts[model_name][token]
                input_parts.append(embedding)
                mask_parts.append([1.0] * model_dim)  # Include in loss
            else:
                # Token doesn't exist in this model - zero pad
                input_parts.append([0.0] * model_dim)
                mask_parts.append([0.0] * model_dim)  # Exclude from loss
        
        # Concatenate all parts
        input_vector = []
        mask_vector = []
        for part in input_parts:
            input_vector.extend(part)
        for part in mask_parts:
            mask_vector.extend(part)
        
        input_tensor = torch.tensor(input_vector, dtype=torch.float32)
        target_tensor = input_tensor.clone()  # Autoencoder target is same as input
        mask_tensor = torch.tensor(mask_vector, dtype=torch.float32)
        
        return input_tensor, target_tensor, mask_tensor

