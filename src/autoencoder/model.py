
from typing import Dict, List, Tuple, Optional
import torch.nn as nn
import torch

class EmbeddingAutoencoder(nn.Module):
    """Autoencoder for multi-model token embeddings with RMSNorm, SiLU, and random subnetwork masking."""

    def __init__(self, input_dim: int, bottleneck_dim: int, hidden_dims: Optional[List[int]] = None):
        """
        Initialize the autoencoder.

        Args:
            input_dim: Total dimension of concatenated input embeddings
            bottleneck_dim: Dimension of the bottleneck layer
            hidden_dims: Optional list of hidden layer dimensions
        """
        super().__init__()

        self.input_dim = input_dim
        self.bottleneck_dim = bottleneck_dim

        # Scale factors for subnetwork sizes: s, m, l, xl
        self.scale_factors = [1/8, 1/4, 1/2, 1.0]
        self.current_subset_hd = None  # Will be set by configure_subnetwork

        # Default hidden dimensions if not provided
        if hidden_dims is None:
            hidden_dims = [int(bottleneck_dim * 3), int(bottleneck_dim * 1.5)]

        # Store dimensions for RMSNorm
        self.hidden_dims = hidden_dims

        # Encoder layers with RMSNorm
        self.encoder_layers = nn.ModuleList()
        self.encoder_rms_norms = nn.ModuleList()
        prev_dim = input_dim

        for hidden_dim in hidden_dims:
            hidden_dim = int(hidden_dim)  # Ensure integer

            # Linear layer
            self.encoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            # RMSNorm for stability
            self.encoder_rms_norms.append(nn.RMSNorm(hidden_dim))

            prev_dim = hidden_dim

        # Bottleneck layer
        self.bottleneck_layer = nn.Linear(prev_dim, bottleneck_dim)
        # Tanh activation after bottleneck
        self.bottleneck_tanh = nn.Tanh()

        # Decoder layers with RMSNorm (mirror of encoder)
        self.decoder_layers = nn.ModuleList()
        self.decoder_rms_norms = nn.ModuleList()
        prev_dim = bottleneck_dim

        for hidden_dim in reversed(hidden_dims):
            hidden_dim = int(hidden_dim)  # Ensure integer

            # Linear layer
            self.decoder_layers.append(nn.Linear(prev_dim, hidden_dim))
            # RMSNorm for stability
            self.decoder_rms_norms.append(nn.RMSNorm(hidden_dim))

            prev_dim = hidden_dim

        # Output layer
        self.output_layer = nn.Linear(prev_dim, input_dim)

        # Activation and dropout layers (shared)
        #self.dropout = nn.Dropout(0.1)
        self.silu = nn.SiLU()  # Swish activation

    def configure_subnetwork(self, flag: str):
        """
        Select a subnetwork size for the current batch.

        Args:
            flag: One of 's', 'm', 'l', 'xl'.
        """
        if flag not in {'s', 'm', 'l', 'xl'}:
            raise ValueError(f"Invalid subnetwork flag '{flag}'. Must be one of s,m,l,xl.")
        idx = {'s': 0, 'm': 1, 'l': 2, 'xl': 3}[flag]
        scale = self.scale_factors[idx]
        self.current_subset_hd = int(self.bottleneck_dim * scale)

    def forward(self, x):
        """Forward pass through the autoencoder with RMSNorm, SiLU, and random subnetwork masking."""
        # Encoder forward pass
        encoded = x

        for i in range(len(self.hidden_dims)):
            encoded = self.encoder_layers[i](encoded)
            encoded = self.encoder_rms_norms[i](encoded)
            encoded = self.silu(encoded)
            #encoded = self.dropout(encoded)

        # Bottleneck
        encoded = self.bottleneck_layer(encoded)
        encoded = self.bottleneck_tanh(encoded)

        # Apply subnetwork mask if configured
        if self.current_subset_hd is not None:
            mask = torch.zeros_like(encoded)
            mask[:, :self.current_subset_hd] = 1.0
            encoded = encoded * mask

        # Decoder forward pass
        decoded = encoded

        for i in range(len(self.hidden_dims)):
            decoded = self.decoder_layers[i](decoded)
            decoded = self.decoder_rms_norms[i](decoded)
            decoded = self.silu(decoded)
            #decoded = self.dropout(decoded)

        # Output layer
        decoded = self.output_layer(decoded)

        return decoded

    def encode(self, x):
        """Encode input to bottleneck representation with RMSNorm and SiLU."""
        encoded = x

        for i in range(len(self.hidden_dims)):
            encoded = self.encoder_layers[i](encoded)
            encoded = self.encoder_rms_norms[i](encoded)
            encoded = self.silu(encoded)
            #encoded = self.dropout(encoded)

        encoded = self.bottleneck_layer(encoded)
        encoded = self.bottleneck_tanh(encoded)

        if self.current_subset_hd is not None:
            mask = torch.zeros_like(encoded)
            mask[:, :self.current_subset_hd] = 1.0
            encoded = encoded * mask

        return encoded

def masked_mse_loss(predictions: torch.Tensor, targets: torch.Tensor, 
                   masks: torch.Tensor) -> torch.Tensor:
    """
    Calculate MSE loss only for masked (non-zero) elements.
    
    Args:
        predictions: Model predictions
        targets: Target values
        masks: Binary masks (1.0 for elements to include in loss, 0.0 to exclude)
        
    Returns:
        Masked MSE loss
    """
    # Element-wise squared differences
    squared_diffs = (predictions - targets) ** 2
    
    # Apply mask
    masked_diffs = squared_diffs * masks
    
    # Calculate mean only over masked elements
    total_masked_elements = masks.sum()
    if total_masked_elements > 0:
        return masked_diffs.sum() / total_masked_elements
    else:
        print("All elements masked. This means there's no signal for this batch!")
        raise ValueError("All elements masked. This means there's no signal for this batch!")
        return torch.tensor(0.0, device=predictions.device)