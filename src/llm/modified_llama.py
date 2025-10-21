"""
Base MatFormer implementation with configurable subnetworks.

This module provides a simplified implementation of MatFormer with subnetwork
configuration support across four sizes (s, m, l, xl) without the gradient
freezing and covariance regularization features found in the frozen variant.

Classes:
    ModifiedLlamaMLP: Extended Llama MLP with basic subnetwork configuration.
    ModifiedLlamaForCausalLM: Modified Llama model with subnetwork management.
"""

import torch.nn.functional as F
from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaMLP


class ModifiedLlamaMLP(LlamaMLP):
    """
    Modified Llama MLP with configurable subnetwork sizes.
    
    This class extends the standard LlamaMLP to support dynamic subnetwork
    configuration across four sizes (s, m, l, xl) by selecting appropriate
    slices of the weight matrices. This is a basic implementation without
    gradient freezing or covariance tracking.
    
    Attributes:
        intermediate_size (int): The full intermediate dimension size.
        scale_factors (list): Scale factors for subnetwork sizes [s, m, l, xl].
        current_subset_hd (int): Current hidden dimension size for subnetwork.
    """
    
    def __init__(self, config, scale_factors):
        """
        Initialize the modified MLP with subnetwork scaling capabilities.
        
        Args:
            config: Model configuration object containing intermediate_size.
            scale_factors (list): List of scale factors for subnetwork sizes [s, m, l, xl].
                Each factor determines the proportion of intermediate_size to use.
        """
        super().__init__(config)
        self.intermediate_size = config.intermediate_size
        self.scale_factors = scale_factors  # List of scale factors for 's', 'm', 'l', 'xl'
        self.current_subset_hd = None
        # Initialize with xl (full network) by default
        self.configure_subnetwork("xl")

    def configure_subnetwork(self, flag):
        """
        Configure subnetwork size based on the specified flag.
        
        This method sets up the current subnetwork configuration by determining
        the appropriate slice of weights to use based on the flag.
        
        Args:
            flag (str): Subnetwork size flag. Must be one of 's', 'm', 'l', or 'xl'.
            
        Raises:
            ValueError: If flag is not one of the valid subnetwork flags.
        """
        hd = self.intermediate_size
        if flag == 's':
            scale = self.scale_factors[0]  # hd/8
        elif flag == 'm':
            scale = self.scale_factors[1]  # hd/4
        elif flag == 'l':
            scale = self.scale_factors[2]  # hd/2
        elif flag == 'xl':
            scale = self.scale_factors[3]  # hd
        else:
            raise ValueError(f"Invalid flag '{flag}'. Must be one of: 's', 'm', 'l', 'xl'")
        
        self.current_subset_hd = int(hd * scale)
    
    def forward(self, x):
        """
        Forward pass through the modified MLP with subnetwork configuration.
        
        This method performs the forward pass using only the weights within
        the current subnetwork slice.
        
        Args:
            x (torch.Tensor): Input tensor with shape (batch_size, seq_length, hidden_size).
            
        Returns:
            torch.Tensor: Output tensor from the MLP with same shape as input.
            
        Raises:
            ValueError: If subnetwork size has not been configured.
        """
        if self.current_subset_hd is None:
            raise ValueError("Subnetwork size not configured. Call `configure_subnetwork` first.")
        gate_proj = self.gate_proj.weight[:self.current_subset_hd]
        up_proj = self.up_proj.weight[:self.current_subset_hd]
        down_proj = self.down_proj.weight[:, :self.current_subset_hd]
        down_proj = F.linear(self.act_fn(F.linear(x, gate_proj) * F.linear(x, up_proj)), down_proj)
        # Don't reset current_subset_hd to None - keep the configuration for subsequent forward passes

        return down_proj


class ModifiedLlamaForCausalLM(LlamaForCausalLM):
    """
    Modified Llama model with configurable subnetworks.
    
    This class extends LlamaForCausalLM to support dynamic subnetwork configuration
    across four sizes (s, m, l, xl). It provides a basic implementation for
    subnetwork switching without gradient freezing or covariance regularization.
    
    Attributes:
        config: The model configuration object.
    """
    
    def __init__(self, config):
        """
        Initialize the modified Llama model with subnetwork support.
        
        Args:
            config: Model configuration object containing model dimensions and
                architecture parameters.
        """
        super().__init__(config)
        scale_factors = [1/8, 1/4, 1/2, 1]  # s, m, l, xl

        # Replace FFN in each layer with ModifiedFFN
        for layer_idx in range(config.num_hidden_layers):
            self.model.layers[layer_idx].mlp = ModifiedLlamaMLP(config, scale_factors)
        self.config = config

    def configure_subnetwork(self, flag):
        """
        Configure the subnetwork for all layers based on the specified flag.
        
        This method propagates the subnetwork configuration to all MLP layers
        in the model, ensuring consistent sizing across the entire network.
        
        Args:
            flag (str): Subnetwork size flag ('s', 'm', 'l', or 'xl').
        """
        for layer_idx in range(self.config.num_hidden_layers):
            self.model.layers[layer_idx].mlp.configure_subnetwork(flag)
