"""
Frozen MatFormer implementation with configurable subnetworks.

This module implements a modified Llama model that supports dynamic subnetwork
configuration across four sizes (s, m, l, xl) with gradient freezing for larger
network sections during training. The implementation includes covariance-based
regularization for ordered training and supports efficient parameter sharing
across subnetwork sizes.

Classes:
    ModifiedLlamaMLP: Extended Llama MLP with subnetwork configuration support.
    ModifiedLlamaForCausalLM: Modified Llama model with subnetwork management.
"""

import torch
import torch.nn.functional as F
from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaMLP
import torch.nn as nn
import logging

logger = logging.getLogger(__name__)


class ModifiedLlamaMLP(LlamaMLP):
    """
    Modified Llama MLP with configurable subnetwork sizes and covariance tracking.
    
    This class extends the standard LlamaMLP to support dynamic subnetwork
    configuration across four sizes (s, m, l, xl) with gradient management
    and covariance computation capabilities. It implements ordered training
    by computing covariance regularization on intermediate representations.
    
    Attributes:
        scale_factors (list): Scale factors for subnetwork sizes [s, m, l, xl].
        current_subset_hd (int): Current hidden dimension size for subnetwork.
        current_flag (str): Current subnetwork size flag ('s', 'm', 'l', or 'xl').
        current_slice (tuple): Start and end indices for current subnetwork slice.
        current_covariance_value (torch.Tensor): Computed covariance value for regularization.
        flag_hierarchy (list): Ordered list of subnetwork flags.
        flag_sizes (dict): Mapping of flags to their corresponding sizes.
    """
    
    def __init__(self, config, scale_factors):
        """
        Initialize the modified MLP with subnetwork scaling capabilities.
        
        Args:
            config: Model configuration object containing hidden_size and intermediate_size.
            scale_factors (list): List of scale factors for subnetwork sizes [s, m, l, xl].
                Each factor determines the proportion of intermediate_size to use.
        """
        super().__init__(config)
        self.hidden_size=config.hidden_size,
        self.intermediate_size = config.intermediate_size
        self.scale_factors = scale_factors  # List of scale factors for 's', 'm', 'l', 'xl'
        self.current_subset_hd = None
        self.current_flag = None
        self.current_slice = None
        self.current_covariance_value = None  # Store covariance value for this layer
        self.flag_hierarchy = ['s', 'm', 'l', 'xl']
        self.flag_sizes = {
            's': int(self.intermediate_size * self.scale_factors[0]),  # hd/8
            'm': int(self.intermediate_size * self.scale_factors[1]),  # hd/4
            'l': int(self.intermediate_size * self.scale_factors[2]),  # hd/2
            'xl': int(self.intermediate_size * self.scale_factors[3])  # hd
        }

    def configure_subnetwork(self, flag):
        """
        Configure subnetwork size based on the specified flag.
        
        This method sets up the current subnetwork configuration by determining
        the appropriate slice of weights to use based on the flag. It also
        calculates the slice boundaries for gradient management.
        
        Args:
            flag (str): Subnetwork size flag. Must be one of 's', 'm', 'l', or 'xl'.
            
        Raises:
            ValueError: If flag is not one of the valid subnetwork flags.
        """
        if flag == 's':
            scale = self.scale_factors[0]  # hd/8
        elif flag == 'm':
            scale = self.scale_factors[1]  # hd/4
        elif flag == 'l':
            scale = self.scale_factors[2]  # hd/2
        elif flag == 'xl':
            scale = self.scale_factors[3]  # hd
        else:
            raise ValueError(f"Invalid flag '{flag}'. Must be one of: {self.flag_hierarchy}")

        self.current_subset_hd = int(self.intermediate_size * scale)
        self.current_flag = flag
        
        current_size = self.flag_sizes[flag]
        if flag != "s":
            previous_size = self.flag_sizes[self.flag_hierarchy[self.flag_hierarchy.index(flag)-1]]
        else:
            previous_size = 0
        
        self.current_slice = (previous_size, current_size)

    def zero_non_slice_gradients(self):
        """
        Zero out gradients for weights outside the current subnetwork slice.
        
        This method ensures that only weights within the current subnetwork
        configuration receive gradient updates by setting gradients to None
        for weights outside the active slice.
        """
        start_idx = self.current_slice[0]
        end_idx = self.current_slice[1]

        # Zero out weights before our slice
        if hasattr(self.gate_proj.weight, 'requires_grad'):
            self.gate_proj.weight[:start_idx].grad = None
        if hasattr(self.up_proj.weight, 'requires_grad'):
            self.up_proj.weight[:start_idx].grad = None
        if hasattr(self.down_proj.weight, 'requires_grad'):
            self.down_proj.weight[:, :start_idx].grad = None
        
        # Zero out weights after our slice
        if hasattr(self.gate_proj.weight, 'requires_grad'):
            self.gate_proj.weight[end_idx:].grad = None
        if hasattr(self.up_proj.weight, 'requires_grad'):
            self.up_proj.weight[end_idx:].grad = None
        if hasattr(self.down_proj.weight, 'requires_grad'):
            self.down_proj.weight[:, end_idx:].grad = None

    def degradient_bigger_networks(self, flag):
        """
        Remove gradient tracking through network sections beyond the currently selected slice.
        
        This method detaches weights outside the current subnetwork to prevent
        gradient flow, effectively freezing larger network sections.
        
        Args:
            flag (str): Current subnetwork size flag.
        """
        current_size = self.flag_sizes[flag]

        # Freeze outside weights
        if hasattr(self.gate_proj.weight, 'requires_grad'):
            self.gate_proj.weight[current_size:].detach()
        if hasattr(self.up_proj.weight, 'requires_grad'):
            self.up_proj.weight[current_size:].detach()
        if hasattr(self.down_proj.weight, 'requires_grad'):
            self.down_proj.weight[:, current_size:].detach()

        # Ensure gradients are calculated for entire current network slice
        if hasattr(self.gate_proj.weight, 'requires_grad'):
            self.gate_proj.weight[:current_size].requires_grad = True
        if hasattr(self.up_proj.weight, 'requires_grad'):
            self.up_proj.weight[:current_size].requires_grad = True
        if hasattr(self.down_proj.weight, 'requires_grad'):
            self.down_proj.weight[:, :current_size].requires_grad = True
    
    def forward(self, x):
        """
        Forward pass through the modified MLP with subnetwork configuration.
        
        This method performs the forward pass using only the weights within
        the current subnetwork slice. It also computes covariance values
        for regularization during training.
        
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
        down_proj_output = F.linear(self.act_fn(F.linear(x, gate_proj) * F.linear(x, up_proj)), down_proj)
        
        # Compute covariance value if training
        self.current_covariance_value = None
        if self.training:
            self.current_covariance_value = self._compute_covariance_value(down_proj_output)
        
        return down_proj_output
    
    def _compute_covariance_value(self, output):
        """
        Compute normalized covariance value for the down_proj output.
        
        This method computes the covariance matrix of the output activations
        and returns the mean absolute value for use as a regularization term.
        
        Args:
            output (torch.Tensor): Tensor with shape (batch_size, seq_length, hidden_size) or (N, M).
            
        Returns:
            torch.Tensor: Scalar covariance value (mean of normalized covariance matrix).
        """
        # Reshape output to 2D: (batch_size * seq_length, hidden_size)
        if output.dim() == 3:
            batch_size, seq_length, hidden_size = output.shape
            output_2d = output.view(-1, hidden_size)  # Shape: (N, M) where N = batch_size * seq_length
        else:
            output_2d = output
        
        # Compute covariance matrix (features as rows, so transpose the output)
        # output_2d is (N, M), we want (M, N) for torch.cov
        cov_matrix = torch.clamp(torch.abs(torch.cov(output_2d.T)), 0, 1e+4)
        
        # Return mean of normalized covariance matrix
        return torch.mean(cov_matrix)


class ModifiedLlamaForCausalLM(LlamaForCausalLM):
    """
    Modified Llama model with configurable subnetworks and covariance regularization.
    
    This class extends LlamaForCausalLM to support dynamic subnetwork configuration
    across four sizes (s, m, l, xl) with optional covariance regularization for
    ordered training. It manages subnetwork switching and gradient freezing
    across all layers.
    
    Attributes:
        model: The underlying Llama model with modified MLP layers.
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

    def configure_subnetwork(self, flag):
        """
        Configure the subnetwork for all layers based on the specified flag.
        
        This method propagates the subnetwork configuration to all MLP layers
        in the model, ensuring consistent sizing across the entire network.
        
        Args:
            flag (str): Subnetwork size flag ('s', 'm', 'l', or 'xl').
        """
        logger.info(f"Configuring network to size : {flag}")
        for layer_idx in range(len(self.model.layers)):
            self.model.layers[layer_idx].mlp.configure_subnetwork(flag)
    
    def unfreeze_all_weights(self):
        """Unfreeze all weights in the model by setting requires_grad=True."""
        for param in self.parameters():
            param.requires_grad = True
    
    def zero_non_slice_gradients(self):
        """
        Zero gradients for all non-slice weights across all layers.
        
        This method ensures that only weights within the current subnetwork
        configuration receive gradient updates by calling the zeroing method
        on each MLP layer.
        """
        logger.info("Removing all non-slice gradients")
        for layer_idx in range(len(self.model.layers)):
            self.model.layers[layer_idx].mlp.zero_non_slice_gradients() 

    def get_covariance_loss(self):
        """
        Aggregate covariance values from all MLP layers for regularization.
        
        This method collects covariance values computed during the forward pass
        from all MLP layers and returns the total for use as a regularization term.
        
        Returns:
            torch.Tensor or None: Total covariance value across all layers,
            or None if no values computed or not in training mode.
        """
        if not self.training:
            return None
            
        total_covariance = 0.0
        layer_count = 0
        
        for layer_idx in range(len(self.model.layers)):
            layer_covariance = self.model.layers[layer_idx].mlp.current_covariance_value
            if layer_covariance is not None:
                total_covariance += layer_covariance
                layer_count += 1
        
        return total_covariance if layer_count > 0 else None

    def get_frozen_parameters_info(self):
        """
        Get information about frozen parameters in the model.
        
        This method provides statistics about parameter freezing, useful for
        monitoring the training process and understanding model capacity.
        
        Returns:
            dict: Dictionary containing parameter statistics:
                - total_parameters (int): Total number of parameters
                - frozen_parameters (int): Number of frozen parameters
                - trainable_parameters (int): Number of trainable parameters
                - frozen_percentage (float): Percentage of frozen parameters
        """
        total_params = 0
        frozen_params = 0
        
        for name, param in self.named_parameters():
            total_params += param.numel()
            if not param.requires_grad:
                frozen_params += param.numel()
        
        return {
            'total_parameters': total_params,
            'frozen_parameters': frozen_params,
            'trainable_parameters': total_params - frozen_params,
            'frozen_percentage': (frozen_params / total_params) * 100 if total_params > 0 else 0
        }
