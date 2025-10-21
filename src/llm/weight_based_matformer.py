"""
Weight-based MatFormer implementation with gradient freezing and weight covariance regularization.

This module provides a MatFormer implementation that supports dynamic
subnetwork configuration with gradient freezing for larger network sections
and weight-based covariance regularization using cosine similarity for ordered training.
It extends the standard Llama architecture with configurable subnetworks across four sizes
(s, m, l, xl) while maintaining gradient flow only within the active slice.

Classes:
    ModifiedLlamaMLP: Extended Llama MLP with subnetwork and gradient control.
    ModifiedLlamaForCausalLM: Modified Llama model with frozen subnetwork training.
"""

import torch
import torch.nn.functional as F
from transformers import LlamaForCausalLM
from transformers.models.llama.modeling_llama import LlamaMLP
import logging

logger = logging.getLogger(__name__)


class ModifiedLlamaMLP(LlamaMLP):
    """
    Modified Llama MLP with subnetwork configuration and gradient freezing.
    
    This class extends the standard LlamaMLP to support dynamic subnetwork
    configuration across four sizes (s, m, l, xl) with gradient freezing
    for larger network sections. It also computes weight-based covariance values for
    regularization during training using cosine similarity.
    
    Attributes:
        intermediate_size (int): The full intermediate dimension size.
        scale_factors (list): Scale factors for subnetwork sizes [s, m, l, xl].
        current_subset_hd (int): Current hidden dimension size for subnetwork.
        current_flag (str): Current subnetwork flag ('s', 'm', 'l', or 'xl').
        current_slice (tuple): Tuple indicating (start, end) indices for current slice.
        current_covariance_value (torch.Tensor or None): Computed covariance value.
        flag_hierarchy (list): Ordered list of subnetwork flags.
        flag_sizes (dict): Mapping of flags to their corresponding sizes.
    """
    
    def __init__(self, config, scale_factors, include_gate_proj_loss=True, include_up_proj_loss=True):
        """
        Initialize the modified MLP with subnetwork and gradient control.
        
        Args:
            config: Model configuration object containing intermediate_size.
            scale_factors (list): List of scale factors for subnetwork sizes [s, m, l, xl].
                Each factor determines the proportion of intermediate_size to use.
            include_gate_proj_loss (bool): Whether to include gate_proj layer in weight covariance loss.
            include_up_proj_loss (bool): Whether to include up_proj layer in weight covariance loss.
        """
        super().__init__(config)
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.scale_factors = scale_factors  # List of scale_factors for 's', 'm', 'l', 'xl'
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
        self.include_gate_proj_loss = include_gate_proj_loss
        self.include_up_proj_loss = include_up_proj_loss

    def configure_subnetwork(self, flag):
        """
        Configure subnetwork size based on the specified flag.
        
        This method sets up the current subnetwork configuration by determining
        the appropriate slice of weights to use based on the flag. It also
        calculates the slice boundaries for gradient freezing.
        
        Args:
            flag (str): Subnetwork size flag. Must be one of 's', 'm', 'l', or 'xl'.
            
        Raises:
            ValueError: If flag is not one of the valid subnetwork flags.
        """
        if flag not in self.flag_hierarchy:
            raise ValueError(f"Invalid flag '{flag}'. Must be one of: {self.flag_hierarchy}")
            
        self.current_subset_hd = self.flag_sizes[flag]
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
        slice receive gradient updates by setting gradients to None for all
        weights outside the active slice.
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
        Remove gradient tracking through network sections beyond current slice.
        
        This method is deprecated in favor of zero_non_slice_gradients() which
        provides more precise gradient control.
        
        Args:
            flag (str): Current subnetwork flag.
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
        the current subnetwork slice and computes covariance values for
        regularization during training.
        
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
            self.current_covariance_value = self._compute_covariance_value()
        
        return down_proj_output
    
    def _compute_covariance_value(self):
        """
        Compute weight-based covariance value using cosine similarity between gate and up projection weights.
        
        This method computes the cosine similarity between the weight vectors of
        different output activations within the gate_proj and up_proj layers
        for the current slice. It calculates pairwise similarities between each
        pair of output weights and returns the mean as a regularization term
        for ordered training. The inclusion of each layer's loss is configurable.
        
        Returns:
            torch.Tensor: Scalar cosine similarity value (mean of all pairwise similarities).
            
        TODO: Verify the mathematical correctness of the cosine similarity calculation
        and ensure the implementation correctly computes the average pairwise
        similarity for both gate_proj and up_proj layers.
        """
        total_similarity = 0.0
        total_count = 0
        
        # Compute cosine similarity for gate_proj layer if enabled
        if self.include_gate_proj_loss:
            # Get the weight matrix for the current slice of gate_proj
            # Shape: (current_subset_hd, hidden_size) - each row is an output weight vector
            gate_weights = self.gate_proj.weight[:self.current_subset_hd]
            
            # Normalize each weight vector (each row)
            # This gives us a matrix where each row is a unit vector
            gate_norm = F.normalize(gate_weights, p=2, dim=1)
            
            # Compute cosine similarity matrix using matrix multiplication
            # cos_sim_matrix[i,j] = dot(gate_norm[i], gate_norm[j])
            # Shape: (current_subset_hd, current_subset_hd)
            cos_sim_matrix = torch.matmul(gate_norm, gate_norm.t())
            
            # Create a mask to exclude self-similarities (diagonal elements)
            # This ensures we only compute similarities between different outputs
            mask = torch.ones_like(cos_sim_matrix)
            mask.fill_diagonal_(0)
            
            # Compute mean of all pairwise similarities for gate_proj
            gate_mean_sim = torch.mean(cos_sim_matrix * mask)
            total_similarity += gate_mean_sim
            total_count += 1
        
        # Compute cosine similarity for up_proj layer if enabled
        if self.include_up_proj_loss:
            # Get the weight matrix for the current slice of up_proj
            # Shape: (current_subset_hd, hidden_size) - each row is an output weight vector
            up_weights = self.up_proj.weight[:self.current_subset_hd]
            
            # Normalize each weight vector (each row)
            # This gives us a matrix where each row is a unit vector
            up_norm = F.normalize(up_weights, p=2, dim=1)
            
            # Compute cosine similarity matrix using matrix multiplication
            # cos_sim_matrix[i,j] = dot(up_norm[i], up_norm[j])
            # Shape: (current_subset_hd, current_subset_hd)
            cos_sim_matrix = torch.matmul(up_norm, up_norm.t())
            
            # Create a mask to exclude self-similarities (diagonal elements)
            # This ensures we only compute similarities between different outputs
            mask = torch.ones_like(cos_sim_matrix)
            mask.fill_diagonal_(0)
            
            # Compute mean of all pairwise similarities for up_proj
            up_mean_sim = torch.mean(cos_sim_matrix * mask)
            total_similarity += up_mean_sim
            total_count += 1
        
        # Return the average of all enabled components
        # If neither is enabled, return 0 (no loss contribution)
        return total_similarity / max(total_count, 1)


class ModifiedLlamaForCausalLM(LlamaForCausalLM):
    """
    Modified Llama model with frozen subnetwork training.
    
    This class extends LlamaForCausalLM to support dynamic subnetwork
    configuration with gradient freezing for larger network sections.
    It provides methods for subnetwork switching, gradient management,
    and weight-based covariance regularization using cosine similarity.
    
    Attributes:
        config: The model configuration object.
    """
    
    def __init__(self, config):
        """
        Initialize the modified Llama model with frozen subnetwork training.
        
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
        Configure the subnetwork for all layers based on the flag.
        
        This method propagates the subnetwork configuration to all MLP layers
        in the model and logs the configuration change.
        
        Args:
            flag (str): Subnetwork size flag ('s', 'm', 'l', or 'xl').
        """
        logger.info(f"Configuring network to size : {flag}")
        for layer_idx in range(len(self.model.layers)):
            self.model.layers[layer_idx].mlp.configure_subnetwork(flag)
    
    def unfreeze_all_weights(self):
        """
        Unfreeze all weights in the model.
        
        This method enables gradient computation for all model parameters,
        effectively removing any gradient freezing that was applied.
        """
        for param in self.parameters():
            param.requires_grad = True
    
    def zero_non_slice_gradients(self):
        """
        Zero out gradients for all non-slice weights across all layers.
        
        This method ensures that only weights within the current subnetwork
        slices receive gradient updates by calling zero_non_slice_gradients
        on each MLP layer.
        """
        logger.info("Removing all non-slice gradients")
        for layer_idx in range(len(self.model.layers)):
            self.model.layers[layer_idx].mlp.zero_non_slice_gradients() 

    def get_covariance_loss(self):
        """
        Aggregate covariance values from all MLP layers.
        
        This method collects covariance values computed by each MLP layer
        during training and returns the total as a regularization term.
        
        Returns:
            torch.Tensor or None: Total covariance value across all layers,
                or None if no values were computed (e.g., not in training mode).
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
        
        This method analyzes the model's parameters to determine how many
        are frozen (not requiring gradients) versus trainable.
        
        Returns:
            dict: Dictionary containing:
                - total_parameters (int): Total number of parameters.
                - frozen_parameters (int): Number of frozen parameters.
                - trainable_parameters (int): Number of trainable parameters.
                - frozen_percentage (float): Percentage of parameters that are frozen.
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
