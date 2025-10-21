"""
Command line argument parser for MatFormer training scripts.

This module provides argument parsing functionality for configuring
MatFormer model training with various hyperparameters and options.
It supports different model types, training configurations, and
optimization parameters.

Functions:
    setup_parser: Create and configure argument parser for training scripts.
"""

import argparse


def setup_parser():
    """
    Create and configure argument parser for MatFormer training scripts.
    
    This function sets up an ArgumentParser with comprehensive options for
    configuring MatFormer model training. It supports different model types,
    training hyperparameters, and optimization settings.
    
    Returns:
        argparse.ArgumentParser: Configured argument parser with all training options.
        
    Supported Model Types:
        - matformer: Base MatFormer model without covariance regularization
        - frozen_matformer: MatFormer with gradient freezing for subnetwork training
        - frozen_cov_matformer: MatFormer with both gradient freezing and covariance regularization
    
    Training Configuration:
        - Subnetwork training across sizes: s, m, l, xl
        - Configurable epochs per subnetwork
        - Learning rate scheduling
        - Mixed precision training support
    
    Examples:
        >>> parser = setup_parser()
        >>> args = parser.parse_args([
        ...     "--model_type", "frozen_matformer",
        ...     "--dataset_dir", "./dataset",
        ...     "--output_dir", "./models/test",
        ...     "--batch_size", "16",
        ...     "--learning_rate", "5e-5"
        ... ])
        >>> print(args.model_type)  # "frozen_matformer"
    """
    parser = argparse.ArgumentParser(description="Train a MatFormer style model with various options")
    
    parser.add_argument(
        "--model_type",
        type=str,
        default="matformer",
        help="Model type to use for training, should be one of 'matformer', 'frozen_matformer', 'frozen_cov_matformer', 'weight_based_matformer'"
    )

    parser.add_argument(
        "--config_name",
        type=str,
        default="NousResearch/Llama-3.2-1B",
        help="Config or model name/path to use as base model (default: NousResearch/Llama-3.2-1B)"
    )

    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for training (default: cuda)"
    )
    
    parser.add_argument(
        "--num_epochs_per_subnetwork",
        type=float,
        default=1.0,
        help="Number of epochs to train each subnetwork. If < 1.0, distributes data across subnetworks within single epoch (default: 1.0)"
    )
    
    parser.add_argument(
        "--covariance_loss_weight",
        type=float,
        default=1,
        help="Weight for covariance loss term (if used) (default: 1)"
    )
    
    parser.add_argument(
        "--dataset_dir",
        type=str,
        required=True,
        help="Directory containing train.parquet and eval.parquet files"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="Directory to save model checkpoints and final model"
    )
    
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Training batch size (default: 32)"
    )
    
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=1e-4,
        help="Base learning rate (default: 1e-4)"
    )
    
    parser.add_argument(
        "--max_length",
        type=int,
        default=512,
        help="Maximum sequence length (default: 512)"
    )
    
    parser.add_argument(
        "--checkpoint_epochs",
        type=int,
        default=1,
        help="Save checkpoint every N epochs (default: 1)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)"
    )
    
    parser.add_argument(
        "--eval_samples",
        type=int,
        default=None,
        help="Number of evaluation samples to use (subset). If not specified, uses all evaluation data."
    )
    
    parser.add_argument(
        "--random_subnetwork_order",
        action="store_true",
        help="Use random subnetwork selection for each batch instead of ordered training (s->m->l->xl)"
    )

    return parser
