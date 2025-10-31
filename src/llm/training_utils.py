"""
Training setup utilities for MatFormer training.
"""

import os
import torch
import random
import numpy as np
from transformers import AutoTokenizer, AutoConfig, get_scheduler
from torch.utils.data import DataLoader


def setup_device(device_arg="cuda"):
    """
    Set up the training device.
    
    Args:
        device_arg (str): Device argument from command line
    
    Returns:
        torch.device: Configured device
    """
    if torch.cuda.is_available() and device_arg.startswith('cuda'):
        device = torch.device(device_arg)
        print(f"Using device: {device}")
    elif device_arg == 'cpu':
        device = torch.device('cpu')
        print(f"Using device: {device}")
    else:
        print(f"Warning: {device_arg} not available, falling back to CPU")
        device = torch.device('cpu')
    
    return device


def setup_seeds(seed=42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed (int): Random seed value
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    print(f"Random seeds set to {seed}")


def setup_tokenizer(model_name="NousResearch/Llama-3.2-1B"):
    """
    Set up the tokenizer.
    
    Args:
        model_name (str): Name of the model to load tokenizer from
    
    Returns:
        AutoTokenizer: Configured tokenizer
    """
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def setup_model(model_class, model_name="NousResearch/Llama-3.2-1B", max_length=512, device=None):
    """
    Set up the model.
    
    Args:
        model_class: The model class to instantiate
        model_name (str): Name of the model to load config from
        max_length (int): Maximum sequence length
        device (torch.device): Device to move model to
    
    Returns:
        model: Configured model
    """
    # TODO : Prints -> logging throughout this file/project
    print("Loading config...")
    config = AutoConfig.from_pretrained(model_name)
    
    # Update config with custom max sequence length
    config.max_position_embeddings = max_length
    print(f"Set max_position_embeddings to {max_length}")
    
    print("Initializing model. This may take a while... ", end="", flush=True)
    model = model_class(config)
    if device is not None:
        model = model.to(device)
    print("Done!")
    
    return model


def setup_training_components(model, learning_rate=1e-4, total_steps=None, warmup_ratio=0.1, disable_amp=False):
    """
    Set up training components (optimizer, scheduler, scaler).
    
    Args:
        model: The model to train
        learning_rate (float): Learning rate for optimizer
        total_steps (int): Total number of training steps
        warmup_ratio (float): Ratio of warmup steps
        disable_amp (bool): Whether to disable automatic mixed precision
    
    Returns:
        tuple: (optimizer, scheduler, scaler)
    """
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    
    scheduler = None
    if total_steps is not None:
        num_warmup_steps = int(warmup_ratio * total_steps)
        scheduler = get_scheduler(
            "cosine",
            optimizer=optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=total_steps,
        )
    
    # Initialize gradient scaler for mixed precision training (only if AMP is enabled)
    scaler = torch.cuda.amp.GradScaler() if not disable_amp else None
    
    print(f"Mixed precision training: {'Disabled' if disable_amp else 'Enabled (AMP)'}")
    
    return optimizer, scheduler, scaler


def setup_dataloaders(train_dataset, eval_dataset, batch_size=8, collate_fn=None, shuffle_train=True, seed=42):
    """
    Set up training and evaluation dataloaders.
    
    Args:
        train_dataset: Training dataset
        eval_dataset: Evaluation dataset
        batch_size (int): Batch size for dataloaders
        collate_fn: Collate function for dataloaders
        shuffle_train (bool): Whether to shuffle training data
        seed (int): Random seed for generator
    
    Returns:
        tuple: (train_dataloader, eval_dataloader)
    """
    train_dataloader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=shuffle_train,
        collate_fn=collate_fn,
        generator=torch.Generator().manual_seed(seed)
    )
    
    eval_dataloader = DataLoader(
        eval_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    return train_dataloader, eval_dataloader
    



def print_training_info(total_steps=None, steps_per_epoch=None, num_epochs=None, disable_amp=False):
    """
    Print training information.
    
    Args:
        total_steps (int, optional): Total training steps
        steps_per_epoch (int, optional): Steps per epoch
        num_epochs (int, optional): Number of epochs
        disable_amp (bool): Whether AMP is disabled
    """
    print(f"Mixed precision training: {'Disabled' if disable_amp else 'Enabled (AMP)'}")
    if total_steps is not None:
        print(f"Total training steps: {total_steps}")
    if steps_per_epoch is not None:
        print(f"Steps per epoch: {steps_per_epoch}")
    if num_epochs is not None:
        print(f"Number of epochs: {num_epochs}")
