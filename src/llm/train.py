"""
Training script for MatFormer models with configurable subnetworks.

This script provides a training pipeline for MatFormer-style models that support
dynamic subnetwork configuration (s, m, l, xl sizes) with optional covariance
regularization for ordered training.
"""

import os
import logging
import torch
from tqdm import tqdm
from llm.dataset_utils import FixedDataset
from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def setup_device(device_str):
    """
    Set up the training device based on availability and user preference.
    
    Args:
        device_str (str): Device string from command line arguments.
        
    Returns:
        torch.device: Configured device for training.
    """
    # Set up device (ordered training has device argument)
    if torch.cuda.is_available() and device_str.startswith('cuda'):
        device = torch.device(device_str)
        logger.info(f"Using device: {device}")
    elif device_str == 'cpu':
        device = torch.device('cpu')
        logger.info(f"Using device: {device}")
    else:
        logger.info(f"Warning: {device_str} not available, falling back to CPU")
        device = torch.device('cpu')
    return device




def base_collate_fn(batch):
    """
    Collate function for batching data without random flags.
    
    Args:
        batch (list): List of samples from the dataset.
        
    Returns:
        dict: Dictionary containing batched input_ids, attention_mask, and labels.
    """
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    labels = torch.stack([item['labels'] for item in batch])

    # Flag will be set externally during training loop
    return {
        "input_ids": input_ids, 
        "attention_mask": attention_mask, 
        "labels": labels
    }

def setup_dataloaders(dataset_dir, seed, tokenizer, max_length, batch_size, eval_samples=None):
    """
    Set up training and evaluation dataloaders from parquet files.
    
    Args:
        dataset_dir (str): Directory containing train.parquet and eval.parquet files.
        seed (int): Random seed for reproducibility.
        tokenizer: Tokenizer instance for processing text.
        max_length (int): Maximum sequence length for tokenization.
        batch_size (int): Batch size for dataloaders.
        eval_samples (int, optional): Number of evaluation samples to use. If None, uses all data.
        
    Returns:
        tuple: (eval_dataloader, train_dataloader) - Evaluation and training dataloaders.
    """
    train_dataset_path = os.path.join(dataset_dir, "train.parquet")
    eval_dataset_path = os.path.join(dataset_dir, "eval.parquet")
    train_dataset = FixedDataset(train_dataset_path, tokenizer, max_length)
    eval_dataset = FixedDataset(eval_dataset_path, tokenizer, max_length)

    # Subset evaluation dataset if eval_samples is specified
    if eval_samples is not None:
        if eval_samples < len(eval_dataset):
            # Use a subset of the evaluation data
            eval_dataset = torch.utils.data.Subset(eval_dataset, range(eval_samples))
            logger.info(f"Using subset of evaluation data: {eval_samples} samples")
        else:
            logger.info(f"Requested {eval_samples} eval samples but dataset has {len(eval_dataset)}, using all")

    eval_dataloader = DataLoader(
        eval_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        collate_fn=base_collate_fn,
        num_workers=4
    )
    train_dataloader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=False,  # Keep same order for all subnetworks
        collate_fn=base_collate_fn,
        generator=torch.Generator().manual_seed(seed),
        num_workers=4
    )
    return eval_dataloader, train_dataloader

def evaluate_model(model, eval_dataloader, flags, accelerator=None):
    """
    Evaluate the model on the evaluation dataset for each subnetwork size.
    
    Args:
        model: The model to evaluate.
        eval_dataloader: DataLoader for evaluation data.
        flags (list): List of subnetwork flags to evaluate (e.g., ['s', 'm', 'l', 'xl']).
        accelerator: Accelerator instance for handling device placement and mixed precision.
        
    Returns:
        dict: Dictionary mapping each flag to its losses {'flag': {'main_loss': float, 'covariance_loss': float, 'total_loss': float}}.
    """
    model.eval()
    eval_losses = {flag: {'main_loss': 0.0, 'covariance_loss': 0.0, 'total_loss': 0.0} for flag in flags}
    num_batches = len(eval_dataloader)

    with torch.no_grad():
        for flag in flags:
            total_main_loss = 0.0
            total_covariance_loss = 0.0
            total_combined_loss = 0.0
            
            for batch in tqdm(eval_dataloader, dynamic_ncols=True):
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]

                # Configure the subnetwork for the flag
                model.configure_subnetwork(flag)

                # Forward pass - Accelerate handles mixed precision automatically
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                
                main_loss = outputs.loss
                
                # Get covariance loss if model has the method
                if hasattr(model, 'get_covariance_loss'):
                    covariance_loss = model.get_covariance_loss()
                    if covariance_loss is not None:
                        combined_loss = main_loss + covariance_loss
                    else:
                        combined_loss = main_loss
                        device = accelerator.device if accelerator else model.device
                        covariance_loss = torch.tensor(0.0, device=device)
                else:
                    combined_loss = main_loss
                    device = accelerator.device if accelerator else model.device
                    covariance_loss = torch.tensor(0.0, device=device)
                
                total_main_loss += main_loss.item()
                total_covariance_loss += covariance_loss.item()
                total_combined_loss += combined_loss.item()

            eval_losses[flag]['main_loss'] = total_main_loss / num_batches
            eval_losses[flag]['covariance_loss'] = total_covariance_loss / num_batches
            eval_losses[flag]['total_loss'] = total_combined_loss / num_batches

    model.train()
    return eval_losses

def save_checkpoint(model, optimizer, scheduler, epoch, step, subnetwork, output_dir):
    """
    Save model checkpoint including optimizer and scheduler states.
    
    Args:
        model: The model to save.
        optimizer: Optimizer instance.
        scheduler: Learning rate scheduler instance.
        epoch (int): Current epoch number.
        step (int): Current step number.
        subnetwork (str): Current subnetwork size being trained.
        output_dir (str): Directory to save the checkpoint.
    """
    checkpoint_dir = os.path.join(output_dir, "checkpoints")
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    checkpoint_path = os.path.join(checkpoint_dir, f"checkpoint_epoch_{epoch}_step_{step}_subnetwork_{subnetwork}.pt")
    
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'epoch': epoch,
        'step': step,
        'subnetwork': subnetwork
    }
    
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved at: {checkpoint_path}")

def save_final_model(model, tokenizer, output_dir):
    """
    Save the final trained model and tokenizer.
    
    Args:
        model: The trained model to save.
        tokenizer: Tokenizer instance to save alongside the model.
        output_dir (str): Directory to save the final model.
    """
    final_model_dir = os.path.join(output_dir, "final_model")
    os.makedirs(final_model_dir, exist_ok=True)
    
    model.save_pretrained(final_model_dir)
    tokenizer.save_pretrained(final_model_dir)
    print(f"Final model saved at: {final_model_dir}")