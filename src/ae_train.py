#!/usr/bin/env python3

"""
Train Embeddageddon - Autoencoder for Multi-Model Token Embeddings

This script implements an autoencoder that learns to compress and reconstruct
token embeddings from multiple language models into a unified representation.

The autoencoder uses a bottleneck layer sized at 1.5x the largest embedding
dimension to create a dense feature space containing information from all models.
"""

import json
import random
from pathlib import Path
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import argparse
from tqdm import tqdm
from datetime import datetime

from utils.loss_tracker import create_loss_tracker
from autoencoder.dataset import MultiModelEmbeddingDataset
from autoencoder.model import EmbeddingAutoencoder, masked_mse_loss
from utils.embeddings import load_embedding_dicts
from utils.tokens import get_common_tokens

def create_timestamped_output_dir(base_name: str = "train_run") -> Path:
    """
    Create a timestamped output directory for organizing training outputs.
    
    Args:
        base_name: Base name for the directory
        
    Returns:
        Path to the created timestamped directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(f"training_runs/{base_name}_{timestamp}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create subdirectories for different types of outputs
    (output_dir / "models").mkdir(exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    (output_dir / "logs").mkdir(exist_ok=True)
    (output_dir / "config").mkdir(exist_ok=True)
    
    return output_dir


def save_training_config(output_dir: Path, args: argparse.Namespace, dataset: MultiModelEmbeddingDataset):
    """
    Save training configuration and dataset info to the output directory.
    
    Args:
        output_dir: Output directory path
        args: Command line arguments
        dataset: Training dataset
    """
    config = {
        "training_type": "mat_train_with_random_subnetwork",
        "training_args": {
            "embedding_dir": args.embedding_dir,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "device": args.device,
            "use_amp": args.use_amp,
            "precision": args.precision
        },
        "dataset_info": {
            "total_tokens": len(dataset.common_tokens),
            "model_names": dataset.model_names,
            "model_dims": dataset.model_dims,
            "total_input_dim": dataset.total_dim,
            "bottleneck_dim": dataset.bottleneck_dim
        },
        "subnetwork_info": {
            "scale_factors": [1/8, 1/4, 1/2, 1.0],
            "subnetwork_sizes": ["s", "m", "l", "xl"],
            "description": "Random subnetwork selection per batch"
        },
        "timestamp": datetime.now().isoformat()
    }
    
    config_path = output_dir / "config" / "training_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"Training configuration saved to: {config_path}")


def train_autoencoder(dataset: MultiModelEmbeddingDataset,
                     output_dir: Path,
                     num_epochs: int = 100,
                     batch_size: int = 32,
                     learning_rate: float = 0.001,
                     device: str = "cuda" if torch.cuda.is_available() else "cpu",
                     use_amp: bool = True,
                     precision: str = "fp16",
                     num_workers: int = 2) -> EmbeddingAutoencoder:
    """
    Train the embedding autoencoder.

    Args:
        dataset: Training dataset
        output_dir: Directory to save all outputs
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Learning rate for optimizer
        device: Device to train on
        use_amp: Whether to use Automatic Mixed Precision
        precision: Precision mode for mixed precision training (fp16 or fp8)
        num_workers: Number of worker processes for data loading (0 = main process only)

    Returns:
        Trained autoencoder model
    """
    # Create data loader with async loading for better GPU utilization
    # Note: num_workers creates separate processes, each with a copy of the dataset
    # Reduce num_workers if you experience OOM errors
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,  # Parallel data loading
        pin_memory=False,         # Disabled: causes memory leak with multiprocessing
        persistent_workers=False  # Disabled: allows worker memory cleanup between epochs
    )

    # Initialize model
    model = EmbeddingAutoencoder(
        input_dim=dataset.total_dim,
        bottleneck_dim=dataset.bottleneck_dim
    )
    model.to(device)

    # Initialize optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Initialize AMP components if using mixed precision
    scaler = None
    autocast_dtype = torch.bfloat16  # Default to FP16

    if use_amp and device == "cuda":
        if precision == "fp8":
            try:
                if hasattr(torch, 'float8_e4m3fn'):
                    autocast_dtype = torch.float8_e4m3fn
                    print("Using Automatic Mixed Precision with FP8 (E4M3)")
                else:
                    print("Warning: FP8 not supported on this PyTorch version. Falling back to BF16.")
                    autocast_dtype = torch.bfloat16
                    precision = "fp16"
            except Exception as e:
                print(f"Warning: FP8 not available ({e}). Falling back to BF16.")
                autocast_dtype = torch.bfloat16
                precision = "fp16"
        else:
            autocast_dtype = torch.bfloat16

        scaler = torch.amp.GradScaler("cuda")
        print(f"Using Automatic Mixed Precision with {precision.upper()}")
    elif use_amp and device != "cuda":
        print("Warning: AMP is only supported on CUDA devices. Falling back to FP32.")
        use_amp = False

    print(f"Training autoencoder on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training samples: {len(dataset)}")
    print(f"Mixed Precision: {'Enabled' if use_amp else 'Disabled'}")
    print(f"Random Subnetwork Training: Enabled (s, m, l, xl)")
    print(f"Output directory: {output_dir}")
    print("-" * 60)

    # Initialize loss tracker with timestamped output directory
    loss_tracker = create_loss_tracker(
        output_dir=str(output_dir / "logs"),
        filename="training_loss.csv"
    )
    print(f"Loss tracking CSV: {loss_tracker.get_csv_path()}")
    print("-" * 60)

    # Training loop
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        num_batches = 0

        # Start epoch tracking
        loss_tracker.start_epoch(epoch + 1)

        progress_bar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{num_epochs}", dynamic_ncols=True)

        for batch_idx, (batch_inputs, batch_targets, batch_masks) in enumerate(progress_bar):
            batch_inputs = batch_inputs.to(device)
            batch_targets = batch_targets.to(device)
            batch_masks = batch_masks.to(device)

            # ---- Random subnetwork selection per batch ----
            flag = random.choice(['s', 'm', 'l', 'xl'])
            #flag = "xl"
            model.configure_subnetwork(flag)
            # Print removed to avoid GPU sync overhead
            # ----------------------------------------------

            optimizer.zero_grad()

            if use_amp and scaler is not None:
                # Mixed precision forward pass
                with torch.amp.autocast("cuda", autocast_dtype):
                    outputs = model(batch_inputs)
                    loss = masked_mse_loss(outputs, batch_targets, batch_masks)

                # Mixed precision backward pass
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard precision forward pass
                outputs = model(batch_inputs)
                loss = masked_mse_loss(outputs, batch_targets, batch_masks)

                # Standard precision backward pass
                loss.backward()
                optimizer.step()

            # Cache loss.item() to avoid multiple GPU syncs
            batch_loss = loss.item()
            total_loss += batch_loss
            num_batches += 1

            # Log batch loss to CSV
            loss_tracker.log_batch(
                epoch=epoch + 1,
                batch=batch_idx + 1,
                batch_loss=batch_loss,
                learning_rate=learning_rate,
                batch_size=batch_size
            )

            # Update progress bar
            progress_bar.set_postfix({"Loss": f"{batch_loss:.6f}", "Subnet": flag})

        # End epoch tracking
        avg_loss = loss_tracker.end_epoch(epoch + 1)
        print(f"Epoch {epoch+1}/{num_epochs}, Average Loss: {avg_loss:.6f}")

        # Save model checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = output_dir / "checkpoints" / f"embeddageddon_model_epoch_{epoch+1}.pth"
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'model_config': {
                    'input_dim': dataset.total_dim,
                    'bottleneck_dim': dataset.bottleneck_dim
                }
            }, checkpoint_path)
            print(f"Saved checkpoint: {checkpoint_path}")

    # Save final model
    final_model_path = output_dir / "models" / "embeddageddon_model_final.pth"
    torch.save({
        'model_state_dict': model.state_dict(),
        'model_config': {
            'input_dim': dataset.total_dim,
            'bottleneck_dim': dataset.bottleneck_dim
        },
        'model_names': dataset.model_names,
        'model_dims': dataset.model_dims
    }, final_model_path)
    print(f"Saved final model: {final_model_path}")

    return model


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description="Train Embeddageddon Autoencoder")
    parser.add_argument("--embedding_dir", type=str, default="embedding_dicts",
                       help="Directory containing embedding pickle files")
    parser.add_argument("--epochs", type=int, default=100,
                       help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                       help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                       help="Learning rate for optimizer")
    parser.add_argument("--device", type=str, default="auto",
                       help="Device to train on (auto, cpu, cuda)")
    parser.add_argument("--use_amp", action="store_true", default=True,
                       help="Use Automatic Mixed Precision (default: True)")
    parser.add_argument("--no_amp", dest="use_amp", action="store_false",
                       help="Disable Automatic Mixed Precision")
    parser.add_argument("--precision", type=str, default="fp16", choices=["fp16", "fp8"],
                       help="Precision mode for mixed precision training (fp16 or fp8)")
    parser.add_argument("--num_workers", type=int, default=2,
                       help="Number of worker processes for data loading (default: 2, use 0 to disable)")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Custom output directory (if not provided, creates timestamped directory)")
    
    args = parser.parse_args()
    
    # Set device
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device
    
    # Create timestamped output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        # Create subdirectories
        (output_dir / "models").mkdir(exist_ok=True)
        (output_dir / "checkpoints").mkdir(exist_ok=True)
        (output_dir / "logs").mkdir(exist_ok=True)
        (output_dir / "config").mkdir(exist_ok=True)
    else:
        output_dir = create_timestamped_output_dir("train_run")
    
    print("=" * 60)
    print("EMBEDDAGEDDON TRAINING")
    print("=" * 60)
    print(f"Embedding directory: {args.embedding_dir}")
    print(f"Training device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Output directory: {output_dir}")
    print("-" * 60)
    
    try:
        # Load embedding dictionaries
        print("Loading embedding dictionaries...")
        embedding_dicts = load_embedding_dicts(args.embedding_dir)
        
        # Get common tokens
        print("Finding common tokens...")
        common_tokens = get_common_tokens(embedding_dicts)
        print(f"Found {len(common_tokens)} unique tokens")
        
        # Get model names in consistent order
        model_names = sorted(embedding_dicts.keys())
        print(f"Models: {model_names}")
        
        # Create dataset
        print("Creating dataset...")
        dataset = MultiModelEmbeddingDataset(embedding_dicts, common_tokens, model_names)
        
        # Save training configuration
        save_training_config(output_dir, args, dataset)
        
        # Train model
        print("Starting training...")
        model = train_autoencoder(
            dataset=dataset,
            output_dir=output_dir,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.learning_rate,
            device=device,
            use_amp=args.use_amp,
            precision=args.precision,
            num_workers=args.num_workers
        )
        
        print("=" * 60)
        print("TRAINING COMPLETED SUCCESSFULLY!")
        print(f"All outputs saved to: {output_dir}")
        print(f"Final model: {output_dir / 'models' / 'embeddageddon_model_final.pth'}")
        print(f"Training logs: {output_dir / 'logs' / 'training_loss.csv'}")
        print(f"Configuration: {output_dir / 'config' / 'training_config.json'}")
        print("=" * 60)
        
    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise


if __name__ == "__main__":
    main()
