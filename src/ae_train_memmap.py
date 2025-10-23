#!/usr/bin/env python3

"""
Train Embeddageddon - Autoencoder using Memory-Mapped Preprocessed Data

This script trains the autoencoder using preprocessed memory-mapped data files,
avoiding memory leaks and enabling efficient streaming from disk.

Usage:
    1. First run: python preprocess_embeddings.py --embedding_dir <dir> --output_dir <preprocessed_dir>
    2. Then run: python ae_train_memmap.py --preprocessed_dir <preprocessed_dir>
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
from autoencoder.memmap_dataset import MemMapEmbeddingDataset
from autoencoder.model import EmbeddingAutoencoder, masked_mse_loss


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


def save_training_config(output_dir: Path, args: argparse.Namespace, dataset: MemMapEmbeddingDataset):
    """
    Save training configuration and dataset info to the output directory.

    Args:
        output_dir: Output directory path
        args: Command line arguments
        dataset: Training dataset
    """
    activation_type = "tanh" if args.use_tanh else "linear"
    config = {
        "training_type": f"memmap_train_with_random_subnetwork_{activation_type}_l2",
        "training_args": {
            "preprocessed_dir": args.preprocessed_dir,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "device": args.device,
            "use_amp": args.use_amp,
            "precision": args.precision,
            "num_workers": args.num_workers,
            "l2_penalty_weight": args.l2_penalty_weight,
            "use_tanh": args.use_tanh,
            "weight_decay": args.weight_decay
        },
        "scheduler_info": {
            "use_onecycle": args.use_onecycle,
            "max_lr": args.learning_rate,
            "pct_start": args.pct_start,
            "div_factor": args.div_factor,
            "final_div_factor": args.final_div_factor,
            "initial_lr": args.learning_rate / args.div_factor if args.use_onecycle else args.learning_rate,
            "final_lr": args.learning_rate / args.final_div_factor if args.use_onecycle else args.learning_rate
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
        "model_architecture": {
            "bottleneck_activation": "tanh" if args.use_tanh else "linear",
            "l2_regularization": True,
            "l2_penalty_weight": args.l2_penalty_weight
        },
        "timestamp": datetime.now().isoformat()
    }

    config_path = output_dir / "config" / "training_config.json"
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Training configuration saved to: {config_path}")


def train_autoencoder(dataset: MemMapEmbeddingDataset,
                     output_dir: Path,
                     num_epochs: int = 100,
                     batch_size: int = 32,
                     learning_rate: float = 0.001,
                     device: str = "cuda" if torch.cuda.is_available() else "cpu",
                     use_amp: bool = True,
                     precision: str = "fp16",
                     num_workers: int = 2,
                     l2_penalty_weight: float = 0.01,
                     use_onecycle: bool = True,
                     pct_start: float = 0.3,
                     div_factor: float = 25.0,
                     final_div_factor: float = 10000.0,
                     use_tanh: bool = False,
                     weight_decay: float = 0.0) -> EmbeddingAutoencoder:
    """
    Train the embedding autoencoder.

    Args:
        dataset: Training dataset
        output_dir: Directory to save all outputs
        num_epochs: Number of training epochs
        batch_size: Batch size for training
        learning_rate: Peak learning rate (max_lr for OneCycleLR if enabled, constant LR otherwise)
        device: Device to train on
        use_amp: Whether to use Automatic Mixed Precision
        precision: Precision mode for mixed precision training (fp16 or fp8)
        num_workers: Number of worker processes for data loading (0 = main process only)
        l2_penalty_weight: Weight for L2 penalty on embeddings (default: 0.01)
        use_onecycle: Whether to use OneCycleLR scheduler (default: True)
        pct_start: Fraction of training spent in warmup phase (default: 0.3)
        div_factor: Initial LR = max_lr / div_factor (default: 25.0)
        final_div_factor: Final LR = max_lr / final_div_factor (default: 10000.0)
        use_tanh: Whether to use tanh activation on bottleneck (default: False for linear)
        weight_decay: Weight decay (L2 penalty on weights) for Adam optimizer (default: 0.0)

    Returns:
        Trained autoencoder model
    """
    # Create data loader with async loading for better GPU utilization
    # Memory-mapped dataset streams from disk, avoiding memory bloat
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,  # Parallel data loading
        pin_memory=False,         # Disabled: causes memory leak with multiprocessing
        persistent_workers=True 
    
    )

    # Initialize model
    model = EmbeddingAutoencoder(
        input_dim=dataset.total_dim,
        bottleneck_dim=dataset.bottleneck_dim,
        use_tanh=use_tanh
    )
    model.to(device)

    # Initialize optimizer
    # If using OneCycleLR, initial LR will be set by the scheduler
    if use_onecycle:
        # Set a low initial LR; OneCycleLR will control the actual learning rate
        initial_lr = learning_rate / div_factor
        optimizer = optim.Adam(model.parameters(), lr=initial_lr, weight_decay=weight_decay)
    else:
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    # Initialize AMP components if using mixed precision
    scaler = None
    autocast_dtype = torch.bfloat16  # Default to BF16

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

    # Initialize OneCycleLR scheduler if requested
    scheduler = None
    if use_onecycle:
        steps_per_epoch = len(dataloader)
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=learning_rate,
            epochs=num_epochs,
            steps_per_epoch=steps_per_epoch,
            pct_start=pct_start,
            anneal_strategy='cos',
            div_factor=div_factor,
            final_div_factor=final_div_factor
        )
        print(f"OneCycleLR Scheduler: Enabled")
        print(f"  Max LR: {learning_rate}")
        print(f"  Initial LR: {learning_rate / div_factor:.6f}")
        print(f"  Final LR: {learning_rate / final_div_factor:.8f}")
        print(f"  Warmup fraction: {pct_start}")
        print(f"  Steps per epoch: {steps_per_epoch}")
    else:
        print(f"Learning Rate Scheduler: Disabled (constant LR: {learning_rate})")

    print(f"Training autoencoder on {device}")
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Training samples: {len(dataset)}")
    print(f"Bottleneck Activation: {'Tanh' if use_tanh else 'Linear'}")
    print(f"Mixed Precision: {'Enabled' if use_amp else 'Disabled'}")
    print(f"Random Subnetwork Training: Enabled (s, m, l, xl)")
    print(f"L2 Penalty Weight: {l2_penalty_weight}")
    print(f"Weight Decay: {weight_decay}")
    print(f"Data loading: {num_workers} workers (memory-mapped)")
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
            model.configure_subnetwork(flag)
            # ----------------------------------------------

            optimizer.zero_grad()

            if use_amp and scaler is not None:
                # Mixed precision forward pass
                with torch.amp.autocast("cuda", autocast_dtype):
                    outputs, l2_penalty = model(batch_inputs, return_l2_penalty=True)
                    reconstruction_loss = masked_mse_loss(outputs, batch_targets, batch_masks)
                    loss = reconstruction_loss + l2_penalty_weight * l2_penalty

                # Mixed precision backward pass
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                # Standard precision forward pass
                outputs, l2_penalty = model(batch_inputs, return_l2_penalty=True)
                reconstruction_loss = masked_mse_loss(outputs, batch_targets, batch_masks)
                loss = reconstruction_loss + l2_penalty_weight * l2_penalty

                # Standard precision backward pass
                loss.backward()
                optimizer.step()

            # Step the scheduler after each batch (if using OneCycleLR)
            if scheduler is not None:
                scheduler.step()

            # Cache loss.item() to avoid multiple GPU syncs
            batch_loss = loss.item()
            current_lr = optimizer.param_groups[0]['lr']
            total_loss += batch_loss
            num_batches += 1

            # Log batch loss to CSV
            loss_tracker.log_batch(
                epoch=epoch + 1,
                batch=batch_idx + 1,
                batch_loss=batch_loss,
                learning_rate=current_lr,
                batch_size=batch_size
            )

            # Update progress bar
            progress_bar.set_postfix({"Loss": f"{batch_loss:.6f}", "LR": f"{current_lr:.6f}", "Subnet": flag})

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
    parser = argparse.ArgumentParser(description="Train Embeddageddon Autoencoder (Memory-Mapped)")
    parser.add_argument("--preprocessed_dir", type=str, default="preprocessed_embeddings",
                       help="Directory containing preprocessed memory-mapped files")
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
    parser.add_argument("--l2_penalty_weight", type=float, default=0.01,
                       help="Weight for L2 penalty on embeddings (default: 0.01)")
    parser.add_argument("--use_onecycle", action="store_true", default=True,
                       help="Use OneCycleLR scheduler (default: True)")
    parser.add_argument("--no_onecycle", dest="use_onecycle", action="store_false",
                       help="Disable OneCycleLR scheduler")
    parser.add_argument("--pct_start", type=float, default=0.3,
                       help="Fraction of training in warmup phase for OneCycleLR (default: 0.3)")
    parser.add_argument("--div_factor", type=float, default=25.0,
                       help="Initial LR = max_lr / div_factor for OneCycleLR (default: 25.0)")
    parser.add_argument("--final_div_factor", type=float, default=10000.0,
                       help="Final LR = max_lr / final_div_factor for OneCycleLR (default: 10000.0)")
    parser.add_argument("--use_tanh", action="store_true", default=False,
                       help="Use tanh activation on bottleneck (default: False for linear)")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                       help="Weight decay (L2 penalty on weights) for Adam optimizer (default: 0.0)")
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

    print("=" * 60)
    print("EMBEDDAGEDDON TRAINING (MEMORY-MAPPED)")
    print("=" * 60)
    print(f"Preprocessed data directory: {args.preprocessed_dir}")
    print(f"Training device: {device}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Output directory: {output_dir}")
    print("-" * 60)

    try:
        # Load preprocessed dataset
        print("Loading memory-mapped dataset...")
        dataset = MemMapEmbeddingDataset(args.preprocessed_dir)

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
            num_workers=args.num_workers,
            l2_penalty_weight=args.l2_penalty_weight,
            use_onecycle=args.use_onecycle,
            pct_start=args.pct_start,
            div_factor=args.div_factor,
            final_div_factor=args.final_div_factor,
            use_tanh=args.use_tanh,
            weight_decay=args.weight_decay
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
