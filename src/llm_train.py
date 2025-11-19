"""
Training script for LLM with optional Embeddageddon embeddings using MatFormer architecture.

This script provides a training pipeline for MatFormer-style models that can optionally load
embeddageddon embeddings instead of regular token embeddings. If no embedding file is provided
or "None" is passed, it will use regular model initialization.
"""

# Standard library imports
import json
import logging
import os
import pickle
import random
import shutil
import sys
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Third-party imports
import bitsandbytes as bnb
import torch
from tqdm import tqdm
from transformers import AutoConfig, get_scheduler
from accelerate import Accelerator
from accelerate.utils import TERecipeKwargs

# Local MatFormer imports
from llm.frozen_matformer import ModifiedLlamaForCausalLM as ModifiedMatformer
from llm.get_args import setup_parser
from llm.modified_llama import ModifiedLlamaForCausalLM as BaseMatformer
from llm.weight_based_matformer import ModifiedLlamaForCausalLM as WeightBasedMatformer
from llm.training_tracker import TrainingTracker
from llm.training_utils import setup_tokenizer, setup_model
from llm.train import (
    setup_device, 
    setup_dataloaders, 
    evaluate_model, 
    save_checkpoint, 
    save_final_model
)

logger = logging.getLogger(__name__)


def print_model_parameters(model):
    """Print detailed breakdown of model parameters by layer and component.
    
    Args:
        model: PyTorch model to analyze
    """
    print("\n" + "="*80)
    print("MODEL PARAMETER BREAKDOWN")
    print("="*80)

    total_params = 0
    layer_params = {}

    for name, param in model.named_parameters():
        num_params = param.numel()
        total_params += num_params

        # Extract layer/component name (first part before the dot or full name if no dot)
        parts = name.split('.')
        if len(parts) > 1:
            # Group by major component (e.g., model.embed_tokens, model.layers.0, etc.)
            if parts[1] == 'layers' and len(parts) > 2:
                # For transformer layers, group by layer number
                layer_key = f"{parts[0]}.{parts[1]}.{parts[2]}"
            else:
                # For other components (embeddings, norm, lm_head, etc.)
                layer_key = f"{parts[0]}.{parts[1]}"
        else:
            layer_key = parts[0]

        if layer_key not in layer_params:
            layer_params[layer_key] = {'params': 0, 'details': {}}

        layer_params[layer_key]['params'] += num_params

        # Store individual parameter details
        param_name = '.'.join(parts[2:]) if len(parts) > 2 and parts[1] == 'layers' else '.'.join(parts[1:]) if len(parts) > 1 else name
        layer_params[layer_key]['details'][param_name] = num_params

    # Print summary by major components
    for layer_name in sorted(layer_params.keys()):
        layer_info = layer_params[layer_name]
        percentage = (layer_info['params'] / total_params) * 100
        print(f"\n{layer_name}:")
        print(f"  Total: {layer_info['params']:,} parameters ({percentage:.2f}%)")

        # Print details for this component (sorted by parameter count)
        sorted_details = sorted(layer_info['details'].items(), key=lambda x: x[1], reverse=True)
        for param_name, param_count in sorted_details:
            param_percentage = (param_count / layer_info['params']) * 100
            print(f"    - {param_name}: {param_count:,} ({param_percentage:.1f}%)")

    print("\n" + "="*80)
    print(f"TOTAL PARAMETERS: {total_params:,}")
    print("="*80 + "\n")



def load_embeddageddon_embeddings(path):
    """Load embeddageddon embeddings from the specified path.
    
    Args:
        path (str): Path to the pickled embeddings file
        
    Returns:
        dict: Dictionary mapping tokens to embeddings
    """
    
    logger.info(f"Loading embeddings from {path}")
    
    with open(path, 'rb') as f:
        embeddings = pickle.load(f)
    
    logger.info(f"Loaded {len(embeddings)} embeddings")
    return embeddings


def setup_model_with_embeddageddon_embeddings(model_class, model_config, device, embedding_path, scale_factor=1.0):
    """Set up model with embeddageddon embeddings instead of regular embeddings.
    
    Args:
        model_class: Model class to instantiate
        model_config (str): Path to model config
        device: Device to place model on
        embedding_path (str): Path to the embeddageddon embeddings file
        scale_factor (float): Factor to divide embeddings by (default: 1.0, no scaling)
        
    Returns:
        Model instance with embeddageddon embeddings
    """
    
    # Load embeddageddon embeddings
    embeddings = load_embeddageddon_embeddings(embedding_path)
    
    # Get embedding dimension
    first_embedding = next(iter(embeddings.values()))
    embedding_dim = first_embedding.shape[0]
    logger.info(f"Embedding dimension: {embedding_dim}")
    
    # Load model config
    config = AutoConfig.from_pretrained(model_config)
    
    # Handle dimension mismatch - truncate if embeddings are larger than hidden_size
    if embedding_dim > config.hidden_size:
        logger.warning(f"Embedding dimension ({embedding_dim}) is larger than config.hidden_size ({config.hidden_size}). "
                      f"Truncating embeddings to first {config.hidden_size} dimensions.")
        # Truncate all embeddings to match config.hidden_size
        embeddings = {token: emb[:config.hidden_size] for token, emb in embeddings.items()}
        embedding_dim = config.hidden_size
    elif embedding_dim < config.hidden_size:
        raise ValueError(f"Embedding dimension ({embedding_dim}) is smaller than config.hidden_size ({config.hidden_size}). "
                        f"Cannot proceed - embeddings would need padding which is not supported.")
    
    assert config.hidden_size == embedding_dim, "Embedding size mismatch after truncation!"
    
    # Initialize model
    model = model_class(config)
    
    # Create tokenizer to get token ordering
    tokenizer = setup_tokenizer()
    vocab = tokenizer.get_vocab()
    total_vocab_size = len(vocab)
    
    # Replace embedding layer with embeddageddon embeddings
    # Start with model's existing embeddings instead of zeros
    embedding_matrix = model.model.embed_tokens.weight.data.clone()
    
    # Track coverage
    tokens_with_embeddings = 0
    
    # Fill embedding matrix with optional scaling
    for token, embedding in embeddings.items():
        if token in vocab:
            token_id = vocab[token]
            if token_id < total_vocab_size:
                embedding_tensor = torch.from_numpy(embedding)
                if scale_factor != 1.0:
                    embedding_tensor = embedding_tensor / scale_factor
                embedding_matrix[token_id] = embedding_tensor
                tokens_with_embeddings += 1
    
    # Log vocabulary coverage statistics
    coverage_percentage = (tokens_with_embeddings / total_vocab_size) * 100
    logger.info(f"Vocabulary coverage: {tokens_with_embeddings}/{total_vocab_size} tokens ({coverage_percentage:.2f}%)")
    
    # Replace the embedding layer
    with torch.no_grad():
        model.model.embed_tokens.weight.copy_(embedding_matrix)
    
    if scale_factor != 1.0:
        logger.info(f"Scaled embeddings by factor of {scale_factor} (divided values)")
    logger.info(f"Replaced embedding layer with embeddageddon embeddings")
    logger.info(f"Model vocab size: {total_vocab_size}, embedding dim: {embedding_dim}")
    
    model.to(device)
    return model


def main():
    """Main training function."""
    parser = setup_parser()
    parser.add_argument("--embedding_file",
                       help="File containing embeddageddon embeddings. Must match the model configs hidden dim")
    parser.add_argument("--use_fp8", action="store_true",
                       help="Enable FP8 mixed precision training with TransformerEngine")
    parser.add_argument("--fp8_format", type=str, default="HYBRID", choices=["HYBRID", "E4M3", "E5M2"],
                       help="FP8 format to use (default: HYBRID for training)")
    parser.add_argument("--fp8_amax_history_len", type=int, default=1024,
                       help="Length of history for FP8 scaling factor computation")
    parser.add_argument("--fp8_amax_compute_algo", type=str, default="most_recent", choices=["max", "most_recent"],
                       help="Algorithm for FP8 scaling factor computation")
    args = parser.parse_args()
    
    # Save arguments to JSON file in output directory
    os.makedirs(args.output_dir, exist_ok=True)
    try:
        args_dict = vars(args)
        args_json_path = os.path.join(args.output_dir, "args_used.json")
        with open(args_json_path, 'w') as f:
            json.dump(args_dict, f, indent=2)
        print(f"Arguments saved to: {args_json_path}")
        
        # Initialize Accelerator with optional FP8 support
        if args.use_fp8:
            # Configure FP8 with TransformerEngine
            te_kwargs = TERecipeKwargs(
                fp8_format=args.fp8_format,
                amax_history_len=args.fp8_amax_history_len,
                amax_compute_algo=args.fp8_amax_compute_algo,
                use_autocast_during_eval=False,  # Better metrics during eval
            )
            accelerator = Accelerator(
                mixed_precision="fp8",
                gradient_accumulation_steps=1,
                kwargs_handlers=[te_kwargs]
            )
            print(f"Using FP8 mixed precision with TransformerEngine (format: {args.fp8_format})")
        else:
            # Use BF16 mixed precision
            accelerator = Accelerator(
                mixed_precision="bf16",
                gradient_accumulation_steps=1,
            )
            print("Using BF16 mixed precision")
        
        device = accelerator.device
        
        # Check if embeddageddon embeddings should be used
        use_embeddageddon = (args.embedding_file and 
                            args.embedding_file.lower() != "none" and 
                            args.embedding_file.strip() != "")
        
        # Setup model - with or without embeddageddon embeddings
        if use_embeddageddon:
            print(f"Using embeddageddon embeddings from: {args.embedding_file}")
            if args.scale_embeddings != 1.0:
                print(f"Scaling embeddings by factor {args.scale_embeddings} (dividing values)")
            
            if args.model_type == "matformer":
                model = setup_model_with_embeddageddon_embeddings(
                    BaseMatformer, args.config_name, device, args.embedding_file, args.scale_embeddings
                )
            elif args.model_type == "weight_based_matformer":
                model = setup_model_with_embeddageddon_embeddings(
                    WeightBasedMatformer, args.config_name, device, args.embedding_file, args.scale_embeddings
                )
            else:
                model = setup_model_with_embeddageddon_embeddings(
                    ModifiedMatformer, args.config_name, device, args.embedding_file, args.scale_embeddings
                )
        else:
            print("Using regular model setup (no embeddageddon embeddings)")
            if args.model_type == "matformer":
                model = setup_model(BaseMatformer, model_name=args.config_name, max_length=args.max_length, device=device)
            elif args.model_type == "weight_based_matformer":
                model = setup_model(WeightBasedMatformer, model_name=args.config_name, max_length=args.max_length, device=device)
            else:
                model = setup_model(ModifiedMatformer, model_name=args.config_name, max_length=args.max_length, device=device)
        
        # Freeze embeddings if requested
        if args.freeze_embeddings:
            print("Freezing embedding layer (no gradient updates)...")
            for param in model.model.embed_tokens.parameters():
                param.requires_grad = False
            print(f"Embedding parameters frozen: {sum(p.numel() for p in model.model.embed_tokens.parameters()):,}")
        
        # Use BitsAndBytes 8-bit optimizer
        optimizer = bnb.optim.Adam8bit(model.parameters(), lr=args.learning_rate)
        
        model.train()
        print_model_parameters(model)

        tokenizer = setup_tokenizer()
        eval_dataloader, train_dataloader = setup_dataloaders(
            args.dataset_dir, 
            args.seed, 
            tokenizer, 
            args.max_length, 
            args.batch_size,
            eval_samples=args.eval_samples
        )
        tracker = TrainingTracker(args.output_dir)


        flags = ['s', 'm', 'l', 'xl']

        total_batches_per_epoch = len(train_dataloader)
        batches_per_subnetwork = int(total_batches_per_epoch * args.num_epochs_per_subnetwork)
        print(f"batches_per_subnetwork : {batches_per_subnetwork}")
        if batches_per_subnetwork == 0:
            batches_per_subnetwork = 1

        num_actual_epochs = len(flags) * args.num_epochs_per_subnetwork
        total_steps_all_subnetworks = batches_per_subnetwork * len(flags)
        num_warmup_steps = int(0.1 * total_steps_all_subnetworks)
        scheduler = get_scheduler(
                "cosine",
                optimizer=optimizer,
                num_warmup_steps=num_warmup_steps,
                num_training_steps=total_steps_all_subnetworks,
            )
        
        # Prepare model, optimizer, dataloaders, and scheduler with Accelerate
        model, optimizer, train_dataloader, eval_dataloader, scheduler = accelerator.prepare(
            model, optimizer, train_dataloader, eval_dataloader, scheduler
        )
        
        global_step = 0

        data_iter = iter(train_dataloader)
        for sub_network in flags:
            model.configure_subnetwork(sub_network)
            subnetwork_loss = 0.0
            subnetwork_main_loss = 0.0
            subnetwork_covariance_loss = 0.0
            num_batches = 0
            print(f"batches_per_subnetwork : {batches_per_subnetwork}")
            for local_step in tqdm(range(batches_per_subnetwork), dynamic_ncols=True):
                try:
                    batch = next(data_iter)
                except StopIteration:
                    data_iter = iter(train_dataloader)
                    batch = next(data_iter)
                    
                input_ids = batch["input_ids"].to(model.device)
                attention_mask = batch["attention_mask"].to(model.device)
                labels = batch["labels"].to(model.device)
                
                # Determine which subnetwork to use for this batch
                if args.random_subnetwork_order:
                    current_subnetwork = random.choice(flags)
                    model.configure_subnetwork(current_subnetwork)
                else:
                    current_subnetwork = sub_network

                # Forward pass - Accelerate handles mixed precision automatically
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                main_loss = outputs.loss
                
                if args.model_type != "matformer":
                    # Get covariance loss
                    covariance_loss = model.get_covariance_loss()
                    
                    # Compute total loss
                    if covariance_loss is not None and args.covariance_loss_weight > 0:
                        total_loss = main_loss + (args.covariance_loss_weight * covariance_loss)
                    else:
                        total_loss = main_loss
                        covariance_loss = torch.tensor(0.0, device=accelerator.device)
                else:
                    total_loss = main_loss
                    covariance_loss = torch.tensor(0.0, device=accelerator.device)
                
                optimizer.zero_grad()
                # Use accelerator.backward() instead of scaler
                accelerator.backward(total_loss)
                if hasattr(model, 'zero_non_slice_gradients'):
                    model.zero_non_slice_gradients()
                optimizer.step()
                scheduler.step()

                current_total_loss = total_loss.item()
                current_main_loss = main_loss.item()
                current_covariance_loss = covariance_loss.item()
                
                subnetwork_loss += current_total_loss
                subnetwork_main_loss += current_main_loss
                subnetwork_covariance_loss += current_covariance_loss
                num_batches += 1
                global_step += 1

                tracker.write_train(epoch=global_step // total_batches_per_epoch, step=global_step,
                                total_loss=total_loss.item(), main_loss=main_loss.item(),
                                covariance_loss=covariance_loss.item(), current_subnetwork=current_subnetwork)
            
            epoch = global_step // total_batches_per_epoch
            eval_losses = evaluate_model(model, eval_dataloader, flags, accelerator)
            for flag, loss_dict in eval_losses.items():
                tracker.write_eval(epoch=epoch, step=global_step, total_loss=loss_dict['total_loss'],
                                main_loss=loss_dict['main_loss'], covariance_loss=loss_dict['covariance_loss'],
                                current_subnetwork=flag)
            # Save checkpoint - unwrap model for saving
            save_checkpoint(accelerator.unwrap_model(model), optimizer, scheduler, epoch, global_step, sub_network, args.output_dir)

        save_final_model(accelerator.unwrap_model(model), tokenizer, args.output_dir)
        tracker.close_files()
    except:
        print(f"Error processing, removing output directory in 5 seconds.")
        time.sleep(5)
        shutil.rmtree(args.output_dir)
        print(f"Raising Error")
        raise


if __name__ == "__main__":
    main()
