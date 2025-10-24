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
import time

# Third-party imports
import bitsandbytes as bnb
import torch
from tqdm import tqdm
from transformers import AutoConfig, get_scheduler

from transformer_engine.common import recipe

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


def setup_model_with_embeddageddon_embeddings(model_class, model_config, device, embedding_path):
    """Set up model with embeddageddon embeddings instead of regular embeddings.
    
    Args:
        model_class: Model class to instantiate
        model_config (str): Path to model config
        device: Device to place model on
        embedding_path (str): Path to the embeddageddon embeddings file
        
    Returns:
        Model instance with embeddageddon embeddings
    """
    
    # Load embeddageddon embeddings
    embeddings = load_embeddageddon_embeddings(embedding_path)
    
    # Get embedding dimension
    first_embedding = next(iter(embeddings.values()))
    embedding_dim = first_embedding.shape[0]
    logger.info(f"Embedding dimension: {embedding_dim}")
    
    # Load model config and modify hidden_size to match embedding dimension
    
    config = AutoConfig.from_pretrained(model_config)
    assert config.hidden_size == embedding_dim, "Embedding size mismatch!"
    
    # Initialize model
    model = model_class(config)
    
    # Replace embedding layer with embeddageddon embeddings
    vocab_size = len(embeddings)
    embedding_matrix = torch.zeros(vocab_size, embedding_dim)
    
    # Create tokenizer to get token ordering
    tokenizer = setup_tokenizer()
    vocab = tokenizer.get_vocab()
    
    # Fill embedding matrix
    for token, embedding in embeddings.items():
        if token in vocab:
            token_id = vocab[token]
            if token_id < vocab_size:
                embedding_matrix[token_id] = torch.from_numpy(embedding)
    
    # Replace the embedding layer
    model.model.embed_tokens = torch.nn.Embedding.from_pretrained(embedding_matrix, freeze=False)
    
    logger.info(f"Replaced embedding layer with embeddageddon embeddings")
    logger.info(f"Model vocab size: {vocab_size}, embedding dim: {embedding_dim}")
    
    model.to(device)
    return model


def main():
    """Main training function."""
    parser = setup_parser()
    parser.add_argument("--embedding_file",
                       help="File containing embeddageddon embeddings. Must match the model configs hidden dim")
    args = parser.parse_args()
    
    # Save arguments to JSON file in output directory
    os.makedirs(args.output_dir, exist_ok=True)
    try:
        args_dict = vars(args)
        args_json_path = os.path.join(args.output_dir, "args_used.json")
        with open(args_json_path, 'w') as f:
            json.dump(args_dict, f, indent=2)
        print(f"Arguments saved to: {args_json_path}")
        
        device = setup_device(args.device)
        

        # Check if embeddageddon embeddings should be used
        use_embeddageddon = (args.embedding_file and 
                            args.embedding_file.lower() != "none" and 
                            args.embedding_file.strip() != "")
        
        # Setup model - with or without embeddageddon embeddings
        if use_embeddageddon:
            print(f"Using embeddageddon embeddings from: {args.embedding_file}")
            if args.model_type == "matformer":
                model = setup_model_with_embeddageddon_embeddings(
                    BaseMatformer, args.config_name, device, args.embedding_file
                )
            elif args.model_type == "weight_based_matformer":
                model = setup_model_with_embeddageddon_embeddings(
                    WeightBasedMatformer, args.config_name, device, args.embedding_file
                )
            else:
                model = setup_model_with_embeddageddon_embeddings(
                    ModifiedMatformer, args.config_name, device, args.embedding_file
                )
        else:
            print("Using regular model setup (no embeddageddon embeddings)")
            if args.model_type == "matformer":
                model = setup_model(BaseMatformer, model_name=args.config_name, max_length=args.max_length, device=device)
            elif args.model_type == "weight_based_matformer":
                model = setup_model(WeightBasedMatformer, model_name=args.config_name, max_length=args.max_length, device=device)
            else:
                model = setup_model(ModifiedMatformer, model_name=args.config_name, max_length=args.max_length, device=device)
        
        optimizer = bnb.optim.Adam8bit(model.parameters(), lr=args.learning_rate)
        
        
        model.train()
        total_params = sum(param.numel() for param in model.parameters())
        print(f"Total number of parameters: {total_params}")

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

                precision_context = torch.amp.autocast('cuda', dtype=torch.bfloat16)
                scaler = torch.amp.GradScaler('cuda')
                
                # Forward pass with appropriate precision context
                with precision_context:
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
                            covariance_loss = torch.tensor(0.0, device=device)
                    else:
                        total_loss = main_loss
                        covariance_loss = torch.tensor(0.0, device=device)
                
                optimizer.zero_grad()
                scaler.scale(total_loss).backward()
                if hasattr(model, 'zero_non_slice_gradients'):
                    model.zero_non_slice_gradients()
                scaler.step(optimizer)
                scaler.update()
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
            eval_losses = evaluate_model(model, eval_dataloader, flags)
            for flag, loss_dict in eval_losses.items():
                tracker.write_eval(epoch=epoch, step=global_step, total_loss=loss_dict['total_loss'],
                                main_loss=loss_dict['main_loss'], covariance_loss=loss_dict['covariance_loss'],
                                current_subnetwork=flag)
            # Save checkpoint
            save_checkpoint(model, optimizer, scheduler, epoch, global_step, sub_network, args.output_dir)

        save_final_model(model, tokenizer, args.output_dir)
        tracker.close_files()
    except:
        print(f"Error processing, removing output directory in 5 seconds.")
        time.sleep(5)
        os.rmdir(args.output_dir)


if __name__ == "__main__":
    main()
