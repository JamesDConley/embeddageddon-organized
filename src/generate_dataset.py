"""
Dataset generation utilities for MatFormer training.

This module provides functionality to generate and prepare datasets for
MatFormer model training. It handles loading datasets from HuggingFace,
splitting them into training and evaluation sets, and saving them in
parquet format for efficient training.

Functions:
    generate_fixed_dataset: Generate and save train/eval datasets from HuggingFace.
    parse_args: Parse command line arguments for dataset generation.
"""

import pandas as pd
import argparse
import os
from datasets import load_dataset
from tqdm import tqdm


def generate_fixed_dataset(dataset_name, split, num_samples, output_dir, eval_split, cache_dir):
    """
    Generate a fixed dataset from HuggingFace and save train/eval splits as separate parquet files.
    
    This function loads a dataset from HuggingFace, processes it to extract text content,
    splits it into training and evaluation sets, and saves both as parquet files for
    efficient training.
    
    Args:
        dataset_name (str): Name of the HuggingFace dataset to load.
        split (str): Dataset split to use (e.g., 'train', 'validation', 'test').
        num_samples (int or None): Number of samples to extract. If None, uses entire dataset.
        output_dir (str): Directory to save the parquet files (will create train.parquet and eval.parquet).
        eval_split (float): Fraction of data to use for evaluation (e.g., 0.1 for 10%).
        cache_dir (str): Directory to cache the original dataset.
    
    Returns:
        tuple: A tuple containing (train_df, eval_df) as pandas DataFrames.
        
    Raises:
        Exception: If dataset loading fails or required fields are missing.
        
    Examples:
        >>> train_df, eval_df = generate_fixed_dataset(
        ...     "vilm/RedPajama-v2-small",
        ...     "train",
        ...     10000,
        ...     "./dataset",
        ...     0.1,
        ...     "/tmp/cache"
        ... )
    """
    print(f"Loading dataset: {dataset_name}")
    print(f"Split: {split}")
    print(f"Number of samples: {num_samples}")
    print(f"Evaluation split: {eval_split} ({eval_split*100:.1f}%)")
    
    # Create cache directory
    os.makedirs(cache_dir, exist_ok=True)
    
    try:
        # Load the dataset
        dataset = load_dataset(dataset_name, split=split, cache_dir=cache_dir)
        print(f"Dataset loaded successfully. Total samples available: {len(dataset)}")
        
        # Handle num_samples parameter
        if num_samples is None:
            num_samples = len(dataset)
            print(f"Using entire dataset: {num_samples} samples")
        elif num_samples > len(dataset):
            print(f"Warning: Requested {num_samples} samples but dataset only has {len(dataset)}. Using all available samples.")
            num_samples = len(dataset)
        else:
            print(f"Using {num_samples} samples from dataset")
        
        # Shuffle with fixed seed for reproducibility
        dataset = dataset.shuffle(seed=42)
        
        # Select the specified number of samples
        dataset = dataset.select(range(num_samples))
        
        print(f"Selected {num_samples} samples from dataset")
        
        # Convert to pandas DataFrame
        print("Converting to pandas DataFrame...")
        data_list = []
        
        for i, example in enumerate(tqdm(dataset, desc="Processing samples", dynamic_ncols=True)):
            # Extract text field (assuming the dataset has a 'text' field)
            if 'text' in example:
                data_list.append({
                    'id': i,
                    'text': example['text']
                })
            else:
                # If no 'text' field, look for other common text fields
                text_fields = ['content', 'document', 'article', 'passage']
                text_content = None
                for field in text_fields:
                    if field in example:
                        text_content = example[field]
                        break
                
                if text_content is None:
                    print(f"Warning: No text field found in sample {i}. Available fields: {list(example.keys())}")
                    continue
                
                data_list.append({
                    'id': i,
                    'text': text_content
                })
        
        # Create DataFrame
        df = pd.DataFrame(data_list)
        print(f"Created DataFrame with {len(df)} samples")
        
        # Split into train and eval
        eval_size = int(len(df) * eval_split)
        train_size = len(df) - eval_size
        
        # Use the same shuffled order but split deterministically
        eval_df = df.iloc[:eval_size].copy()
        train_df = df.iloc[eval_size:].copy()
        
        # Reset indices and update IDs
        eval_df = eval_df.reset_index(drop=True)
        train_df = train_df.reset_index(drop=True)
        eval_df['id'] = range(len(eval_df))
        train_df['id'] = range(len(train_df))
        
        print(f"Split dataset:")
        print(f"  Training samples: {len(train_df)}")
        print(f"  Evaluation samples: {len(eval_df)}")
        
        # Create output directory and define fixed filenames
        os.makedirs(output_dir, exist_ok=True)
        train_output_path = os.path.join(output_dir, "train.parquet")
        eval_output_path = os.path.join(output_dir, "eval.parquet")
        
        # Save train parquet
        print(f"Saving training data to: {train_output_path}")
        train_df.to_parquet(train_output_path, index=False)
        
        # Save eval parquet
        print(f"Saving evaluation data to: {eval_output_path}")
        eval_df.to_parquet(eval_output_path, index=False)
        
        print(f"Dataset saved successfully!")
        print(f"Training dataset shape: {train_df.shape}")
        print(f"Evaluation dataset shape: {eval_df.shape}")
        
        # Print statistics for training data
        print(f"Training data text length statistics:")
        train_text_lengths = train_df['text'].str.len()
        print(f"  Mean: {train_text_lengths.mean():.1f}")
        print(f"  Median: {train_text_lengths.median():.1f}")
        print(f"  Min: {train_text_lengths.min()}")
        print(f"  Max: {train_text_lengths.max()}")
        
        # Print statistics for evaluation data
        print(f"Evaluation data text length statistics:")
        eval_text_lengths = eval_df['text'].str.len()
        print(f"  Mean: {eval_text_lengths.mean():.1f}")
        print(f"  Median: {eval_text_lengths.median():.1f}")
        print(f"  Min: {eval_text_lengths.min()}")
        print(f"  Max: {eval_text_lengths.max()}")
        
        return train_df, eval_df
        
    except Exception as e:
        print(f"Error loading dataset: {e}")
        print("Please check the dataset name and split are correct.")
        print("You can find available datasets at: https://huggingface.co/datasets")
        raise


def parse_args():
    """
    Parse command line arguments for dataset generation.
    
    This function sets up and parses command line arguments for the dataset
    generation script, providing sensible defaults for common use cases.
    
    Returns:
        argparse.Namespace: Parsed command line arguments.
        
    Examples:
        >>> args = parse_args()
        >>> print(args.dataset_name)  # "vilm/RedPajama-v2-small"
    """
    parser = argparse.ArgumentParser(description="Generate fixed dataset for MatFormer training comparison")
    
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="vilm/RedPajama-v2-small",
        help="HuggingFace dataset name (default: vilm/RedPajama-v2-small)"
    )
    
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        help="Dataset split to use (default: train)"
    )
    
    parser.add_argument(
        "--num_samples",
        type=int,
        default=None,
        help="Number of samples to extract (default: None, uses entire dataset)"
    )
    
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./dataset",
        help="Directory to save the dataset files (will create train.parquet and eval.parquet) (default: ./dataset)"
    )
    
    parser.add_argument(
        "--eval_split",
        type=float,
        default=0.1,
        help="Fraction of data to use for evaluation (default: 0.1)"
    )
    
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/big_data/datasets_cache",
        help="Directory to cache dataset (default: /big_data/datasets_cache)"
    )
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    
    print("="*60)
    print("Fixed Dataset Generator for MatFormer Training Comparison")
    print("="*60)
    
    generate_fixed_dataset(
        dataset_name=args.dataset_name,
        split=args.split,
        num_samples=args.num_samples,
        output_dir=args.output_dir,
        eval_split=args.eval_split,
        cache_dir=args.cache_dir
    )
    
    print("\n" + "="*60)
    print("Dataset generation completed!")
    print(f"Training data saved to: {os.path.join(args.output_dir, 'train.parquet')}")
    print(f"Evaluation data saved to: {os.path.join(args.output_dir, 'eval.parquet')}")
    print(f"You can now use '{args.output_dir}' with the modified training scripts.")
    print("="*60)
