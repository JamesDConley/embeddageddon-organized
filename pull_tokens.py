#!/usr/bin/env python3
"""
Script to extract a specific number of tokens from parquet files.
"""
import argparse
import glob
import random
from pathlib import Path
import pandas as pd


def main():
    parser = argparse.ArgumentParser(
        description="Extract a specific number of tokens from parquet files"
    )
    parser.add_argument(
        "--input_folder",
        type=str,
        required=True,
        help="Path to folder containing .parquet files"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        required=True,
        help="Path to output parquet file (will be split into chunks)"
    )
    parser.add_argument(
        "--num_tokens",
        type=int,
        required=True,
        help="Target number of tokens to extract"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for shuffling files (default: 42)"
    )
    parser.add_argument(
        "--chunk_size",
        type=int,
        default=100000,
        help="Number of rows per chunk file (default: 100000)"
    )
    
    args = parser.parse_args()
    
    # Constants
    CHARS_PER_TOKEN = 10
    MAX_TOKENS = 512
    MAX_CHARS = CHARS_PER_TOKEN * MAX_TOKENS
    CHUNK_SIZE = args.chunk_size
    
    # List all parquet files and shuffle
    pattern = str(Path(args.input_folder) / "*.parquet")
    all_paths = glob.glob(pattern)
    
    if not all_paths:
        raise ValueError(f"No .parquet files found in {args.input_folder}")
    
    print(f"Found {len(all_paths)} parquet files")
    
    # Shuffle the file list
    random.seed(args.seed)
    random.shuffle(all_paths)
    
    # Setup output directory
    output_path = Path(args.output_file)
    output_dir = output_path.parent
    output_stem = output_path.stem
    output_suffix = output_path.suffix
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Variables for chunked writing
    accumulated_data = []
    accumulated_rows = 0
    total_tokens = 0
    chunk_index = 0
    reached_target = False
    
    for i, path in enumerate(all_paths):
        print(f"Processing file {i+1}/{len(all_paths)}: {Path(path).name}")
        
        # Read the file
        df = pd.read_parquet(path)
        
        # Subset text to MAX_CHARS characters if above it
        df['text'] = df['text'].apply(
            lambda x: x[:MAX_CHARS] if len(x) > MAX_CHARS else x
        )
        
        # Replace token_count with min of current value and MAX_TOKENS
        df['token_count'] = df['token_count'].apply(
            lambda x: min(x, MAX_TOKENS)
        )
        
        # Select only needed columns
        df = df[['text', 'token_count']].copy()
        
        # Add to accumulated data
        accumulated_data.append(df)
        accumulated_rows += len(df)
        
        # Update total tokens
        file_tokens = df['token_count'].sum()
        total_tokens += file_tokens
        
        print(f"  Total tokens accumulated: {total_tokens}/{args.num_tokens}")
        
        # Check if we've reached the target
        if total_tokens >= args.num_tokens:
            reached_target = True
        
        # Write chunk if we have enough rows OR reached target
        if accumulated_rows >= CHUNK_SIZE or reached_target:
            # Combine accumulated data
            chunk_df = pd.concat(accumulated_data, ignore_index=True)
            
            # If we reached target, trim the chunk
            if reached_target:
                # Calculate cumulative tokens to find where to cut
                chunk_df['cumulative_tokens'] = chunk_df['token_count'].cumsum()
                
                # Adjust for tokens already written in previous chunks
                tokens_in_previous_chunks = total_tokens - chunk_df['token_count'].sum()
                chunk_df['cumulative_tokens'] += tokens_in_previous_chunks
                
                # Find rows that don't exceed target
                valid_indices = chunk_df[chunk_df['cumulative_tokens'] <= args.num_tokens].index
                
                if len(valid_indices) > 0:
                    last_valid_idx = valid_indices[-1]
                    
                    # Check if adding one more row gets closer without going over
                    if last_valid_idx + 1 < len(chunk_df):
                        tokens_at_valid = chunk_df.loc[last_valid_idx, 'cumulative_tokens']
                        tokens_with_next = chunk_df.loc[last_valid_idx + 1, 'cumulative_tokens']
                        
                        if tokens_with_next <= args.num_tokens or \
                           (tokens_with_next - args.num_tokens) < (args.num_tokens - tokens_at_valid):
                            last_valid_idx += 1
                    
                    chunk_df = chunk_df.iloc[:last_valid_idx + 1].copy()
                else:
                    # Keep at least one row
                    chunk_df = chunk_df.iloc[:1].copy()
                
                # Remove helper column
                chunk_df = chunk_df[['text', 'token_count']]
                
                print(f"  Trimmed final chunk to {len(chunk_df)} rows")
            
            # Write chunk to file
            chunk_filename = f"{output_stem}_chunk_{chunk_index:04d}{output_suffix}"
            chunk_path = output_dir / chunk_filename
            chunk_df.to_parquet(chunk_path, index=False)
            
            print(f"  Wrote chunk {chunk_index}: {chunk_path.name} ({len(chunk_df)} rows)")
            
            chunk_index += 1
            
            # Clear accumulated data
            accumulated_data = []
            accumulated_rows = 0
            
            if reached_target:
                break
        
        if reached_target:
            break
    
    # Calculate final token count
    final_tokens = total_tokens
    if reached_target and 'chunk_df' in locals():
        final_tokens = chunk_df['token_count'].sum()
        # Add tokens from previous chunks
        final_tokens += total_tokens - chunk_df['token_count'].sum() - file_tokens + chunk_df['token_count'].sum()
        # Simpler: recalculate from cumulative if available
        if 'cumulative_tokens' in chunk_df.columns:
            final_tokens = int(chunk_df['cumulative_tokens'].iloc[-1])
    
    print(f"\nSuccess!")
    print(f"Output files: {output_stem}_chunk_XXXX{output_suffix}")
    print(f"Total chunks: {chunk_index}")
    print(f"Estimated total tokens: {int(final_tokens)}")
    print(f"Target tokens: {args.num_tokens}")


if __name__ == "__main__":
    main()