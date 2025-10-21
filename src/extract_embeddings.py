#!/usr/bin/env python3

"""
Efficient version of save_embeddings.py that only loads the necessary model files
for embedding extraction instead of loading the full model.

This script uses a generic extractor designed to work with any model type, downloading
only the required files (index and specific safetensors) from HuggingFace Hub.

Features:
- Universal support for any HuggingFace model with safetensors format
- Efficient file caching to avoid re-downloading
- Fast pickle format for saving embeddings (much faster than JSON)
- Automatic cleanup on errors
"""

import os
from embedding_extraction.extractor import GenericEfficientEmbeddingModel
from embedding_extraction.extractor_utils import save_embeddings_pickle

# We're just pulling embeddings which is just copying data
# No need to run on GPU
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def main():
    """Main function to process specified models efficiently."""
    
    models = [
        "openai/gpt-oss-120b",
        "openai/gpt-oss-20b", 
        "Qwen/Qwen3-Coder-30B-A3B-Instruct",
        "Qwen/Qwen3-30B-A3B-Instruct-2507",
        "Qwen/Qwen3-Coder-480B-A35B-Instruct",
        "Qwen/Qwen3-235B-A22B-Instruct-2507",
        "deepseek-ai/DeepSeek-R1-0528",
        "mistralai/Devstral-Small-2507",
        "mistralai/Magistral-Small-2507",
        "zai-org/GLM-4.5-Air",
        "zai-org/GLM-4.5",
        "moonshotai/Kimi-K2-Instruct",
        "MiniMaxAI/MiniMax-Text-01",
        "Qwen/QwQ-32B"
    ]
    
    model_dir = "data/embedding_extraction/models"
    output_dir = "data/embedding_extraction/embedding_dicts"
    
    print(f"Starting efficient embedding extraction for {len(models)} models")
    print(f"Model cache directory: {model_dir}")
    print(f"Output directory: {output_dir}")
    print("-" * 60)
    
    successful = 0
    failed = 0
    
    for model_name in models:
        if save_embeddings_for_model(model_name, model_dir, output_dir):
            successful += 1
        else:
            failed += 1
        print("-" * 60)
    
    print(f"Efficient embedding extraction complete!")
    print(f"Successfully processed: {successful} models")
    print(f"Failed to process: {failed} models")
    
    if failed > 0:
        print(f"Check the error messages above for details on failed models.")


def save_embeddings_for_model(model_name: str, model_dir: str = "hface_home", output_dir: str = "embedding_dicts") -> bool:
    """
    Save embeddings for a single model to a pickle file using efficient loading.
    
    Args:
        model_name: Name of the model to process
        model_dir: Directory containing cached models
        output_dir: Directory to save embedding pickle files
        
    Returns:
        True if successful, False otherwise
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Create safe filename from model name
    safe_model_name = model_name.replace("/", "_").replace("\\", "_")
    output_file = os.path.join(output_dir, f"{safe_model_name}_efficient.pkl")
    
    # Check if output file already exists
    if os.path.exists(output_file):
        print(f"Processing model: {model_name}")
        print(f"  Output file already exists: {output_file}")
        print(f"  Skipping {model_name}")
        return True
    
    try:
        print(f"Processing model: {model_name}")
        
        # Get the generic extractor for this model
        extractor = GenericEfficientEmbeddingModel(model_name, model_dir)
        if not extractor:
            return False
        
        # Get all embeddings
        print(f"  Extracting embeddings...")
        all_embeddings = extractor.get_all_embeddings()
        
        if not all_embeddings:
            print(f"  Warning: No embeddings found for {model_name}")
            return False
            
        print(f"  Found {len(all_embeddings)} token embeddings")
        
        # Save to pickle file (much faster than JSON)
        if save_embeddings_pickle(all_embeddings, output_file):
            print(f"  Successfully saved embeddings for {model_name}")
            return True
        else:
            return False
        
    except Exception as e:
        print(f"  Error processing {model_name}: {str(e)}")
        # Clean up partial file on error
        if os.path.exists(output_file):
            try:
                os.remove(output_file)
                print(f"  Cleaned up partial file: {output_file}")
            except Exception as cleanup_error:
                print(f"  Warning: Could not clean up partial file {output_file}: {cleanup_error}")
        raise e



if __name__ == "__main__":
    main()
