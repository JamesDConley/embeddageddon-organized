from typing import Dict, List
from pathlib import Path
import pickle
import numpy as np

def load_embedding_dicts(embedding_dir: str) -> Dict[str, Dict[str, List[float]]]:
    """
    Load embedding dictionaries from pickle files.
    
    Args:
        embedding_dir: Directory containing embedding pickle files
        
    Returns:
        Dictionary mapping model names to their embedding dictionaries
    """
    embedding_dicts = {}
    embedding_path = Path(embedding_dir)
    
    if not embedding_path.exists():
        raise FileNotFoundError(f"Embedding directory not found: {embedding_dir}")
    
    pkl_files = list(embedding_path.glob("*.pkl"))
    if not pkl_files:
        raise FileNotFoundError(f"No pickle files found in {embedding_dir}")
    
    print(f"Loading embeddings from {len(pkl_files)} files...")
    
    for pkl_file in pkl_files:
        model_name = pkl_file.stem  # Remove .pkl extension
        print(f"  Loading {model_name}...")
        
        with open(pkl_file, 'rb') as f:
            embedding_dict = pickle.load(f)
        
        # Convert numpy arrays to lists if needed
        if embedding_dict and isinstance(next(iter(embedding_dict.values())), np.ndarray):
            embedding_dict = {token: emb.tolist() if isinstance(emb, np.ndarray) else emb 
                            for token, emb in embedding_dict.items()}
        
        embedding_dicts[model_name] = embedding_dict
        print(f"    Loaded {len(embedding_dict)} token embeddings")
    
    return embedding_dicts