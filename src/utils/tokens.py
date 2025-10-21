from typing import Dict, List

def get_common_tokens(embedding_dicts: Dict[str, Dict[str, List[float]]]) -> List[str]:
    """
    Get tokens that appear in at least 2 models.
    
    Args:
        embedding_dicts: Dictionary mapping model names to their embedding dictionaries
        
    Returns:
        List of tokens that appear in at least 2 models
    """
    # Count how many models each token appears in
    token_counts = {}
    for model_dict in embedding_dicts.values():
        for token in model_dict.keys():
            token_counts[token] = token_counts.get(token, 0) + 1
    
    # Filter tokens that appear in at least 2 models
    common_tokens = [token for token, count in token_counts.items() if count >= 2]
    
    return sorted(common_tokens)