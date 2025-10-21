from abc import ABC, abstractmethod
from typing import List, Dict, Optional
import numpy as np

class EmbeddingModel(ABC):
    @abstractmethod
    def get_vocab(self) -> List[str]:
        """Return all tokens in this model's vocabulary."""
        raise NotImplementedError

    @abstractmethod
    def get_embedding(self, token: str) -> Optional[np.ndarray]:
        """Return the embedding for the token if present, else None."""
        raise NotImplementedError

    @abstractmethod
    def get_all_embeddings(self) -> Dict[str, np.ndarray]:
        """Return a dict mapping each token to its embedding."""
        raise NotImplementedError