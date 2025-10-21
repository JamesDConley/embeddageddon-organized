"""
Dataset utilities for MatFormer training.

This module provides dataset classes and utilities for loading and processing
text data for MatFormer model training. It includes functionality for loading
parquet files, tokenizing text sequences, and preparing data for training.

Classes:
    FixedDataset: PyTorch Dataset for loading tokenized text from parquet files.
"""

import pandas as pd
from torch.utils.data import Dataset


class FixedDataset(Dataset):
    """
    Dataset class for loading and tokenizing text data from parquet files.
    
    This class provides a PyTorch Dataset implementation that loads text data
    from parquet files, tokenizes it using a provided tokenizer, and prepares
    it for language model training. It handles padding, truncation, and
    returns properly formatted tensors for training.
    
    Attributes:
        df (pandas.DataFrame): Loaded dataframe containing text data.
        tokenizer: HuggingFace tokenizer for text processing.
        max_length (int): Maximum sequence length for tokenization.
    
    Examples:
        >>> from transformers import AutoTokenizer
        >>> tokenizer = AutoTokenizer.from_pretrained("NousResearch/Llama-3.2-1B")
        >>> dataset = FixedDataset("train.parquet", tokenizer, max_length=512)
        >>> sample = dataset[0]
        >>> print(sample.keys())  # dict_keys(['input_ids', 'attention_mask', 'labels'])
    """
    
    def __init__(self, parquet_path, tokenizer, max_length=512):
        """
        Initialize the FixedDataset with parquet file and tokenizer.
        
        Args:
            parquet_path (str): Path to the parquet file containing text data.
            tokenizer: HuggingFace tokenizer for text tokenization.
            max_length (int, optional): Maximum sequence length for tokenization.
                Defaults to 512.
                
        Raises:
            FileNotFoundError: If the parquet file does not exist.
            KeyError: If the parquet file does not contain a 'text' column.
        """
        self.df = pd.read_parquet(parquet_path)
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        print(f"Loaded dataset with {len(self.df)} samples")
        print(f"Sample text lengths - Mean: {self.df['text'].str.len().mean():.1f}, "
              f"Max: {self.df['text'].str.len().max()}")
    
    def __len__(self):
        """
        Return the number of samples in the dataset.
        
        Returns:
            int: Number of text samples in the dataset.
        """
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Get a tokenized sample from the dataset.
        
        This method retrieves a text sample by index, tokenizes it using the
        provided tokenizer, and returns the tokenized representation suitable
        for language model training. The labels are set to be identical to
        the input_ids for causal language modeling.
        
        Args:
            idx (int): Index of the sample to retrieve.
            
        Returns:
            dict: Dictionary containing:
                - input_ids (torch.Tensor): Tokenized input sequence.
                - attention_mask (torch.Tensor): Attention mask for the sequence.
                - labels (torch.Tensor): Target labels for training (same as input_ids).
                
        Raises:
            IndexError: If idx is out of bounds.
        """
        text = self.df.iloc[idx]['text']
        
        # Tokenize the text
        encoded = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        return {
            'input_ids': encoded['input_ids'].squeeze(),
            'attention_mask': encoded['attention_mask'].squeeze(),
            'labels': encoded['input_ids'].squeeze().clone()
        }
