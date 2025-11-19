"""
Data class for loading and managing training run data.
"""

import json
import os
import torch
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Union

import pandas as pd
from transformers import AutoTokenizer, AutoConfig

# Import model classes at module level
from llm.modified_llama import ModifiedLlamaForCausalLM as BaseMatformer
from llm.weight_based_matformer import ModifiedLlamaForCausalLM as WeightBasedMatformer
from llm.frozen_matformer import ModifiedLlamaForCausalLM as FrozenMatformer


@dataclass
class TrainingRunData:
    """
    Data class that loads all data from a training run directory.
    
    This class loads training/evaluation metrics and configuration,
    while storing paths to model files for optional lazy loading.
    
    Attributes:
        run_dir: Path to the training run directory
        args: Dictionary of training arguments/configuration
        train_df: DataFrame containing training metrics (epoch, step, losses, etc.)
        eval_df: DataFrame containing evaluation metrics
        checkpoint_paths: List of paths to checkpoint files
        final_model_path: Path to the final trained model directory
        run_name: Name of the training run (derived from directory name)
    """
    
    run_dir: str
    args: Dict[str, Any] = field(default_factory=dict)
    train_df: Optional[pd.DataFrame] = None
    eval_df: Optional[pd.DataFrame] = None
    checkpoint_paths: List[str] = field(default_factory=list)
    final_model_path: Optional[str] = None
    run_name: str = ""
    
    def __post_init__(self):
        """Load all data after initialization."""
        self.run_dir = str(Path(self.run_dir).resolve())
        self.run_name = Path(self.run_dir).name
        
        # Load configuration
        self._load_args()
        
        # Load metrics dataframes
        self._load_train_metrics()
        self._load_eval_metrics()
        
        # Find model paths
        self._find_checkpoint_paths()
        self._find_final_model_path()
    
    def _load_args(self):
        """Load training arguments from args_used.json."""
        args_path = Path(self.run_dir) / "args_used.json"
        
        if args_path.exists():
            with open(args_path, 'r') as f:
                self.args = json.load(f)
        else:
            print(f"Warning: args_used.json not found in {self.run_dir}")
            self.args = {}
    
    def _load_train_metrics(self):
        """Load training metrics from metrics/train_loss.csv."""
        train_path = Path(self.run_dir) / "metrics" / "train_loss.csv"
        
        if train_path.exists():
            self.train_df = pd.read_csv(train_path)
            # Remove carriage returns if present
            if self.train_df is not None:
                self.train_df.columns = self.train_df.columns.str.strip()
        else:
            print(f"Warning: train_loss.csv not found in {self.run_dir}/metrics")
            self.train_df = None
    
    def _load_eval_metrics(self):
        """Load evaluation metrics from metrics/eval_loss.csv."""
        eval_path = Path(self.run_dir) / "metrics" / "eval_loss.csv"
        
        if eval_path.exists():
            self.eval_df = pd.read_csv(eval_path)
            # Remove carriage returns if present
            if self.eval_df is not None:
                self.eval_df.columns = self.eval_df.columns.str.strip()
        else:
            print(f"Warning: eval_loss.csv not found in {self.run_dir}/metrics")
            self.eval_df = None
    
    def _find_checkpoint_paths(self):
        """Find all checkpoint files in the checkpoints directory."""
        checkpoints_dir = Path(self.run_dir) / "checkpoints"
        
        if checkpoints_dir.exists() and checkpoints_dir.is_dir():
            # Find all .pt files
            checkpoint_files = sorted(checkpoints_dir.glob("*.pt"))
            self.checkpoint_paths = [str(f) for f in checkpoint_files]
        else:
            self.checkpoint_paths = []
    
    def _find_final_model_path(self):
        """Find the final model directory."""
        final_model_dir = Path(self.run_dir) / "final_model"
        
        if final_model_dir.exists() and final_model_dir.is_dir():
            self.final_model_path = str(final_model_dir)
        else:
            self.final_model_path = None
    
    def get_train_summary(self) -> Dict[str, Any]:
        """
        Get a summary of training metrics.
        
        Returns:
            Dictionary with summary statistics (min/max/final losses, etc.)
        """
        if self.train_df is None or self.train_df.empty:
            return {}
        
        summary = {
            "total_steps": len(self.train_df),
            "total_epochs": self.train_df["epoch"].max() + 1 if "epoch" in self.train_df.columns else None,
            "final_step": self.train_df["step"].iloc[-1] if "step" in self.train_df.columns else None,
            "final_total_loss": self.train_df["total_loss"].iloc[-1] if "total_loss" in self.train_df.columns else None,
            "min_total_loss": self.train_df["total_loss"].min() if "total_loss" in self.train_df.columns else None,
            "final_main_loss": self.train_df["main_loss"].iloc[-1] if "main_loss" in self.train_df.columns else None,
            "min_main_loss": self.train_df["main_loss"].min() if "main_loss" in self.train_df.columns else None,
        }
        
        return summary
    
    def get_eval_summary(self) -> Dict[str, Any]:
        """
        Get a summary of evaluation metrics.
        
        Returns:
            Dictionary with evaluation summary statistics
        """
        if self.eval_df is None or self.eval_df.empty:
            return {}
        
        summary = {
            "num_evaluations": len(self.eval_df),
            "final_eval_loss": self.eval_df["total_loss"].iloc[-1] if "total_loss" in self.eval_df.columns else None,
            "min_eval_loss": self.eval_df["total_loss"].min() if "total_loss" in self.eval_df.columns else None,
        }
        
        return summary
    
    def get_subnetwork_stats(self) -> Optional[pd.DataFrame]:
        """
        Get statistics grouped by subnetwork type (s, m, l, xl).
        
        Returns:
            DataFrame with statistics per subnetwork, or None if no data
        """
        if self.train_df is None or "current_subnetwork" not in self.train_df.columns:
            return None
        
        # Group by subnetwork and calculate statistics
        stats = self.train_df.groupby("current_subnetwork").agg({
            "total_loss": ["mean", "min", "max", "std"],
            "main_loss": ["mean", "min", "max", "std"],
            "step": "count"
        }).round(4)
        
        # Rename count column
        stats.columns = ['_'.join(col).strip() for col in stats.columns.values]
        stats = stats.rename(columns={"step_count": "num_steps"})
        
        return stats
    
    def load_model(self, device: str = "cuda", model_type: str = "final") -> Any:
        """
        Load a trained model from this training run.
        
        This method loads either the final model or a checkpoint based on the
        training configuration stored in args_used.json. It automatically determines
        the correct model class based on the model_type in the config.
        
        Args:
            device: Device to load the model on (default: "cuda")
            model_type: Type of model to load. Options:
                - "final": Load the final trained model (default)
                - "checkpoint": Load the latest checkpoint
                - "checkpoint_0", "checkpoint_1", etc.: Load specific checkpoint by index
                
        Returns:
            Loaded model instance ready for inference or further training
            
        Raises:
            ValueError: If model path doesn't exist or model_type is invalid
            
        Example:
            >>> run = load_training_run("data/models/my_run")
            >>> model = run.load_model(device="cuda")
            >>> model.eval()
            >>> # Use model for inference
        """
        # Determine which path to load from
        if model_type == "final":
            if not self.final_model_path:
                raise ValueError("Final model not found in training run directory")
            model_path = self.final_model_path
            load_from_checkpoint = False
        elif model_type.startswith("checkpoint"):
            if not self.checkpoint_paths:
                raise ValueError("No checkpoints found in training run directory")
            
            if model_type == "checkpoint":
                # Load latest checkpoint
                checkpoint_path = self.checkpoint_paths[-1]
            else:
                # Load specific checkpoint by index
                try:
                    idx = int(model_type.split("_")[1])
                    checkpoint_path = self.checkpoint_paths[idx]
                except (IndexError, ValueError):
                    raise ValueError(f"Invalid checkpoint specification: {model_type}")
            
            model_path = checkpoint_path
            load_from_checkpoint = True
        else:
            raise ValueError(f"Invalid model_type: {model_type}. Must be 'final', 'checkpoint', or 'checkpoint_N'")
        
        # Get model configuration from args
        if not self.args:
            raise ValueError("No training arguments found. Cannot determine model type.")
        
        config_name = self.args.get("config_name")
        training_model_type = self.args.get("model_type", "matformer")
        
        if not config_name:
            raise ValueError("config_name not found in training arguments")
        
        # Select appropriate model class based on training model type
        if training_model_type == "matformer":
            ModelClass = BaseMatformer
        elif training_model_type == "weight_based_matformer":
            ModelClass = WeightBasedMatformer
        elif training_model_type in ["frozen_matformer", "frozen_cov_matformer"]:
            ModelClass = FrozenMatformer
        else:
            raise ValueError(f"Unknown model_type in training args: {training_model_type}")
        
        print(f"Loading {training_model_type} from {model_path}...")
        
        # Load model
        if load_from_checkpoint:
            # Load from checkpoint file
            config = AutoConfig.from_pretrained(config_name)
            model = ModelClass(config)
            
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Loaded checkpoint from epoch {checkpoint.get('epoch')}, step {checkpoint.get('step')}")
        else:
            # Load from final model directory (HuggingFace format)
            model = ModelClass.from_pretrained(model_path)
        
        model.to(device)
        print(f"Model loaded successfully on {device}")
        
        return model
    
    def load_tokenizer(self) -> AutoTokenizer:
        """
        Load the tokenizer used for this training run.
        
        Returns:
            AutoTokenizer instance
            
        Raises:
            ValueError: If tokenizer cannot be found
            
        Example:
            >>> run = load_training_run("data/models/my_run")
            >>> tokenizer = run.load_tokenizer()
            >>> tokens = tokenizer("Hello world", return_tensors="pt")
        """
        # Try loading from final model directory first
        if self.final_model_path and Path(self.final_model_path).exists():
            print(f"Loading tokenizer from {self.final_model_path}")
            return AutoTokenizer.from_pretrained(self.final_model_path)
        
        # Otherwise load from original config
        config_name = self.args.get("config_name")
        if not config_name:
            raise ValueError("Cannot determine tokenizer: config_name not found in training arguments")
        
        print(f"Loading tokenizer from {config_name}")
        return AutoTokenizer.from_pretrained(config_name)
    
    def __repr__(self) -> str:
        """String representation of the training run."""
        lines = [
            f"TrainingRunData(run_name='{self.run_name}')",
            f"  Run directory: {self.run_dir}",
            f"  Train steps: {len(self.train_df) if self.train_df is not None else 0}",
            f"  Eval points: {len(self.eval_df) if self.eval_df is not None else 0}",
            f"  Checkpoints: {len(self.checkpoint_paths)}",
            f"  Final model: {'Yes' if self.final_model_path else 'No'}",
        ]
        
        return "\n".join(lines)


def load_training_run(run_dir: str) -> TrainingRunData:
    """
    Convenience function to load a training run.
    
    Args:
        run_dir: Path to the training run directory
        
    Returns:
        TrainingRunData object with all loaded data
        
    Example:
        >>> run = load_training_run("data/language_models/embeddageddon/my_run")
        >>> print(run.get_train_summary())
        >>> print(run.train_df.head())
    """
    return TrainingRunData(run_dir=run_dir)