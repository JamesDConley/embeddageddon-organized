"""
Data class for loading and managing training run data.
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any

import pandas as pd


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