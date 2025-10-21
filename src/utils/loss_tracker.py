#!/usr/bin/env python3

"""
Loss Tracking Module for Embeddageddon Training

This module provides functionality to track and log training losses to CSV files
with per-batch granularity. It's designed to be easily integrated into existing
training loops with minimal code changes.
"""

import csv
import os
import time
from pathlib import Path
from typing import Optional, Dict, Any
import threading


class LossTracker:
    """
    A thread-safe CSV loss tracker that logs training metrics per batch.
    
    Features:
    - Per-batch loss logging
    - Automatic CSV file creation and management
    - Thread-safe writing for concurrent access
    - Configurable output directory and filename
    - Automatic timestamping
    - Epoch and batch tracking
    """
    
    def __init__(self, 
                 output_dir: str = "training_logs",
                 filename: Optional[str] = None,
                 auto_flush: bool = True):
        """
        Initialize the loss tracker.
        
        Args:
            output_dir: Directory to save CSV files
            filename: Custom filename (if None, auto-generates with timestamp)
            auto_flush: Whether to flush CSV file after each write
        """
        self.output_dir = Path(output_dir)
        self.auto_flush = auto_flush
        self._lock = threading.Lock()
        
        # Create output directory if it doesn't exist
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate filename if not provided
        if filename is None:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            filename = f"training_loss_{timestamp}.csv"
        
        self.csv_path = self.output_dir / filename
        
        # Initialize CSV file with headers
        self._initialize_csv()
        
        print(f"Loss tracking initialized: {self.csv_path}")
    
    def _initialize_csv(self):
        """Initialize the CSV file with headers."""
        headers = [
            'timestamp',
            'epoch',
            'batch',
            'batch_loss',
            'epoch_avg_loss',
            'learning_rate',
            'batch_size',
            'total_batches_processed'
        ]
        
        with open(self.csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(headers)
        
        # Initialize tracking variables
        self.total_batches_processed = 0
        self.current_epoch_losses = []
    
    def log_batch(self, 
                  epoch: int,
                  batch: int,
                  batch_loss: float,
                  learning_rate: Optional[float] = None,
                  batch_size: Optional[int] = None,
                  **kwargs):
        """
        Log a single batch's training metrics.
        
        Args:
            epoch: Current epoch number (1-indexed)
            batch: Current batch number within epoch (1-indexed)
            batch_loss: Loss value for this batch
            learning_rate: Current learning rate (optional)
            batch_size: Batch size (optional)
            **kwargs: Additional metrics to log (currently ignored)
        """
        with self._lock:
            # Update tracking
            self.total_batches_processed += 1
            self.current_epoch_losses.append(batch_loss)
            
            # Calculate running average for current epoch
            epoch_avg_loss = sum(self.current_epoch_losses) / len(self.current_epoch_losses)
            
            # Prepare row data
            timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
            row_data = [
                timestamp,
                epoch,
                batch,
                f"{batch_loss:.8f}",  # High precision for loss
                f"{epoch_avg_loss:.8f}",
                f"{learning_rate:.8f}" if learning_rate is not None else "",
                batch_size if batch_size is not None else "",
                self.total_batches_processed
            ]
            
            # Write to CSV
            with open(self.csv_path, 'a', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(row_data)
                
                if self.auto_flush:
                    csvfile.flush()
    
    def start_epoch(self, epoch: int):
        """
        Signal the start of a new epoch.
        
        Args:
            epoch: Epoch number (1-indexed)
        """
        with self._lock:
            # Reset epoch-specific tracking
            self.current_epoch_losses = []
            print(f"Loss tracker: Starting epoch {epoch}")
    
    def end_epoch(self, epoch: int) -> float:
        """
        Signal the end of an epoch and return the average loss.
        
        Args:
            epoch: Epoch number (1-indexed)
            
        Returns:
            Average loss for the completed epoch
        """
        with self._lock:
            if self.current_epoch_losses:
                avg_loss = sum(self.current_epoch_losses) / len(self.current_epoch_losses)
                print(f"Loss tracker: Epoch {epoch} completed, average loss: {avg_loss:.6f}")
                return avg_loss
            else:
                print(f"Loss tracker: Epoch {epoch} completed with no recorded losses")
                return 0.0
    
    def get_csv_path(self) -> str:
        """
        Get the path to the CSV file.
        
        Returns:
            String path to the CSV file
        """
        return str(self.csv_path)
    
    def get_stats(self) -> Dict[str, Any]:
        """
        Get current tracking statistics.
        
        Returns:
            Dictionary with current statistics
        """
        with self._lock:
            return {
                'csv_path': str(self.csv_path),
                'total_batches_processed': self.total_batches_processed,
                'current_epoch_batches': len(self.current_epoch_losses),
                'current_epoch_avg_loss': (
                    sum(self.current_epoch_losses) / len(self.current_epoch_losses)
                    if self.current_epoch_losses else 0.0
                )
            }


def create_loss_tracker(output_dir: str = "training_logs",
                       filename: Optional[str] = None,
                       auto_flush: bool = True) -> LossTracker:
    """
    Convenience function to create a loss tracker.
    
    Args:
        output_dir: Directory to save CSV files
        filename: Custom filename (if None, auto-generates with timestamp)
        auto_flush: Whether to flush CSV file after each write
        
    Returns:
        Configured LossTracker instance
    """
    return LossTracker(output_dir=output_dir, filename=filename, auto_flush=auto_flush)


# Example usage
if __name__ == "__main__":
    # Demo usage
    tracker = create_loss_tracker()
    
    # Simulate training
    for epoch in range(1, 4):
        tracker.start_epoch(epoch)
        
        for batch in range(1, 6):
            # Simulate decreasing loss
            loss = 1.0 / (epoch * batch)
            tracker.log_batch(
                epoch=epoch,
                batch=batch,
                batch_loss=loss,
                learning_rate=0.001,
                batch_size=32
            )
        
        avg_loss = tracker.end_epoch(epoch)
    
    print(f"Demo completed. Check: {tracker.get_csv_path()}")
    print(f"Stats: {tracker.get_stats()}")
