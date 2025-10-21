"""
Training metrics tracking utilities for MatFormer models.

This module provides a TrainingTracker class for logging training and evaluation
metrics to CSV files during model training.
"""

import os
import csv


class TrainingTracker:
    """
    Tracks training and evaluation metrics for MatFormer models.
    
    This class handles the creation of metrics directories and CSV files for
    logging training progress, including losses for different subnetwork sizes.
    """
    
    def __init__(self, output_dir):
        """
        Initialize the TrainingTracker with output directory.
        
        Args:
            output_dir (str): Directory where metrics will be saved.
        """
        self.output_dir = output_dir
        self.metrics_dir = os.path.join(output_dir, "metrics")
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        train_loss_path = os.path.join(self.metrics_dir, "train_loss.csv")
        eval_loss_path = os.path.join(self.metrics_dir, "eval_loss.csv")

        # Open files and create writers
        self.train_loss_file = open(train_loss_path, 'w', newline='')
        self.eval_loss_file = open(eval_loss_path, 'w', newline='')

        self.train_loss_writer = csv.writer(self.train_loss_file)
        self.eval_loss_writer = csv.writer(self.eval_loss_file)

        self.base_cols = ['epoch', 'step', 'total_loss', 'main_loss', 'covariance_loss', 'current_subnetwork']
        self.write_cols()

    def write_cols(self):
        """Write column headers to both train and eval CSV files."""
        self.train_loss_writer.writerow(self.base_cols)
        self.eval_loss_writer.writerow(self.base_cols)
        self.train_loss_file.flush()
        self.eval_loss_file.flush()

    def write_train(self, epoch, step, total_loss, main_loss, covariance_loss, current_subnetwork):
        """
        Write training metrics to the training CSV file.
        
        Args:
            epoch (int): Current epoch number.
            step (int): Current training step.
            total_loss (float): Total loss value.
            main_loss (float): Main loss component.
            covariance_loss (float): Covariance loss component.
            current_subnetwork (str): Current subnetwork size being trained.
        """
        self.train_loss_writer.writerow([epoch, step, total_loss, main_loss, covariance_loss, current_subnetwork])
        self.train_loss_file.flush()
    
    def write_eval(self, epoch, step, total_loss, main_loss, covariance_loss, current_subnetwork):
        """
        Write evaluation metrics to the evaluation CSV file.
        
        Args:
            epoch (int): Current epoch number.
            step (int): Current training step.
            total_loss (float): Total loss value.
            main_loss (float): Main loss component.
            covariance_loss (float): Covariance loss component.
            current_subnetwork (str): Subnetwork size being evaluated.
        """
        self.eval_loss_writer.writerow([epoch, step, total_loss, main_loss, covariance_loss, current_subnetwork])
        self.eval_loss_file.flush()
    
    def close_files(self):
        """Close all open file handles."""
        self.eval_loss_file.close()
        self.train_loss_file.close()
