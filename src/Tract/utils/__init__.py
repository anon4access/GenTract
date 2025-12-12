# Expose commonly used utilities.
import yaml
from typing import Union

import wandb

from .torch_utils import setup_torch
from .visualizer  import save_nifti_image, plot_reconstruction
from .streamlines import generate_rotation_matrix_torch, apply_affine_transform_torch

def load_config(config_path):
    """Load configuration from a YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


class AverageLoss:
    """
    Utility class to track losses
    and metrics during training.
    """

    def __init__(self):
        self.losses_accumulator = {}
    
    def put(self, loss_key:str, loss_value:Union[int,float]) -> None:
        """
        Store value

        Args:
            loss_key (str): Metric name
            loss_value (int | float): Metric value to store
        """
        if loss_key not in self.losses_accumulator:
            self.losses_accumulator[loss_key] = []
        self.losses_accumulator[loss_key].append(loss_value)
    
    def pop_avg(self, loss_key:str) -> float:
        """
        Average the stored values of a given metric

        Args:
            loss_key (str): Metric name

        Returns:
            float: average of the stored values
        """
        if loss_key not in self.losses_accumulator:
            return None
        losses = self.losses_accumulator[loss_key]
        self.losses_accumulator[loss_key] = []
        return sum(losses) / len(losses)
    
    def get_avg(self, loss_key:str) -> float:
        """
        Average the stored values of a given metric
        without removing it.

        Args:
            loss_key (str): Metric name

        Returns:
            float: average of the stored values
        """
        if loss_key not in self.losses_accumulator:
            return None
        losses = self.losses_accumulator[loss_key]
        return sum(losses) / len(losses)

    def to_wandb(self, step: int):
        """
        Logs the average value of all the metrics stored 
        into weights & biases.

        Args:
            step (int): Tensorboard logging global step 
        """
        for metric_key in self.losses_accumulator.keys():            
            wandb.log({metric_key: self.pop_avg(metric_key), "iter": step})