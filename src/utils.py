"""
Utility functions for FlowRAG.
"""

import random
import numpy as np
import torch
from loguru import logger


def seed_everything(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    

def get_device() -> torch.device:
    """Get the available device (CUDA or CPU)."""
    if torch.cuda.is_available():
        device_count = torch.cuda.device_count()
        device_name = torch.cuda.get_device_name(0)
        logger.info(f"Found {device_count} CUDA device(s). Using: {device_name}")
        return torch.device('cuda')
    else:
        logger.info("CUDA is not available. Using CPU.")
        return torch.device('cpu')


class CosineScheduler:
    """Cosine learning rate scheduler with warmup."""
    
    def __init__(self, optimizer, warmup: int = 100, total: int = 1000, ratio: float = 0.1):
        self.optimizer = optimizer
        self.warmup = warmup
        self.total = total
        self.ratio = ratio
        self.step_count = 0
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        
    def step(self):
        self.step_count += 1
        lr_ratio = self._get_lr_ratio()
        for param_group, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            param_group['lr'] = base_lr * lr_ratio
            
    def _get_lr_ratio(self) -> float:
        if self.step_count < self.warmup:
            return self.step_count / self.warmup
        else:
            progress = (self.step_count - self.warmup) / max(1, self.total - self.warmup)
            return self.ratio + (1 - self.ratio) * 0.5 * (1 + np.cos(np.pi * progress))
