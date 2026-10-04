"""
Dataset loading and processing for continual learning RAG.
"""

import os
import json
from typing import Dict, List, Optional, Tuple, Any
import torch
from torch.utils.data import Dataset
from loguru import logger


# Supported datasets and their configurations
DATASET_REGISTRY = {
    'nq': {'type': 'extraction'},
    'hotpotqa': {'type': 'extraction'},
    'quoref': {'type': 'extraction'},
    'qasper': {'type': 'extraction'},
    'ropes': {'type': 'extraction'},
    'openbookqa': {'type': 'multi-choice'},
    'multirc': {'type': 'multi-choice'},
    'convqa': {'type': 'abstraction'},
    'narrativeqa': {'type': 'abstraction'},
    'drop': {'type': 'abstraction'},
    'boolq': {'type': 'boolqa'},
    'np_boolqa': {'type': 'boolqa'},
    'covidqa': {'type': 'abstraction'},
    'newnewsqa': {'type': 'abstraction'},
}


class CLDataset(Dataset):
    """
    Continual Learning Dataset for RAG.
    
    Supports loading multiple QA datasets for continual learning scenarios.
    Each dataset contains questions, answers, and contexts.
    
    Args:
        mode: 'train' or 'test'
        datatype: 'qa' for question-answer pairs, 'context' for passages
        datasets: List of dataset names to load
        data_dir: Directory containing preprocessed datasets
        preload_data: Pre-loaded dataset (to share contexts between qa and context modes)
    """
    
    def __init__(
        self,
        mode: str = 'train',
        datatype: str = 'qa',
        datasets: Optional[List[str]] = None,
        data_dir: str = './cl_datasets',
        preload_data: Optional[List[Dict]] = None
    ):
        self.mode = mode
        self.datatype = datatype
        self.data_dir = data_dir
        
        # Filter valid dataset names
        if datasets is None:
            datasets = list(DATASET_REGISTRY.keys())
        self.dataset_names = [d for d in datasets if d in DATASET_REGISTRY]
        
        # Current dataset state
        self.current_idx = 0
        self.current_data: Dict[str, List] = {}
        
        # Load all datasets
        if preload_data is None:
            self.all_datasets = self._load_all_datasets()
        else:
            self.all_datasets = preload_data
    
    def _load_all_datasets(self) -> List[Dict]:
        """Load all datasets from disk."""
        datasets = []
        for name in self.dataset_names:
            data = self._load_single_dataset(name)
            if data:
                datasets.append(data)
        return datasets
    
    def _load_single_dataset(self, name: str) -> Optional[Dict]:
        """Load a single dataset from JSON file."""
        filepath = os.path.join(self.data_dir, name, f'{self.mode}.json')
        
        if not os.path.exists(filepath):
            logger.warning(f"Dataset file not found: {filepath}")
            return None
        
        logger.info(f"Loading [{name}][{self.mode}] from: {filepath}")
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return data
    
    def load_dataset(self, idx: int) -> None:
        """Load a specific dataset by index for iteration."""
        if idx < 0 or idx >= len(self.all_datasets):
            raise IndexError(f"Dataset index {idx} out of range [0, {len(self.all_datasets)})")
        
        self.current_idx = idx
        self.current_data = self.all_datasets[idx]
        logger.info(f"Loaded dataset {idx}: {self.dataset_names[idx]}")
    
    @property
    def num_datasets(self) -> int:
        """Number of datasets loaded."""
        return len(self.all_datasets)
    
    def __len__(self) -> int:
        if not self.current_data:
            return 0
        
        if self.datatype == 'qa':
            return len(self.current_data.get('question', []))
        else:
            return len(self.current_data.get('context', []))
    
    def __getitem__(self, index: int) -> Dict[str, Any]:
        if self.datatype == 'qa':
            return {
                'index': index,
                'question': self.current_data['question'][index],
                'answers': self.current_data['answer'][index]
            }
        else:
            return {
                'index': index,
                'context': self.current_data['context'][index]
            }
    
    def get_dataset_name(self, idx: Optional[int] = None) -> str:
        """Get the name of a dataset by index."""
        if idx is None:
            idx = self.current_idx
        return self.dataset_names[idx] if idx < len(self.dataset_names) else "unknown"
