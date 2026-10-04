"""
Data collators for batching.
"""

from typing import List, Dict, Any
import torch


class QACollator:
    """Collator for question-answer data."""
    
    def __init__(self, tokenizer=None, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, batch: List[Dict[str, Any]]) -> tuple:
        indices = torch.tensor([item['index'] for item in batch])
        questions = [item['question'] for item in batch]
        answers = [item['answers'] for item in batch]
        
        return indices, questions, answers


class ContextCollator:
    """Collator for context/passage data."""
    
    def __init__(self, tokenizer=None, max_length: int = 512):
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __call__(self, batch: List[Dict[str, Any]]) -> tuple:
        indices = torch.tensor([item['index'] for item in batch])
        contexts = [item['context'] for item in batch]
        
        return indices, contexts
