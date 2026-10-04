"""
Evaluation metrics for RAG systems.
"""

import re
from typing import Dict, List, Any
from collections import Counter

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction


def normalize_answer(text: str) -> str:
    """Normalize answer text for evaluation."""
    text = text.lower()
    # Remove punctuation
    text = re.sub(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']', ' ', text)
    # Remove articles
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    # Normalize whitespace
    text = ' '.join(text.split())
    return text


def compute_qa_metrics(prediction: str, reference: str) -> Dict[str, float]:
    """
    Compute QA evaluation metrics.
    
    Args:
        prediction: Model prediction
        reference: Ground truth answer
        
    Returns:
        Dictionary of metric scores
    """
    pred_normalized = normalize_answer(str(prediction))
    ref_normalized = normalize_answer(str(reference))
    
    pred_tokens = pred_normalized.split()
    ref_tokens = ref_normalized.split()
    
    # Exact Match
    em = int(pred_normalized == ref_normalized)
    
    # F1 Score
    common = Counter(pred_tokens) & Counter(ref_tokens)
    num_same = sum(common.values())
    
    if num_same == 0:
        f1 = precision = recall = 0.0
    else:
        precision = num_same / len(pred_tokens) if pred_tokens else 0
        recall = num_same / len(ref_tokens) if ref_tokens else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    
    # BLEU Scores
    smooth = SmoothingFunction()
    bleu1 = sentence_bleu([ref_tokens], pred_tokens, weights=(1, 0, 0, 0), 
                          smoothing_function=smooth.method1)
    bleu2 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.5, 0.5, 0, 0),
                          smoothing_function=smooth.method1)
    bleu3 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.33, 0.33, 0.33, 0),
                          smoothing_function=smooth.method1)
    bleu4 = sentence_bleu([ref_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25),
                          smoothing_function=smooth.method1)
    
    return {
        'em': em,
        'f1': f1,
        'precision': precision,
        'recall': recall,
        'bleu1': bleu1,
        'bleu2': bleu2,
        'bleu3': bleu3,
        'bleu4': bleu4
    }


def compute_retrieval_metrics(
    retrieved_ids: List[str],
    relevant_ids: List[str],
    k_values: List[int] = [1, 5, 10, 20]
) -> Dict[str, float]:
    """
    Compute retrieval evaluation metrics.
    
    Args:
        retrieved_ids: List of retrieved document IDs
        relevant_ids: List of relevant document IDs
        k_values: Values of k for Recall@k and MRR@k
        
    Returns:
        Dictionary of metric scores
    """
    relevant_set = set(relevant_ids)
    
    metrics = {}
    
    for k in k_values:
        retrieved_at_k = set(retrieved_ids[:k])
        
        # Recall@k
        if relevant_set:
            recall = len(retrieved_at_k & relevant_set) / len(relevant_set)
        else:
            recall = 0.0
        metrics[f'recall@{k}'] = recall
        
        # Precision@k
        precision = len(retrieved_at_k & relevant_set) / k
        metrics[f'precision@{k}'] = precision
    
    # MRR (Mean Reciprocal Rank)
    mrr = 0.0
    for i, doc_id in enumerate(retrieved_ids):
        if doc_id in relevant_set:
            mrr = 1.0 / (i + 1)
            break
    metrics['mrr'] = mrr
    
    return metrics


def compute_forgetting(metric_matrix, metric_name: str = 'f1') -> float:
    """
    Compute forgetting measure for continual learning.
    
    Args:
        metric_matrix: Task x Task metric matrix
        metric_name: Name of the metric
        
    Returns:
        Average forgetting across tasks
    """
    n_tasks = metric_matrix.shape[0]
    
    if n_tasks < 2:
        return 0.0
    
    # Forgetting = max performance - final performance
    max_perf = metric_matrix.max(axis=0)
    final_perf = metric_matrix[-1, :]
    
    forgetting = (max_perf - final_perf)[:-1]  # Exclude last task
    
    return forgetting.mean()
