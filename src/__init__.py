"""
FlowRAG: Continual Learning for Retrieval-Augmented Generation
==============================================================

A framework for continual learning in RAG systems, addressing the challenge
of knowledge forgetting when adapting to new domains.

Main Components:
    - data: Dataset loading and processing
    - models: Embedders and generators
    - retrieval: Vector indexing and retrieval
    - training: Continual learning methods
    - evaluation: Metrics and evaluation
"""

__version__ = "1.0.0"
__author__ = "FlowRAG Team"

from src.config import FlowRAGConfig
