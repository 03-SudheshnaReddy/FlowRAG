"""
Retriever for document retrieval.
"""

from typing import List, Optional, Dict, Any
import numpy as np
import torch
from loguru import logger

from src.retrieval.index import FaissIndex, Document


class Retriever:
    """
    Retriever that combines embedder and index for document retrieval.
    
    Args:
        index: FaissIndex instance
        top_k: Default number of documents to retrieve
    """
    
    def __init__(
        self,
        index: FaissIndex,
        top_k: int = 5
    ):
        self.index = index
        self.top_k = top_k
    
    def retrieve(
        self,
        query_embedding: np.ndarray,
        top_k: Optional[int] = None
    ) -> List[Document]:
        """
        Retrieve documents for a single query.
        
        Args:
            query_embedding: Query vector
            top_k: Number of documents to retrieve
            
        Returns:
            List of Document objects
        """
        k = top_k or self.top_k
        return self.index.search(query_embedding, k)
    
    def batch_retrieve(
        self,
        query_embeddings: np.ndarray,
        top_k: Optional[int] = None
    ) -> List[List[Document]]:
        """
        Retrieve documents for multiple queries.
        
        Args:
            query_embeddings: Query vectors [num_queries, dim]
            top_k: Number of documents per query
            
        Returns:
            List of Document lists
        """
        k = top_k or self.top_k
        return self.index.batch_search(query_embeddings, k)
    
    @property
    def num_documents(self) -> int:
        """Number of documents in the index."""
        return self.index.num_vectors
