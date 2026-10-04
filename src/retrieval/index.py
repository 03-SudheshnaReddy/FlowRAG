"""
Faiss index management for efficient similarity search.
"""

import os
import pickle
from typing import List, Optional, Tuple, Dict, Any
from dataclasses import dataclass

import numpy as np
import faiss
from loguru import logger


@dataclass
class Document:
    """Simple document class for retrieval results."""
    content: str
    score: float = 0.0
    id: str = ""


class FaissIndex:
    """
    Faiss index for efficient similarity search.
    
    Supports loading pre-built indices and performing batch searches.
    
    Args:
        index_dir: Directory containing Faiss index files
    """
    
    def __init__(self, index_dir: str = "./faiss_db"):
        self.index_dir = index_dir
        os.makedirs(index_dir, exist_ok=True)
        
        self._index: Optional[faiss.Index] = None
        self._documents: List[Document] = []
    
    def load(self, index_name: str) -> bool:
        """
        Load a Faiss index and associated documents.
        
        Args:
            index_name: Name of the index (without extension)
            
        Returns:
            True if loaded successfully, False otherwise
        """
        index_path = os.path.join(self.index_dir, f"{index_name}.index")
        docs_path = os.path.join(self.index_dir, f"{index_name}_docs.pkl")
        
        if not os.path.exists(index_path) or not os.path.exists(docs_path):
            logger.warning(f"Index files not found: {index_name}")
            return False
        
        # Load Faiss index
        self._index = faiss.read_index(index_path)
        
        # Load documents
        with open(docs_path, 'rb') as f:
            self._documents = pickle.load(f)
        
        logger.info(f"Loaded index: {index_name} ({self._index.ntotal} vectors)")
        return True
    
    def save(self, index_name: str) -> None:
        """Save the current index and documents."""
        if self._index is None:
            raise ValueError("No index to save")
        
        index_path = os.path.join(self.index_dir, f"{index_name}.index")
        docs_path = os.path.join(self.index_dir, f"{index_name}_docs.pkl")
        
        faiss.write_index(self._index, index_path)
        
        with open(docs_path, 'wb') as f:
            pickle.dump(self._documents, f)
        
        logger.info(f"Saved index: {index_name}")
    
    def build(
        self,
        embeddings: np.ndarray,
        documents: List[str],
        index_type: str = "FLAT"
    ) -> None:
        """
        Build a new Faiss index.
        
        Args:
            embeddings: Document embeddings [num_docs, dim]
            documents: List of document contents
            index_type: Type of index ("FLAT" or "HNSW")
        """
        dim = embeddings.shape[1]
        
        if index_type == "FLAT":
            self._index = faiss.IndexFlatL2(dim)
        elif index_type == "HNSW":
            self._index = faiss.IndexHNSWFlat(dim, 32)
        else:
            raise ValueError(f"Unknown index type: {index_type}")
        
        # Add embeddings
        self._index.add(embeddings.astype(np.float32))
        
        # Store documents
        self._documents = [
            Document(content=doc, id=f"doc_{i}")
            for i, doc in enumerate(documents)
        ]
        
        logger.info(f"Built {index_type} index with {len(documents)} documents")
    
    def search(
        self,
        query_embedding: np.ndarray,
        top_k: int = 5
    ) -> List[Document]:
        """
        Search for similar documents.
        
        Args:
            query_embedding: Query vector [dim] or [1, dim]
            top_k: Number of results to return
            
        Returns:
            List of Document objects with scores
        """
        if self._index is None:
            raise ValueError("Index not loaded")
        
        if query_embedding.ndim == 1:
            query_embedding = query_embedding.reshape(1, -1)
        
        scores, indices = self._index.search(query_embedding.astype(np.float32), top_k)
        
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx != -1 and idx < len(self._documents):
                doc = self._documents[idx]
                results.append(Document(
                    content=doc.content,
                    score=float(score),
                    id=doc.id
                ))
        
        return results
    
    def batch_search(
        self,
        query_embeddings: np.ndarray,
        top_k: int = 5
    ) -> List[List[Document]]:
        """
        Batch search for similar documents.
        
        Args:
            query_embeddings: Query vectors [num_queries, dim]
            top_k: Number of results per query
            
        Returns:
            List of Document lists for each query
        """
        if self._index is None:
            raise ValueError("Index not loaded")
        
        if query_embeddings.ndim == 1:
            query_embeddings = query_embeddings.reshape(1, -1)
        
        scores, indices = self._index.search(query_embeddings.astype(np.float32), top_k)
        
        batch_results = []
        for query_scores, query_indices in zip(scores, indices):
            results = []
            for score, idx in zip(query_scores, query_indices):
                if idx != -1 and idx < len(self._documents):
                    doc = self._documents[idx]
                    results.append(Document(
                        content=doc.content,
                        score=float(score),
                        id=doc.id
                    ))
            batch_results.append(results)
        
        return batch_results
    
    @property
    def num_vectors(self) -> int:
        """Number of vectors in the index."""
        return self._index.ntotal if self._index else 0
    
    @property
    def dimension(self) -> int:
        """Dimension of vectors."""
        return self._index.d if self._index else 0


class MultiDatasetIndex:
    """
    Manages indices for multiple datasets.
    
    Supports both merged and separate index modes for continual learning.
    """
    
    def __init__(
        self,
        index_dir: str = "./faiss_db",
        model_abbr: str = "ctr"
    ):
        self.index_dir = index_dir
        self.model_abbr = model_abbr
        self._cache: Dict[str, FaissIndex] = {}
    
    def get_index(
        self,
        datasets: List[str],
        separate: bool = False
    ) -> FaissIndex:
        """
        Get a Faiss index for the specified datasets.
        
        Args:
            datasets: List of dataset names
            separate: If True, only use the last dataset's index
            
        Returns:
            FaissIndex containing the combined or separate index
        """
        if separate and datasets:
            # Use only the last (current) dataset
            datasets = [datasets[-1]]
        
        cache_key = tuple(sorted(datasets))
        
        if cache_key in self._cache:
            return self._cache[cache_key]
        
        # Load and merge indices
        combined = self._merge_indices(datasets)
        self._cache[cache_key] = combined
        
        return combined
    
    def _merge_indices(self, datasets: List[str]) -> FaissIndex:
        """Merge multiple dataset indices into one."""
        combined = FaissIndex(self.index_dir)
        
        all_embeddings = []
        all_documents = []
        dim = None
        
        for dataset_name in datasets:
            index_name = f"{self.model_abbr}_{dataset_name}_flat"
            temp_index = FaissIndex(self.index_dir)
            
            if not temp_index.load(index_name):
                logger.warning(f"Could not load index for {dataset_name}")
                continue
            
            if dim is None:
                dim = temp_index.dimension
            
            # Extract embeddings from index
            embeddings = np.zeros(
                (temp_index.num_vectors, dim),
                dtype=np.float32
            )
            for i in range(temp_index.num_vectors):
                embeddings[i] = temp_index._index.reconstruct(i)
            
            all_embeddings.append(embeddings)
            all_documents.extend([d.content for d in temp_index._documents])
        
        if all_embeddings:
            merged_embeddings = np.vstack(all_embeddings)
            combined.build(merged_embeddings, all_documents, "FLAT")
            logger.info(f"Merged {len(datasets)} indices: {combined.num_vectors} vectors")
        
        return combined


def build_index_from_dataset(
    dataset_name: str,
    retriever: str = "contriever",
    index_type: str = "FLAT",
    data_dir: str = "./cl_datasets",
    index_dir: str = "./faiss_db"
) -> None:
    """
    Build a Faiss index from a dataset.
    
    Args:
        dataset_name: Name of the dataset (e.g., 'nq', 'covidqa')
        retriever: Name of the retriever model
        index_type: Type of Faiss index ('FLAT' or 'HNSW')
        data_dir: Directory containing datasets
        index_dir: Directory to save the index
    """
    import json
    from sentence_transformers import SentenceTransformer
    from src.config import RETRIEVER_MODELS
    
    # Retriever abbreviation mapping
    RETRIEVER_ABBR = {
        "contriever": "ctr",
        "e5": "e5",
        "bge": "bge",
        "gte": "gte",
        "dragon": "dragon"
    }
    
    # Get model name and abbreviation
    model_info = RETRIEVER_MODELS.get(retriever)
    if model_info is None:
        raise ValueError(f"Unknown retriever: {retriever}. Available: {list(RETRIEVER_MODELS.keys())}")
    
    model_name = model_info['passage']  # Use passage encoder for indexing
    model_abbr = RETRIEVER_ABBR.get(retriever, retriever[:3])
    
    logger.info(f"Building index for {dataset_name} with {retriever} ({model_name})")
    
    # Load dataset
    train_path = os.path.join(data_dir, dataset_name, 'train.json')
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Dataset not found: {train_path}")
    
    with open(train_path, 'r') as f:
        data = json.load(f)
    
    # Extract contexts/passages based on data format
    contexts = []
    
    # Format 1: Dict with 'question', 'answer', 'context' as parallel lists
    if isinstance(data, dict) and 'context' in data:
        if isinstance(data['context'], list):
            contexts = [ctx for ctx in data['context'] if ctx]
    
    # Format 2: List of items with 'context' field
    elif isinstance(data, list):
        for item in data:
            if 'context' in item and item['context']:
                contexts.append(item['context'])
            elif 'ctxs' in item and item['ctxs']:
                for ctx in item['ctxs']:
                    if isinstance(ctx, dict) and 'text' in ctx:
                        contexts.append(ctx['text'])
                    elif isinstance(ctx, str):
                        contexts.append(ctx)
    
    # Format 3: Dict with 'data' key containing list
    elif isinstance(data, dict) and 'data' in data:
        for item in data['data']:
            if 'context' in item and item['context']:
                contexts.append(item['context'])
    
    # Remove duplicates while preserving order
    seen = set()
    unique_contexts = []
    for ctx in contexts:
        if ctx not in seen:
            seen.add(ctx)
            unique_contexts.append(ctx)
    contexts = unique_contexts
    
    if not contexts:
        raise ValueError(f"No contexts found in {dataset_name}")
    
    logger.info(f"Found {len(contexts)} unique contexts")
    
    # Load embedding model
    logger.info(f"Loading embedding model: {model_name}")
    model = SentenceTransformer(model_name)
    
    # Generate embeddings
    logger.info("Generating embeddings...")
    embeddings = model.encode(
        contexts,
        batch_size=32,
        show_progress_bar=True,
        convert_to_numpy=True
    )
    
    # Build and save index
    index = FaissIndex(index_dir)
    index.build(embeddings, contexts, index_type)
    
    index_name = f"{model_abbr}_{dataset_name}_flat"
    index.save(index_name)
    
    logger.info(f"Index saved: {index_name} ({index.num_vectors} vectors, dim={index.dimension})")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Build Faiss index for a dataset")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name (e.g., nq, covidqa)")
    parser.add_argument("--retriever", type=str, default="contriever", help="Retriever model name")
    parser.add_argument("--index_type", type=str, default="FLAT", choices=["FLAT", "HNSW"], help="Index type")
    parser.add_argument("--data_dir", type=str, default="./cl_datasets", help="Data directory")
    parser.add_argument("--index_dir", type=str, default="./faiss_db", help="Index directory")
    
    args = parser.parse_args()
    
    build_index_from_dataset(
        dataset_name=args.dataset,
        retriever=args.retriever,
        index_type=args.index_type,
        data_dir=args.data_dir,
        index_dir=args.index_dir
    )
