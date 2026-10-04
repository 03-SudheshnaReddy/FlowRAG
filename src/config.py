"""
FlowRAG Configuration
=====================

Centralized configuration for the FlowRAG framework.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import yaml


# Retriever model mappings
RETRIEVER_MODELS = {
    "contriever": {
        "query": "facebook/contriever",
        "passage": "facebook/contriever",
        "dim": 768
    },
    "e5": {
        "query": "intfloat/e5-large-v2",
        "passage": "intfloat/e5-large-v2",
        "dim": 1024
    },
    "bge": {
        "query": "BAAI/bge-large-en-v1.5",
        "passage": "BAAI/bge-large-en-v1.5",
        "dim": 1024
    },
    "gte": {
        "query": "thenlper/gte-base",
        "passage": "thenlper/gte-base",
        "dim": 768
    },
    "dragon": {
        "query": "facebook/dragon-plus-query-encoder",
        "passage": "facebook/dragon-plus-context-encoder",
        "dim": 768
    }
}


@dataclass
class ModelConfig:
    """Configuration for models."""
    
    # Retriever settings
    retriever_name: str = "contriever"
    embedding_dim: int = 768
    
    # Generator settings
    generator_name: str = "Qwen/Qwen2.5-7B-Instruct"
    max_new_tokens: int = 64
    temperature: float = 0.1
    
    @property
    def query_encoder(self) -> str:
        return RETRIEVER_MODELS.get(self.retriever_name, RETRIEVER_MODELS["contriever"])["query"]
    
    @property
    def passage_encoder(self) -> str:
        return RETRIEVER_MODELS.get(self.retriever_name, RETRIEVER_MODELS["contriever"])["passage"]


@dataclass
class TrainingConfig:
    """Configuration for training."""
    
    # Learning rate: if None, auto-select based on cl_method
    # FlowRAG (fp): lr=5e-3, Baselines (replug, emdr, fid, atlas): lr=1e-5
    learning_rate: float = None
    batch_size: int = 1
    max_steps: int = 5000
    warmup_ratio: float = 0.1  # warmup_steps = max_steps * warmup_ratio
    eval_interval: int = 500
    
    # Continual learning method
    cl_method: str = "fp"  # offline, fp, replug, emdr, fid, atlas, l2r
    
    # FlowRAG (FusionPrompt) specific settings
    prompt_len: int = 150
    prompt_layer: int = 7  # Insert prompts into encoder layers 1-7
    use_ilf: bool = True   # Inner Layer Fusion
    use_cef: bool = True   # Cross Embedder Fusion  
    use_ggf: bool = False  # Generator-Guided Fusion
    cef_temperature: float = 0.1  # KL-divergence temperature for retrieval-likelihood alignment
    ggf_temperature: float = 1.0
    beta: float = 0.6  # Eq. 10 weighting for generator-guided state alignment


@dataclass
class RetrievalConfig:
    """Configuration for retrieval."""
    
    top_k: int = 5
    index_type: str = "FLAT"  # FLAT or HNSW
    use_faiss: bool = True
    use_separate_index: bool = False  # Each dataset uses its own index
    chunk_size: int = 128
    chunk_overlap: int = 0
    faiss_db_path: str = "./faiss_db"


@dataclass
class DataConfig:
    """Configuration for datasets."""
    
    datasets: List[str] = field(default_factory=lambda: ["nq", "covidqa", "convqa", "newnewsqa"])
    data_dir: str = "./cl_datasets"
    output_dir: str = "./output"


@dataclass
class FlowRAGConfig:
    """Main configuration class for FlowRAG."""
    
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    data: DataConfig = field(default_factory=DataConfig)
    
    # Runtime settings
    gpu: int = 0
    seed: int = 42
    
    @classmethod
    def from_yaml(cls, path: str) -> "FlowRAGConfig":
        """Load configuration from YAML file."""
        with open(path, 'r') as f:
            config_dict = yaml.safe_load(f)
        return cls.from_dict(config_dict)
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "FlowRAGConfig":
        """Create config from dictionary."""
        return cls(
            model=ModelConfig(**config_dict.get("model", {})),
            training=TrainingConfig(**config_dict.get("training", {})),
            retrieval=RetrievalConfig(**config_dict.get("retrieval", {})),
            data=DataConfig(**config_dict.get("data", {})),
            gpu=config_dict.get("gpu", 0),
            seed=config_dict.get("seed", 42),
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        from dataclasses import asdict
        return asdict(self)
