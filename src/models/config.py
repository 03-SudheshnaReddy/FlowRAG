"""Model-side configuration objects."""

from dataclasses import dataclass


@dataclass
class PromptConfig:
    """Configuration for layer-wise FlowRAG prompt embeddings."""

    num_tasks: int
    prompt_len: int = 150
    prompt_layer: int = 7
    top_k: int = 5
    batch_size: int = 1
    use_ilf: bool = True
    use_cef: bool = True
    use_ggf: bool = False
