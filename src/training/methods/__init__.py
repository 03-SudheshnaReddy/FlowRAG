"""
Continual Learning Methods for RAG.

This module contains different continual learning methods:
- offline: No training, just evaluation
- fp (FlowRAG): FusionPrompt with ILF, CEF, and GGF
- replug: REPLUG distillation
- emdr: EMDR2 training
- fid: Fusion-in-Decoder
- atlas: ATLAS training
- l2r: Learning to Retrieve
"""

from typing import Dict, List, Any


# Available methods
METHODS = {
    'offline': 'Offline evaluation without training',
    'fp': 'FlowRAG (FusionPrompt) - prompt-based continual learning',
    'replug': 'REPLUG-style distillation',
    'emdr': 'EMDR2 training',
    'fid': 'Fusion-in-Decoder',
    'atlas': 'ATLAS training',
    'l2r': 'Learning to Retrieve'
}


def get_method_info(method: str) -> str:
    """Get description of a continual learning method."""
    return METHODS.get(method, f"Unknown method: {method}")
