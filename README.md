## FlowRAG: Continual Learning for Retrieval-Augmented Generation

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## Overview

**FlowRAG** is a lightweight continual learning framework for **dynamic retriever adaptation** in Retrieval-Augmented Generation (RAG).  
It targets real-world RAG settings where document corpora evolve over time, requiring the retriever to **adapt to new domains** while **mitigating catastrophic forgetting** on previously learned knowledge.

### Key Features

- 🔄 **Continual Learning**: Seamlessly adapt to new domains without forgetting previous knowledge
- 🎯 **FusionPrompt**: Novel prompt-tuning approach for knowledge fusion
- 🔧 **Flexible Architecture**: Modular design supporting different retrievers and generators

## Highlights
- **Parameter-efficient**: updates only ~**0.64%** of retriever parameters.
- **Robust continual learning**: strong retrieval & QA performance with improved resistance to catastrophic forgetting in non-stationary settings.
- **Validated across domains**: evaluated on **four** QA domains/datasets (e.g., CovidQA / NewsQA / ConvQA / NQ) under sequential training.

## Installation

### Prerequisites

- Python 3.11+
- CUDA 12.1+ (for GPU support)
- Git

```bash
git clone https://github.com/intellistream/FlowRAG.git
cd FlowRAG
conda env create -f environment.yml && conda activate flowrag
pip install -e .
```

## Quick Start

**1. Download Datasets** from [Google Drive](https://drive.google.com/file/d/15YoaPuC-7vUtf6PTZO1hH-ECKmZp45vD/view?usp=sharing) and extract to `cl_datasets/`.

**2. Build Index:**
```bash
python -m src.retrieval.index --dataset nq --retriever contriever
```

**3. Run FlowRAG:**
```bash
python run.py --config configs/online_flowrag.yaml
# Or: python run.py --datasets nq covidqa convqa --cl_method fp --retriever contriever
```

## Models

| Component | Model | HuggingFace Path |
|-----------|-------|------------------|
| Retriever | Contriever | [facebook/contriever](https://huggingface.co/facebook/contriever) |
| Generator | Qwen2.5-7B-Instruct | [Qwen/Qwen2.5-7B-Instruct](https://huggingface.co/Qwen/Qwen2.5-7B-Instruct) |
| Generator | Llama3-ChatQA-2-8B | [nvidia/Llama3-ChatQA-2-8B](https://huggingface.co/nvidia/Llama3-ChatQA-2-8B) |

## Experiments

```bash
# Offline (single dataset)
python run.py --datasets nq --cl_method offline --retriever contriever

# Online Baselines (replug/emdr/fid)
python run.py --datasets nq covidqa convqa newnewsqa --cl_method replug --retriever contriever

# FlowRAG (ours)
python run.py --datasets nq covidqa convqa newnewsqa --cl_method fp --retriever contriever \
    --prompt_len 150 --prompt_layer 7 --use_ilf --use_cef
```

See `configs/` for detailed configuration options.

## Evaluation

```bash
python -m src.evaluation.evaluator --exp_name flowrag_exp1 --num_tasks 4
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- [Hugging Face Transformers](https://github.com/huggingface/transformers)
- [Sentence Transformers](https://www.sbert.net/)
- [Faiss](https://github.com/facebookresearch/faiss)
- [vLLM](https://github.com/vllm-project/vllm)

