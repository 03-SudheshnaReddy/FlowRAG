#!/usr/bin/env python
"""
FlowRAG - Continual Learning for Retrieval-Augmented Generation

Main entry point for training and evaluation.

Usage:
    # Using command line arguments
    python run.py --datasets nq covidqa convqa --cl_method fp --retriever contriever
    
    # Using config file
    python run.py --config configs/online_flowrag.yaml
"""

import os
import argparse
from loguru import logger

from src.config import FlowRAGConfig
from src.training import FlowRAGTrainer


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="FlowRAG: Continual Learning for RAG",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Config file (highest priority)
    parser.add_argument('--config', type=str, default=None,
                        help='Path to YAML config file')
    
    # Dataset options
    parser.add_argument('--datasets', nargs='+', 
                        default=['nq', 'covidqa', 'convqa', 'newnewsqa'],
                        help='List of datasets for continual learning')
    parser.add_argument('--data_dir', type=str, default='./cl_datasets',
                        help='Directory containing dataset files')
    parser.add_argument('--output_dir', type=str, default='./output',
                        help='Directory for output files')
    parser.add_argument('--exp_name', type=str, default=None,
                        help='Experiment name for output subfolder (e.g., flowrag_nq_covidqa)')
    
    # Model options
    parser.add_argument('--retriever', type=str, default='contriever',
                        choices=['contriever', 'e5', 'bge', 'gte', 'dragon'],
                        help='Retriever model name')
    parser.add_argument('--generator', type=str, 
                        default='Qwen/Qwen2.5-7B-Instruct',
                        help='Generator model name or path')
    parser.add_argument('--max_new_tokens', type=int, default=64,
                        help='Maximum tokens to generate')
    parser.add_argument('--load_in_4bit', action='store_true', default=False,
                        help='Load the generator with bitsandbytes 4-bit NF4 quantization')
    
    # Training options
    parser.add_argument('--cl_method', type=str, default='fp',
                        choices=['offline', 'fp', 'replug', 'emdr', 'fid', 'l2r'],
                        help='Continual learning method')
    parser.add_argument('--lr', type=float, default=5e-3,
                        help='Learning rate (5e-3 for FlowRAG, 1e-5 for baselines)')
    parser.add_argument('--warmup_ratio', type=float, default=0.1,
                        help='Warm-up ratio for learning rate scheduler')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size')
    parser.add_argument('--max_steps', type=int, default=5000,
                        help='Maximum training steps per task')
    parser.add_argument('--eval_interval', type=int, default=500,
                        help='Steps between evaluations')
    parser.add_argument('--samples_per_task', type=int, default=1000,
                        help='Number of QA pairs sampled per task')
    
    # FlowRAG options (paper settings)
    parser.add_argument('--prompt_len', type=int, default=150,
                        help='Length of learnable prompts (paper: 150)')
    parser.add_argument('--prompt_layer', type=int, default=7,
                        help='Number of encoder layers with prompts (paper: 1-7)')
    parser.add_argument('--use_ilf', action='store_true', default=True,
                        help='Use Inner Layer Fusion')
    parser.add_argument('--use_cef', action='store_true', default=True,
                        help='Use Cross Embedder Fusion')
    parser.add_argument('--use_ggf', action='store_true', default=False,
                        help='Use Generator-Guided Fusion')
    parser.add_argument('--temperature', type=float, default=0.1,
                        help='KL temperature for retrieval-likelihood alignment (paper: 0.1)')
    parser.add_argument('--kl_temp_state', type=float, default=1.0,
                        help='KL temperature for aggregated-state alignment (paper: 1.0)')
    parser.add_argument('--beta', type=float, default=0.6,
                        help='Beta weight in Eq.(10) (paper: 0.6)')
    
    # L2R specific options
    parser.add_argument('--memory_size', type=int, default=500,
                        help='Replay memory size for L2R method')
    
    # Retrieval options
    parser.add_argument('--top_k', type=int, default=5,
                        help='Number of documents to retrieve')
    parser.add_argument('--index_type', type=str, default='FLAT',
                        choices=['FLAT', 'HNSW'],
                        help='Faiss index type')
    parser.add_argument('--separate_index', action='store_true', default=False,
                        help='Use separate index per dataset')
    parser.add_argument('--faiss_db', type=str, default='./faiss_db',
                        help='Faiss index directory')
    
    # Runtime options
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


def args_to_config(args) -> FlowRAGConfig:
    """Convert command line args to FlowRAGConfig."""
    from src.config import ModelConfig, TrainingConfig, RetrievalConfig, DataConfig
    
    return FlowRAGConfig(
        model=ModelConfig(
            retriever_name=args.retriever,
            generator_name=args.generator,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            load_in_4bit=args.load_in_4bit
        ),
        training=TrainingConfig(
            learning_rate=args.lr,
            batch_size=args.batch_size,
            max_steps=args.max_steps,
            eval_interval=args.eval_interval,
            cl_method=args.cl_method,
            prompt_len=args.prompt_len,
            prompt_layer=args.prompt_layer,
            use_ilf=args.use_ilf,
            use_cef=args.use_cef,
            use_ggf=args.use_ggf,
            ggf_temperature=args.kl_temp_state,
            beta=args.beta
        ),
        retrieval=RetrievalConfig(
            top_k=args.top_k,
            index_type=args.index_type,
            use_separate_index=args.separate_index,
            faiss_db_path=args.faiss_db
        ),
        data=DataConfig(
            datasets=args.datasets,
            data_dir=args.data_dir,
            output_dir=os.path.join(args.output_dir, args.exp_name) if args.exp_name else args.output_dir
        ),
        gpu=args.gpu,
        seed=args.seed
    )


def main():
    """Main entry point."""
    args = parse_args()
    
    # Set GPU
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    # Load config
    if args.config:
        logger.info(f"Loading config from: {args.config}")
        config = FlowRAGConfig.from_yaml(args.config)
    else:
        config = args_to_config(args)
    
    logger.info(f"Configuration: {config}")
    
    # Create trainer and run
    trainer = FlowRAGTrainer(config)
    trainer.run()
    
    logger.info("Completed!")


if __name__ == "__main__":
    main()
