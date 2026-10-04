"""
FlowRAG Trainer - Main training and evaluation logic.
"""

import os
import re
import csv
import time
from datetime import datetime
from typing import Dict, List, Optional, Any, Tuple
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from tqdm import tqdm
from jinja2 import Template
from loguru import logger

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

from src.config import FlowRAGConfig
from src.utils import seed_everything, get_device, CosineScheduler
from src.data import CLDataset, QACollator, ContextCollator
from src.models import create_embedder, create_generator, PromptConfig
from src.retrieval import FaissIndex, Retriever
from src.retrieval.index import MultiDatasetIndex


# Prompt templates
LLAMA_TEMPLATE = """<s>[INST]Answer the question based on the provided context in a few words. 
Context:
{% for doc in documents %}
{{ doc.content }}
{% endfor %}

Question:
{{query}}
[/INST]
Assistant:
"""

CHATQA_TEMPLATE = """<|begin_of_text|>System: This is a chat between a user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions based on the context. The assistant should also indicate when the answer cannot be found in the context.
{% for doc in documents %}
{{ doc.content }}
{% endfor %}
User: {{query}}
Answer the user's question in a few words.
Assistant:"""


class FlowRAGTrainer:
    """
    Main trainer for FlowRAG continual learning.
    
    Handles training, evaluation, and result tracking for continual learning
    scenarios in RAG systems.
    
    Args:
        config: FlowRAGConfig instance
    """
    
    METRICS = ['em', 'f1', 'bleu1', 'bleu2', 'bleu3', 'bleu4']
    
    def __init__(self, config: FlowRAGConfig):
        self.config = config
        self.device = get_device()
        
        # Set seed
        seed_everything(config.seed)
        
        # Initialize components
        self._init_datasets()
        self._init_models()
        self._init_metrics()
        
        # Prompt template
        if "nvidia" in config.model.generator_name.lower():
            self.prompt_template = Template(CHATQA_TEMPLATE)
        else:
            self.prompt_template = Template(LLAMA_TEMPLATE)
    
    def _init_datasets(self):
        """Initialize datasets."""
        cfg = self.config.data
        
        # Collators
        self.qa_collator = QACollator()
        self.ctx_collator = ContextCollator()
        
        # Datasets
        self.train_qa = CLDataset(
            mode='train',
            datatype='qa',
            datasets=cfg.datasets,
            data_dir=cfg.data_dir
        )
        self.train_ctx = CLDataset(
            mode='train',
            datatype='context',
            datasets=cfg.datasets,
            data_dir=cfg.data_dir,
            preload_data=self.train_qa.all_datasets
        )
        
        self.test_qa = CLDataset(
            mode='test',
            datatype='qa',
            datasets=cfg.datasets,
            data_dir=cfg.data_dir
        )
        self.test_ctx = CLDataset(
            mode='test',
            datatype='context',
            datasets=cfg.datasets,
            data_dir=cfg.data_dir,
            preload_data=self.test_qa.all_datasets
        )
        
        self.num_tasks = self.train_qa.num_datasets
        logger.info(f"Loaded {self.num_tasks} tasks: {cfg.datasets}")
    
    def _init_models(self):
        """Initialize models."""
        cfg = self.config
        
        # Prompt config for FlowRAG
        prompt_config = None
        if cfg.training.cl_method == 'fp':
            prompt_config = PromptConfig(
                num_tasks=self.num_tasks,
                prompt_len=cfg.training.prompt_len,
                prompt_layer=cfg.training.prompt_layer,
                top_k=cfg.retrieval.top_k,
                batch_size=cfg.training.batch_size,
                use_ilf=cfg.training.use_ilf,
                use_cef=cfg.training.use_cef,
                use_ggf=cfg.training.use_ggf
            )
        
        # Create embedder
        self.embedder = create_embedder(
            retriever_name=cfg.model.retriever_name,
            use_prompt=(cfg.training.cl_method == 'fp'),
            prompt_config=prompt_config,
            device=self.device
        )
        
        # Index manager
        model_abbr = {
            'contriever': 'ctr',
            'e5': 'e5',
            'bge': 'bge',
            'gte': 'gte',
            'dragon': 'drg'
        }.get(cfg.model.retriever_name, 'ctr')
        
        self.index_manager = MultiDatasetIndex(
            index_dir=cfg.retrieval.faiss_db_path,
            model_abbr=model_abbr
        )
        
        # Generator (lazy loaded)
        self.generator = None
    
    def _init_metrics(self):
        """Initialize metric tracking."""
        n = self.num_tasks
        self.metric_matrix = {m: np.zeros((n, n)) for m in self.METRICS}
        self.forgetting = {m: np.zeros(n - 1) for m in self.METRICS}
    
    def _get_generator(self, for_training: bool = False):
        """Get or create generator."""
        if self.generator is None:
            self.generator = create_generator(
                model_name=self.config.model.generator_name,
                max_new_tokens=self.config.model.max_new_tokens,
                use_vllm=not for_training,
                device=self.device,
                load_in_4bit=self.config.model.load_in_4bit
            )
        return self.generator
    
    def _init_optimizer(self, task_id: int, num_steps: int):
        """Initialize optimizer for a task."""
        if self.config.training.cl_method == 'fp':
            self.embedder.set_task(task_id)
        
        params = self.embedder.get_trainable_params()
        
        # Use different learning rates for different methods
        # FlowRAG (fp): lr=5e-3, Baselines (replug, emdr, fid, atlas): lr=1e-5
        if self.config.training.learning_rate is not None:
            lr = self.config.training.learning_rate
        elif self.config.training.cl_method == 'fp':
            lr = 5e-3  # FlowRAG default
        else:
            lr = 1e-5  # Baseline default (fine-tuning)
        
        self.optimizer = torch.optim.AdamW(
            params,
            lr=lr,
            weight_decay=0.0001,
            betas=(0.9, 0.999),
            eps=1e-6
        )
        
        # Calculate warmup steps from ratio
        warmup_steps = int(num_steps * self.config.training.warmup_ratio)
        
        self.scheduler = CosineScheduler(
            self.optimizer,
            warmup=warmup_steps,
            total=num_steps,
            ratio=0.1
        )
        
        logger.info(f"Optimizer initialized: lr={lr}, warmup_steps={warmup_steps}, total_steps={num_steps}")
    
    @staticmethod
    def kl_div_loss(
        scores: torch.Tensor,
        gold_scores: torch.Tensor,
        temperature: float = 0.1
    ) -> torch.Tensor:
        """Compute KL divergence loss."""
        # Ensure tensors are on same device
        gold_scores = gold_scores.to(scores.device)
        gold_probs = F.softmax(gold_scores / temperature, dim=-1)
        log_probs = F.log_softmax(scores / temperature, dim=-1)
        return F.kl_div(log_probs, gold_probs, reduction='batchmean')
    
    def train_task(self, task_id: int):
        """Train on a single task."""
        cfg = self.config
        
        # Load data
        self.train_qa.load_dataset(task_id)
        task_name = self.train_qa.get_dataset_name(task_id)
        
        dataloader = DataLoader(
            self.train_qa,
            sampler=RandomSampler(self.train_qa),
            batch_size=cfg.training.batch_size,
            drop_last=True,
            num_workers=0,
            collate_fn=self.qa_collator
        )
        
        # Setup retriever
        current_datasets = cfg.data.datasets[:task_id + 1]
        index = self.index_manager.get_index(
            current_datasets,
            separate=cfg.retrieval.use_separate_index
        )
        retriever = Retriever(index, top_k=cfg.retrieval.top_k)
        
        # Setup optimizer
        num_steps = min(len(dataloader), cfg.training.max_steps)
        self._init_optimizer(task_id, num_steps)
        
        # Get generator for scoring
        generator = self._get_generator(for_training=True)
        
        # Training loop
        self.embedder.train_mode()
        progress = tqdm(
            enumerate(dataloader),
            total=num_steps,
            desc=f"Task {task_id + 1}/{self.num_tasks} [{task_name}]"
        )
        
        for step, batch in progress:
            if step >= cfg.training.max_steps:
                break
            
            indices, queries, answers = batch
            
            # Preprocess queries for specific models
            processed_queries = self._preprocess_queries(queries)
            
            # Encode queries - use training method for gradient tracking
            query_embeddings, query_states = self.embedder.encode_for_training_with_state(
                processed_queries,
                normalize=True
            )
            
            # Retrieve documents (use detached embeddings for retrieval)
            docs_list = []
            doc_embeddings_list = []
            doc_states_list = []
            
            for q_emb in query_embeddings:
                results = retriever.retrieve(q_emb.detach().cpu().numpy())
                docs = [r.content for r in results]
                docs_list.append(docs)
                
                # Encode documents - NO gradient needed for documents
                with torch.no_grad():
                    doc_embs, doc_states = self.embedder.encode_with_state(
                        docs,
                        normalize=True,
                        encoder_type="passage"
                    )
                doc_embeddings_list.append(doc_embs)
                if doc_states is not None:
                    doc_states_list.append(doc_states)
            
            # Compute retriever scores
            doc_embeddings = torch.stack(doc_embeddings_list)
            doc_states = (
                torch.stack(doc_states_list)
                if len(doc_states_list) == len(doc_embeddings_list)
                else None
            )
            retriever_scores = torch.einsum(
                "id,ijd->ij",
                query_embeddings,
                doc_embeddings
            )
            retriever_scores = retriever_scores / np.sqrt(query_embeddings.size(-1))
            
            # Get gold scores from generator
            # (This is where different CL methods compute their supervision signal)
            gold_scores = self._compute_gold_scores(
                queries, docs_list, answers, generator
            )
            
            # Compute loss
            loss = self.kl_div_loss(
                retriever_scores,
                gold_scores,
                temperature=cfg.model.temperature
            )

            if (
                cfg.training.use_ggf
                and query_states is not None
                and doc_states is not None
            ):
                state_scores = torch.einsum(
                    "id,ijd->ij",
                    query_states,
                    doc_states
                )
                state_scores = state_scores / cfg.training.ggf_temperature
                state_loss = self.kl_div_loss(
                    state_scores,
                    gold_scores,
                    temperature=1.0
                )
                loss = loss + cfg.training.beta * state_loss
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.optimizer.param_groups[0]['params'],
                max_norm=1.0
            )
            self.optimizer.step()
            self.scheduler.step()
            
            progress.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Periodic evaluation
            if (step + 1) % cfg.training.eval_interval == 0:
                self._run_intermediate_eval(task_id, step + 1, retriever)
    
    def _preprocess_queries(self, queries: List[str]) -> List[str]:
        """Preprocess queries based on retriever type."""
        name = self.config.model.retriever_name
        
        if name == 'e5':
            return [f"query: {q}" for q in queries]
        elif name == 'bge':
            prefix = "Represent this sentence for searching relevant passages: "
            return [prefix + q for q in queries]
        return queries
    
    def _compute_gold_scores(
        self,
        queries: List[str],
        docs_list: List[List[str]],
        answers: List[str],
        generator
    ) -> torch.Tensor:
        """Compute gold scores using generator."""
        # FlowRAG scoring method
        return generator.get_flowrag_score(queries, docs_list, answers)
    
    def _run_intermediate_eval(self, task_id: int, step: int, retriever: Retriever):
        """Run evaluation during training."""
        logger.info(f"Step {step}: Running intermediate evaluation...")
        self.embedder.eval_mode()
        
        for eval_id in range(task_id + 1):
            self.evaluate_task(task_id, eval_id, step=step)
        
        self.embedder.train_mode()
    
    def evaluate_task(
        self,
        train_task_id: int,
        eval_task_id: int,
        step: Optional[int] = None
    ):
        """Evaluate on a task."""
        self.test_qa.load_dataset(eval_task_id)
        task_name = self.test_qa.get_dataset_name(eval_task_id)
        
        dataloader = DataLoader(
            self.test_qa,
            sampler=SequentialSampler(self.test_qa),
            batch_size=self.config.training.batch_size,
            num_workers=0,
            collate_fn=self.qa_collator
        )
        
        # Setup retriever
        current_datasets = self.config.data.datasets[:train_task_id + 1]
        
        if self.config.retrieval.use_separate_index:
            eval_datasets = [self.config.data.datasets[eval_task_id]]
        else:
            eval_datasets = current_datasets
        
        index = self.index_manager.get_index(eval_datasets, separate=False)
        
        # Collect prompts and answers
        all_queries = []
        all_answers = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc=f"Eval [{task_name}]"):
                _, queries, answers = batch
                all_queries.extend(queries)
                all_answers.extend(answers)
        
        # Encode all queries
        processed_queries = self._preprocess_queries(all_queries)
        query_embeddings = self.embedder.encode(
            processed_queries,
            batch_size=32,
            normalize=True,
            show_progress=True,
            encoder_type="query"
        )
        
        # Batch retrieve
        results = index.batch_search(query_embeddings.cpu().numpy(), self.config.retrieval.top_k)
        
        # Generate prompts
        prompts = [
            self.prompt_template.render(documents=docs, query=query)
            for query, docs in zip(all_queries, results)
        ]
        
        # Save results
        self._save_results(
            train_task_id, eval_task_id, step,
            prompts, all_answers
        )
    
    def _save_results(
        self,
        train_id: int,
        eval_id: int,
        step: Optional[int],
        prompts: List[str],
        answers: List[str]
    ):
        """Save evaluation results."""
        output_dir = self.config.data.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        if step:
            filename = f"{train_id}_{eval_id}_step_{step}.csv"
        else:
            filename = f"{train_id}_{eval_id}.csv"
        
        df = pd.DataFrame({"prompt": prompts, "answer": answers})
        df.to_csv(os.path.join(output_dir, filename), index=False)
        logger.info(f"Saved: {filename}")
    
    @staticmethod
    def compute_metrics(prediction: str, answer: str) -> Dict[str, float]:
        """Compute evaluation metrics."""
        # Normalize text
        def normalize(s):
            s = s.lower()
            s = re.sub(r'[!"#$%&()*+,-./:;<=>?@\[\]\\^`{|}~_\']', ' ', s)
            s = re.sub(r'\b(a|an|the)\b', ' ', s)
            return ' '.join(s.split())
        
        pred_tokens = normalize(str(prediction)).split()
        ans_tokens = normalize(str(answer)).split()
        
        # F1
        common = Counter(pred_tokens) & Counter(ans_tokens)
        num_same = sum(common.values())
        
        if num_same == 0:
            f1 = 0.0
        else:
            precision = num_same / len(pred_tokens) if pred_tokens else 0
            recall = num_same / len(ans_tokens) if ans_tokens else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        
        # BLEU
        smooth = SmoothingFunction()
        bleu1 = sentence_bleu([ans_tokens], pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smooth.method1)
        bleu2 = sentence_bleu([ans_tokens], pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smooth.method1)
        bleu3 = sentence_bleu([ans_tokens], pred_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smooth.method1)
        bleu4 = sentence_bleu([ans_tokens], pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smooth.method1)
        
        # Exact match
        em = int(normalize(str(prediction)) == normalize(str(answer)))
        
        return {
            'em': em, 'f1': f1,
            'bleu1': bleu1, 'bleu2': bleu2, 'bleu3': bleu3, 'bleu4': bleu4
        }
    
    def run(self):
        """Run the full training and evaluation pipeline."""
        cfg = self.config.training
        
        if cfg.cl_method == 'offline':
            self._run_offline()
        else:
            self._run_online()
    
    def _run_offline(self):
        """Run offline evaluation (no training)."""
        logger.info("Running offline evaluation...")
        
        for task_id in range(self.num_tasks):
            if self.config.retrieval.use_separate_index:
                self.evaluate_task(task_id, task_id)
            else:
                for eval_id in range(task_id + 1):
                    self.evaluate_task(task_id, eval_id)
    
    def _run_online(self):
        """Run online continual learning."""
        logger.info(f"Running online training with method: {self.config.training.cl_method}")
        
        for task_id in range(self.num_tasks):
            logger.info(f"{'='*20} Task {task_id + 1}/{self.num_tasks} {'='*20}")
            
            # Train
            self.train_task(task_id)
            
            # Evaluate all seen tasks
            logger.info(f"Task {task_id + 1} complete. Running final evaluation...")
            self.embedder.eval_mode()
            
            for eval_id in range(task_id + 1):
                self.evaluate_task(task_id, eval_id)
        
        logger.info("Training complete!")
