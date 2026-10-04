"""Offline smoke test for the full FlowRAG training pipeline."""

from __future__ import annotations

import json
import os
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import torch
from transformers import BertConfig, BertModel, GPT2Config, GPT2LMHeadModel

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))

from src.config import FlowRAGConfig
from src.data import dataset as dataset_module
from src.models.config import PromptConfig
from src.models.embedder import LayerWisePromptEmbedder
from src.models.generator import HFGenerator, TinyCharTokenizer
from src.retrieval.index import FaissIndex
import src.training.trainer as trainer_module
from src.training import FlowRAGTrainer


LOSS_VALUES = []


class BatchEncodingLite(dict):
    def to(self, device):
        return BatchEncodingLite({key: value.to(device) for key, value in self.items()})


class TinyTextTokenizer:
    def __init__(self, vocab_size: int = 97):
        self.vocab_size = vocab_size
        self.pad_token_id = 0

    def __call__(self, texts, padding=True, truncation=True, return_tensors=None):
        if isinstance(texts, str):
            texts = [texts]
        encoded = [
            [2 + (ord(ch) % (self.vocab_size - 2)) for ch in text]
            for text in texts
        ]
        max_len = max(len(ids) for ids in encoded)
        padded = [ids + [self.pad_token_id] * (max_len - len(ids)) for ids in encoded]
        masks = [[1] * len(ids) + [0] * (max_len - len(ids)) for ids in encoded]
        return BatchEncodingLite(
            {
                "input_ids": torch.tensor(padded, dtype=torch.long),
                "attention_mask": torch.tensor(masks, dtype=torch.long),
            }
        )


def run_stage(name, fn):
    try:
        result = fn()
        print(f"PASS: {name}")
        return result
    except Exception:
        print(f"FAIL: {name}")
        traceback.print_exc()
        raise


def write_fake_dataset(data_dir: Path, datasets):
    dataset_module.DATASET_REGISTRY.update(
        {name: {"type": "extraction"} for name in datasets}
    )
    for name in datasets:
        task_dir = data_dir / name
        task_dir.mkdir(parents=True, exist_ok=True)
        for split in ("train", "test"):
            rows = 8
            data = {
                "question": [f"{name} {split} question {i}?" for i in range(rows)],
                "answer": [f"answer {i % 3}" for i in range(rows)],
                "context": [
                    f"{name} {split} context passage {i}. answer {i % 3}."
                    for i in range(rows)
                ],
            }
            with open(task_dir / f"{split}.json", "w") as f:
                json.dump(data, f)


def build_fake_indices(index_dir: Path, data_dir: Path, datasets, dim: int = 32):
    rng = np.random.default_rng(123)
    for name in datasets:
        with open(data_dir / name / "train.json", "r") as f:
            data = json.load(f)
        contexts = data["context"]
        embeddings = rng.normal(size=(len(contexts), dim)).astype(np.float32)
        embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
        index = FaissIndex(str(index_dir))
        index.build(embeddings, contexts, "FLAT")
        index.save(f"ctr_{name}_flat")


def make_embedder(retriever_name, use_prompt, prompt_config, device):
    config = BertConfig(
        vocab_size=97,
        hidden_size=32,
        num_hidden_layers=4,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=128,
    )
    tokenizer = TinyTextTokenizer(vocab_size=97)
    return LayerWisePromptEmbedder(
        model_name="random-bert",
        prompt_config=prompt_config,
        use_prompt=use_prompt,
        device=device,
        query_model=BertModel(config),
        passage_model=BertModel(config),
        query_tokenizer=tokenizer,
        passage_tokenizer=tokenizer,
    )


def make_generator(model_name, max_new_tokens=64, use_vllm=False, device="cpu"):
    config = GPT2Config(
        n_embd=32,
        n_layer=2,
        n_head=2,
        vocab_size=100,
        n_positions=128,
        n_ctx=128,
        bos_token_id=1,
        eos_token_id=1,
        pad_token_id=0,
    )
    return HFGenerator(
        "random-gpt2",
        max_new_tokens=max_new_tokens,
        use_vllm=False,
        device=device,
        model=GPT2LMHeadModel(config),
        tokenizer=TinyCharTokenizer(vocab_size=100),
    )


def patch_models_and_losses():
    trainer_module.create_embedder = make_embedder
    trainer_module.create_generator = make_generator

    original_kl = FlowRAGTrainer.kl_div_loss

    def recording_kl(scores, gold_scores, temperature=0.1):
        loss = original_kl(scores, gold_scores, temperature)
        LOSS_VALUES.append(float(loss.detach().cpu()))
        return loss

    FlowRAGTrainer.kl_div_loss = staticmethod(recording_kl)


def make_config(data_dir: Path, output_dir: Path, index_dir: Path, datasets):
    cfg = FlowRAGConfig()
    cfg.data.datasets = list(datasets)
    cfg.data.data_dir = str(data_dir)
    cfg.data.output_dir = str(output_dir)
    cfg.retrieval.faiss_db_path = str(index_dir)
    cfg.retrieval.top_k = 2
    cfg.model.retriever_name = "contriever"
    cfg.model.generator_name = "random-gpt2"
    cfg.model.max_new_tokens = 8
    cfg.training.cl_method = "fp"
    cfg.training.use_ilf = True
    cfg.training.use_cef = True
    cfg.training.use_ggf = True
    cfg.training.prompt_layer = 2
    cfg.training.prompt_len = 5
    cfg.training.max_steps = 2
    cfg.training.batch_size = 1
    cfg.training.eval_interval = 1
    cfg.training.learning_rate = 5e-3
    cfg.seed = 123
    return cfg


def assert_outputs(output_dir: Path):
    expected = ["0_0.csv", "1_0.csv", "1_1.csv"]
    missing = [name for name in expected if not (output_dir / name).exists()]
    if missing:
        raise AssertionError(f"Missing output CSVs: {missing}")


def assert_losses_finite():
    if not LOSS_VALUES:
        raise AssertionError("No losses were recorded")
    if not all(np.isfinite(value) for value in LOSS_VALUES):
        raise AssertionError(f"Non-finite losses recorded: {LOSS_VALUES}")
    print(f"recorded losses: {[round(value, 6) for value in LOSS_VALUES]}")


def main():
    work_dir = Path(tempfile.mkdtemp(prefix="flowrag_smoke_"))
    print(f"smoke work dir: {work_dir}")
    try:
        data_dir = work_dir / "cl_datasets"
        output_dir = work_dir / "output"
        index_dir = work_dir / "faiss_db"
        datasets = ["taskA", "taskB"]

        run_stage("dataset load", lambda: write_fake_dataset(data_dir, datasets))
        run_stage("dataset indices", lambda: build_fake_indices(index_dir, data_dir, datasets))
        run_stage("embedder init", patch_models_and_losses)
        run_stage("generator init", lambda: make_generator("random-gpt2").load())

        config = make_config(data_dir, output_dir, index_dir, datasets)
        trainer = run_stage("trainer init", lambda: FlowRAGTrainer(config))

        original_train_task = trainer.train_task
        original_evaluate_task = trainer.evaluate_task
        original_init_optimizer = trainer._init_optimizer
        trained_tasks = set()
        evaluated_tasks = set()
        grad_check = {"captured": False}

        def init_optimizer_with_grad_probe(task_id, num_steps):
            result = original_init_optimizer(task_id, num_steps)
            original_step = trainer.optimizer.step

            def step_with_grad_probe(*args, **kwargs):
                if task_id == 0 and not grad_check["captured"]:
                    prompt_grad = trainer.embedder.prompt_embeddings.grad
                    fusion_grad = (
                        trainer.embedder
                        .cross_layer_fusion
                        .state_fusion
                        .query
                        .weight
                        .grad
                    )
                    prompt_norm = None if prompt_grad is None else prompt_grad.norm().item()
                    fusion_norm = None if fusion_grad is None else fusion_grad.norm().item()
                    print(f"prompt gradient norm before first optimizer.step(): {prompt_norm}")
                    print(f"state fusion query weight gradient norm before first optimizer.step(): {fusion_norm}")
                    assert prompt_norm is not None, "prompt_embeddings.grad is None"
                    assert fusion_norm is not None, "StateFusion query.weight.grad is None"
                    assert prompt_norm != 0.0, "prompt_embeddings.grad norm is exactly zero"
                    assert fusion_norm != 0.0, "StateFusion query.weight.grad norm is exactly zero"
                    assert np.isfinite(prompt_norm), f"prompt grad norm is non-finite: {prompt_norm}"
                    assert np.isfinite(fusion_norm), f"fusion grad norm is non-finite: {fusion_norm}"
                    grad_check["captured"] = True
                return original_step(*args, **kwargs)

            trainer.optimizer.step = step_with_grad_probe
            return result

        def train_task_with_stage(task_id):
            result = original_train_task(task_id)
            trained_tasks.add(task_id)
            print(f"PASS: task {task_id + 1} train")
            return result

        def evaluate_task_with_stage(train_task_id, eval_task_id, step=None):
            result = original_evaluate_task(train_task_id, eval_task_id, step)
            if step is None:
                evaluated_tasks.add((train_task_id, eval_task_id))
            return result

        trainer.train_task = train_task_with_stage
        trainer.evaluate_task = evaluate_task_with_stage
        trainer._init_optimizer = init_optimizer_with_grad_probe

        run_stage("full trainer run", trainer.run)
        if not grad_check["captured"]:
            raise AssertionError("Gradient diagnostic did not run during task 1")

        if 0 not in trained_tasks:
            raise AssertionError("Task 1 train did not complete")
        if 1 not in trained_tasks:
            raise AssertionError("Task 2 train did not complete")
        if (0, 0) not in evaluated_tasks:
            raise AssertionError("Task 1 eval did not complete")
        if (1, 0) not in evaluated_tasks or (1, 1) not in evaluated_tasks:
            raise AssertionError("Task 2 eval did not complete")
        print("PASS: task 1 eval")
        print("PASS: task 2 eval")

        run_stage("output CSVs", lambda: assert_outputs(output_dir))
        run_stage("finite losses", assert_losses_finite)
        print("PASS: smoke test completed")
    finally:
        if os.environ.get("FLOWRAG_KEEP_SMOKE_DIR") != "1":
            shutil.rmtree(work_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
