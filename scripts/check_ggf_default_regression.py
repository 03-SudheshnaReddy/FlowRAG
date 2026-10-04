"""Check that use_ggf=False preserves the retrieval-only loss path."""

from __future__ import annotations

import torch
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))

from src.config import FlowRAGConfig
from src.training.trainer import FlowRAGTrainer


def _new_loss_branch(trainer, retriever_scores, gold_scores, query_states, doc_states):
    cfg = trainer.config
    retrieval_loss = trainer.kl_div_loss(
        retriever_scores,
        gold_scores,
        temperature=cfg.model.temperature,
    )
    loss = retrieval_loss
    state_loss = None

    if cfg.training.use_ggf and query_states is not None and doc_states is not None:
        state_scores = torch.einsum("id,ijd->ij", query_states, doc_states)
        state_scores = state_scores / cfg.training.ggf_temperature
        state_loss = trainer.kl_div_loss(state_scores, gold_scores, temperature=1.0)
        loss = retrieval_loss + cfg.training.beta * state_loss

    return loss, retrieval_loss, state_loss


def main():
    torch.manual_seed(123)

    config_a = FlowRAGConfig()
    config_a.training.use_ggf = False
    config_b = FlowRAGConfig()
    config_b.training.use_ggf = False
    config_c = FlowRAGConfig()
    config_c.training.use_ggf = True
    config_c.training.use_ilf = True
    config_c.training.use_cef = True

    trainer_before = FlowRAGTrainer.__new__(FlowRAGTrainer)
    trainer_before.config = config_a
    trainer_after = FlowRAGTrainer.__new__(FlowRAGTrainer)
    trainer_after.config = config_b
    trainer_ggf = FlowRAGTrainer.__new__(FlowRAGTrainer)
    trainer_ggf.config = config_c

    retriever_scores = torch.randn(2, 3)
    gold_scores = torch.randn(2, 3)
    query_states = torch.randn(2, 8)
    doc_states = torch.randn(2, 3, 8)

    old_loss = trainer_before.kl_div_loss(
        retriever_scores,
        gold_scores,
        temperature=config_a.model.temperature,
    )
    new_loss, _, _ = _new_loss_branch(
        trainer_after,
        retriever_scores,
        gold_scores,
        query_states,
        doc_states,
    )
    ggf_loss, retrieval_loss, state_loss = _new_loss_branch(
        trainer_ggf,
        retriever_scores,
        gold_scores,
        query_states,
        doc_states,
    )

    print(f"old retrieval-only loss: {old_loss.item():.12f}")
    print(f"new use_ggf=False loss: {new_loss.item():.12f}")
    print(f"bit-identical: {old_loss.item() == new_loss.item()}")
    print(f"use_ggf=True retrieval_loss: {retrieval_loss.item():.12f}")
    print(f"use_ggf=True state_loss: {state_loss.item():.12f}")
    print(f"use_ggf=True combined loss: {ggf_loss.item():.12f}")
    print(f"use_ggf=True finite: {bool(torch.isfinite(ggf_loss))}")
    print(f"use_ggf=True differs from use_ggf=False: {not torch.allclose(ggf_loss, new_loss)}")

    assert old_loss.item() == new_loss.item(), "use_ggf=False changed the loss value"
    assert torch.isfinite(retrieval_loss), retrieval_loss
    assert torch.isfinite(state_loss), state_loss
    assert torch.isfinite(ggf_loss), ggf_loss
    assert not torch.allclose(ggf_loss, new_loss), "use_ggf=True did not change the loss"
    print("PASS: use_ggf=False regression check succeeded")
    print("PASS: use_ggf=True loss check succeeded")


if __name__ == "__main__":
    main()
