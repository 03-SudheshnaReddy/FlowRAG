"""State fusion module for FlowRAG layer-state construction."""

from __future__ import annotations

import math

import torch
import torch.nn as nn


def transpose_for_scores(
    tensor: torch.Tensor,
    num_attention_heads: int,
    attention_head_size: int,
) -> torch.Tensor:
    """Convert [B, T, D] projections to [B, heads, T, head_dim]."""

    new_shape = tensor.size()[:-1] + (num_attention_heads, attention_head_size)
    tensor = tensor.view(new_shape)
    return tensor.permute(0, 2, 1, 3)


class StateFusion(nn.Module):
    """Algorithm 1 state fusion with multi-head attention.

    ``forward`` returns the paper's expanded intermediate sequence with shape
    [B, T+2, D]. Algorithm 2 needs stackable layer states, so ``layer_state``
    mean-pools that intermediate to a fixed [B, D] vector.
    """

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        if hidden_size % num_heads != 0:
            raise ValueError("hidden_size must be divisible by num_heads")

        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention_head_size = hidden_size // num_heads
        self.all_head_size = self.num_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.output = nn.Linear(hidden_size, hidden_size)
        self.layer_norm = nn.LayerNorm(hidden_size)

    def forward(self, H: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        mu_h = H.mean(dim=1)
        sigma_h = H.std(dim=1, unbiased=False)
        mu_p = P.mean(dim=1)
        sigma_p = P.std(dim=1, unbiased=False)

        Q = self.query(H)
        K = self.key(P)
        V = self.value(P)

        h_stats = torch.stack([mu_h, sigma_h], dim=1)
        p_stats = torch.stack([mu_p, sigma_p], dim=1)

        Q = torch.cat([Q, h_stats], dim=1)
        K = torch.cat([K, p_stats], dim=1)
        V = torch.cat([V, p_stats], dim=1)

        query_layer = transpose_for_scores(Q, self.num_heads, self.attention_head_size)
        key_layer = transpose_for_scores(K, self.num_heads, self.attention_head_size)
        value_layer = transpose_for_scores(V, self.num_heads, self.attention_head_size)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        context_layer = context_layer.view(context_layer.size()[:-2] + (self.all_head_size,))

        residual = torch.cat([H, h_stats], dim=1)
        return self.layer_norm(self.output(context_layer) + residual)

    def layer_state(self, H: torch.Tensor, P: torch.Tensor) -> torch.Tensor:
        """Return the fixed-size layer state used by cross-layer fusion."""

        return self.forward(H, P).mean(dim=1)


class CrossLayerFusion(nn.Module):
    """Algorithm 2 cross-layer fusion over fixed-size pooled layer states."""

    def __init__(self, hidden_size: int, num_heads: int):
        super().__init__()
        self.state_fusion = StateFusion(hidden_size, num_heads)
        self.pre_layer_states = []

    def reset_state(self):
        self.pre_layer_states = []

    def forward(
        self,
        H: torch.Tensor,
        P: torch.Tensor,
        is_first_prompt_layer: bool,
        is_last_prompt_layer: bool,
    ):
        """
        H: hidden states [B, T, D] at this layer
        P: prompt slice [B, p, D] at this layer
        Returns: (A_j, L_j)
          L_j: pooled layer state [B, D], always returned and appended
          A_j: aggregated state for Eq. 4, or None before the last prompt layer
        """

        raw = self.state_fusion(H, P)
        # StateFusion grows sequence length by 2 via appended stat tokens. Algorithm
        # 2 recursively stacks previous layer states, so storing raw [B, T+2, D]
        # outputs would make lengths grow across layers and Stack() would fail.
        # Mean-pooling defines each Layer State L_j as fixed [B, D].
        L_j = raw.mean(dim=1)
        self.pre_layer_states.append(L_j)

        if is_first_prompt_layer:
            A_j = L_j
        else:
            stacked = torch.stack(self.pre_layer_states, dim=0)
            S = stacked.mean(dim=0)
            A_j_raw = self.state_fusion(S.unsqueeze(1), L_j.unsqueeze(1))
            A_j = A_j_raw.mean(dim=1)

        if not is_last_prompt_layer:
            A_j = None

        return A_j, L_j


def _run_standalone_test() -> bool:
    H = torch.randn(2, 10, 32, requires_grad=True)
    P = torch.randn(2, 5, 32)
    fusion = StateFusion(hidden_size=32, num_heads=4)

    output = fusion(H, P)
    layer_state = fusion.layer_state(H, P)

    print(f"H shape: {list(H.shape)}")
    print(f"P shape: {list(P.shape)}")
    print(f"output shape: {list(output.shape)}")
    print(f"layer_state shape: {list(layer_state.shape)}")

    # Output length is 12 because Algorithm 1 appends mu_h and sigma_h to the
    # query/state side: T + 2 = 10 + 2. It is not T=10 or T+p=15.
    assert list(output.shape) == [2, 12, 32], output.shape
    assert list(layer_state.shape) == [2, 32], layer_state.shape

    output.sum().backward()
    assert H.grad is not None, "Expected gradients to flow back to H"

    clf = CrossLayerFusion(hidden_size=32, num_heads=4)
    clf.reset_state()
    A_0, L_0 = clf(H.detach(), P, is_first_prompt_layer=True, is_last_prompt_layer=False)
    A_1, L_1 = clf(H.detach(), P, is_first_prompt_layer=False, is_last_prompt_layer=True)
    print(f"CrossLayerFusion first A shape: {None if A_0 is None else list(A_0.shape)}")
    print(f"CrossLayerFusion first L shape: {list(L_0.shape)}")
    print(f"CrossLayerFusion last A shape: {list(A_1.shape)}")
    print(f"CrossLayerFusion last L shape: {list(L_1.shape)}")
    assert A_0 is None
    assert list(L_0.shape) == [2, 32], L_0.shape
    assert list(A_1.shape) == [2, 32], A_1.shape
    assert list(L_1.shape) == [2, 32], L_1.shape
    return True


if __name__ == "__main__":
    try:
        _run_standalone_test()
        print("PASS: StateFusion standalone test succeeded")
    except Exception as exc:
        print(f"FAIL: StateFusion standalone test failed: {exc}")
        raise
