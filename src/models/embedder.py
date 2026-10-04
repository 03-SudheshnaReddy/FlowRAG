"""Prompt-aware BERT-style embedders for FlowRAG."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Iterable, List, Optional, Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModel, AutoTokenizer, BertConfig, BertModel
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions

try:
    from src.config import RETRIEVER_MODELS
except ImportError:  # Allows `python src/models/embedder.py` from the repo root.
    sys.path.append(str(Path(__file__).resolve().parents[2]))
    from src.config import RETRIEVER_MODELS

try:
    from .config import PromptConfig
    from .fusion import CrossLayerFusion, StateFusion, transpose_for_scores
except ImportError:  # Allows `python src/models/embedder.py` for the standalone test.
    from config import PromptConfig
    from fusion import CrossLayerFusion, StateFusion, transpose_for_scores


class LayerWisePromptEmbedder(nn.Module):
    """BERT-style encoder with FlowRAG K/V-only prompts.

    For prompted layers, query projections are computed from real token hidden
    states, while key/value projections are computed from prompt-prefixed hidden
    states. Prompt positions are visible to real tokens but do not become output
    sequence positions.
    """

    def __init__(
        self,
        model_name: str = "facebook/contriever",
        query_model_name: Optional[str] = None,
        passage_model_name: Optional[str] = None,
        prompt_config: Optional[PromptConfig] = None,
        use_prompt: bool = False,
        device: Optional[torch.device | str] = None,
        model: Optional[BertModel] = None,
        query_model: Optional[BertModel] = None,
        passage_model: Optional[BertModel] = None,
        tokenizer=None,
        query_tokenizer=None,
        passage_tokenizer=None,
    ):
        super().__init__()
        self.query_model_name = query_model_name or model_name
        self.passage_model_name = passage_model_name or self.query_model_name
        self.model_name = self.query_model_name
        self.prompt_config = prompt_config
        self.use_prompt = use_prompt and prompt_config is not None
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.current_task = 0
        self._last_aggregated_state = None

        self.query_model = query_model or model
        if self.query_model is None:
            self.query_model = AutoModel.from_pretrained(self.query_model_name)

        if passage_model is not None:
            self.passage_model = passage_model
        elif self.passage_model_name == self.query_model_name:
            self.passage_model = self.query_model
        else:
            self.passage_model = AutoModel.from_pretrained(self.passage_model_name)

        if tokenizer is not None:
            self.query_tokenizer = tokenizer
            self.passage_tokenizer = tokenizer
        else:
            self.query_tokenizer = query_tokenizer
            self.passage_tokenizer = passage_tokenizer

        if self.query_tokenizer is None and self.query_model_name != "random-bert":
            self.query_tokenizer = AutoTokenizer.from_pretrained(self.query_model_name)
        if self.passage_tokenizer is None and self.passage_model_name != "random-bert":
            self.passage_tokenizer = AutoTokenizer.from_pretrained(self.passage_model_name)

        # Backward-compatible aliases for existing diagnostics/tests.
        self.model = self.query_model
        self.tokenizer = self.query_tokenizer

        self.query_model.to(self.device)
        self.passage_model.to(self.device)
        self._freeze_base_encoder()

        hidden_size = self.query_model.config.hidden_size
        num_layers = len(self.query_model.encoder.layer)
        num_heads = self.query_model.config.num_attention_heads
        if self.passage_model.config.hidden_size != hidden_size:
            raise ValueError("query and passage encoders must have the same hidden_size")

        if self.use_prompt:
            if prompt_config.prompt_layer > num_layers:
                raise ValueError(
                    f"prompt_layer={prompt_config.prompt_layer} exceeds encoder layers={num_layers}"
                )
            self.prompt_embeddings = nn.Parameter(
                torch.empty(
                    prompt_config.num_tasks,
                    prompt_config.prompt_layer,
                    prompt_config.prompt_len,
                    hidden_size,
                )
            )
            nn.init.normal_(self.prompt_embeddings, mean=0.0, std=0.02)
        else:
            self.register_parameter("prompt_embeddings", None)

        self.cross_layer_fusion = (
            CrossLayerFusion(hidden_size, num_heads) if self.use_prompt else None
        )
        self.to(self.device)

    @classmethod
    def from_random_bert(
        cls,
        config: BertConfig,
        prompt_config: PromptConfig,
        device: Optional[torch.device | str] = None,
    ) -> "LayerWisePromptEmbedder":
        model = BertModel(config)
        return cls(
            model_name="random-bert",
            prompt_config=prompt_config,
            use_prompt=True,
            device=device,
            query_model=model,
            passage_model=model,
            query_tokenizer=None,
            passage_tokenizer=None,
        )

    def _freeze_base_encoder(self) -> None:
        for param in self.query_model.parameters():
            param.requires_grad = False
        for param in self.passage_model.parameters():
            param.requires_grad = False

    def set_task(self, task_id: int):
        if self.prompt_config is not None and not 0 <= task_id < self.prompt_config.num_tasks:
            raise ValueError(f"task_id={task_id} outside [0, {self.prompt_config.num_tasks})")
        self.current_task = task_id

    def get_trainable_params(self):
        return [p for p in self.parameters() if p.requires_grad]

    def train_mode(self):
        self.train()

    def eval_mode(self):
        self.eval()

    def encode_for_training(
        self,
        texts: Sequence[str],
        normalize: bool = True,
    ) -> torch.Tensor:
        if self.query_tokenizer is None:
            raise RuntimeError("encode_for_training requires a tokenizer-backed embedder")
        batch = self._tokenize(texts, encoder_type="query")
        return self.forward(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            normalize=normalize,
            encoder_type="query",
        )

    def encode_for_training_with_state(
        self,
        texts: Sequence[str],
        normalize: bool = True,
    ):
        embedding = self.encode_for_training(texts, normalize=normalize)
        state = self._last_aggregated_state
        if state is not None and normalize:
            state = F.normalize(state, p=2, dim=-1)
        return embedding, state

    @torch.no_grad()
    def encode(
        self,
        texts: Sequence[str],
        batch_size: Optional[int] = None,
        normalize: bool = True,
        show_progress: bool = False,
        encoder_type: str = "query",
    ) -> torch.Tensor:
        tokenizer = self._get_tokenizer(encoder_type)
        if tokenizer is None:
            raise RuntimeError("encode requires a tokenizer-backed embedder")

        text_list = list(texts)
        if not text_list:
            hidden_size = self.query_model.config.hidden_size
            return torch.empty(0, hidden_size, device=self.device)

        effective_batch_size = batch_size or len(text_list)
        chunks = self._batched(text_list, effective_batch_size)
        if show_progress:
            chunks = tqdm(list(chunks), desc="Encoding")

        outputs = []
        was_training = self.training
        self.eval()
        for chunk in chunks:
            batch = self._tokenize(chunk, encoder_type=encoder_type)
            outputs.append(
                self.forward(
                    input_ids=batch["input_ids"],
                    attention_mask=batch["attention_mask"],
                    normalize=normalize,
                    encoder_type=encoder_type,
                )
            )
        if was_training:
            self.train()
        return torch.cat(outputs, dim=0)

    @torch.no_grad()
    def encode_with_state(
        self,
        texts: Sequence[str],
        batch_size: Optional[int] = None,
        normalize: bool = True,
        show_progress: bool = False,
        encoder_type: str = "query",
    ):
        tokenizer = self._get_tokenizer(encoder_type)
        if tokenizer is None:
            raise RuntimeError("encode_with_state requires a tokenizer-backed embedder")

        text_list = list(texts)
        if not text_list:
            hidden_size = self.query_model.config.hidden_size
            empty = torch.empty(0, hidden_size, device=self.device)
            return empty, None

        effective_batch_size = batch_size or len(text_list)
        chunks = self._batched(text_list, effective_batch_size)
        if show_progress:
            chunks = tqdm(list(chunks), desc="Encoding")

        outputs = []
        states = []
        has_state = True
        was_training = self.training
        self.eval()
        for chunk in chunks:
            batch = self._tokenize(chunk, encoder_type=encoder_type)
            embedding = self.forward(
                input_ids=batch["input_ids"],
                attention_mask=batch["attention_mask"],
                normalize=normalize,
                encoder_type=encoder_type,
            )
            outputs.append(embedding)
            state = self._last_aggregated_state
            if state is None:
                has_state = False
            else:
                if normalize:
                    state = F.normalize(state, p=2, dim=-1)
                states.append(state)
        if was_training:
            self.train()

        embeddings = torch.cat(outputs, dim=0)
        aggregated_states = torch.cat(states, dim=0) if has_state else None
        return embeddings, aggregated_states

    def _get_tokenizer(self, encoder_type: str):
        if encoder_type == "query":
            return self.query_tokenizer
        if encoder_type == "passage":
            return self.passage_tokenizer
        raise ValueError(f"Unknown encoder_type: {encoder_type}")

    def _get_model(self, encoder_type: str):
        if encoder_type == "query":
            return self.query_model
        if encoder_type == "passage":
            return self.passage_model
        raise ValueError(f"Unknown encoder_type: {encoder_type}")

    def _tokenize(self, texts: Sequence[str], encoder_type: str):
        return self._get_tokenizer(encoder_type)(
            list(texts),
            padding=True,
            truncation=True,
            return_tensors="pt",
        ).to(self.device)

    @staticmethod
    def _batched(items: Sequence[str], batch_size: int) -> Iterable[List[str]]:
        for start in range(0, len(items), batch_size):
            yield list(items[start : start + batch_size])

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
        normalize: bool = True,
        encoder_type: str = "query",
    ) -> torch.Tensor:
        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)

        outputs = self._forward_sequence(input_ids, attention_mask, token_type_ids, encoder_type)
        pooled = self._mean_pool(outputs.last_hidden_state, attention_mask)
        if normalize:
            pooled = F.normalize(pooled, p=2, dim=-1)
        return pooled

    def _forward_sequence(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None,
        encoder_type: str = "query",
    ) -> BaseModelOutputWithPoolingAndCrossAttentions:
        model = self._get_model(encoder_type)
        if not self.use_prompt:
            return model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                return_dict=True,
            )

        hidden_states = model.embeddings(
            input_ids=input_ids,
            token_type_ids=token_type_ids,
        )
        self._last_aggregated_state = None
        if (
            self.cross_layer_fusion is not None
            and self.prompt_config.use_ilf
            and self.prompt_config.use_cef
        ):
            self.cross_layer_fusion.reset_state()

        for layer_idx, layer_module in enumerate(model.encoder.layer):
            if layer_idx < self.prompt_config.prompt_layer:
                hidden_states = self._forward_prompted_layer(
                    layer_module,
                    hidden_states,
                    attention_mask,
                    layer_idx,
                    is_first_prompt_layer=(layer_idx == 0),
                    is_last_prompt_layer=(layer_idx == self.prompt_config.prompt_layer - 1),
                )
            else:
                hidden_states = self._forward_unprompted_layer(
                    layer_module,
                    hidden_states,
                    attention_mask,
                )

        return BaseModelOutputWithPoolingAndCrossAttentions(last_hidden_state=hidden_states)

    def _forward_prompted_layer(
        self,
        layer_module: nn.Module,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
        layer_idx: int,
        is_first_prompt_layer: bool,
        is_last_prompt_layer: bool,
    ) -> torch.Tensor:
        self_attention = layer_module.attention.self
        prompt = self.prompt_embeddings[self.current_task, layer_idx]
        prompt = prompt.unsqueeze(0).expand(hidden_states.size(0), -1, -1)

        kv_parts = [prompt, hidden_states]
        mask_parts = []
        if (
            self.prompt_config.use_ilf
            and self.prompt_config.use_cef
            and self.cross_layer_fusion is not None
        ):
            A_j, _ = self.cross_layer_fusion(
                hidden_states,
                prompt,
                is_first_prompt_layer=is_first_prompt_layer,
                is_last_prompt_layer=is_last_prompt_layer,
            )
            if A_j is not None:
                self._last_aggregated_state = A_j
                kv_parts = [A_j.unsqueeze(1), *kv_parts]
                mask_parts.append(
                    torch.ones(
                        hidden_states.size(0),
                        1,
                        dtype=attention_mask.dtype,
                        device=attention_mask.device,
                    )
                )

        kv_hidden_states = torch.cat(kv_parts, dim=1)
        prompt_mask = torch.ones(
            hidden_states.size(0),
            prompt.size(1),
            dtype=attention_mask.dtype,
            device=attention_mask.device,
        )
        kv_attention_mask = torch.cat([*mask_parts, prompt_mask, attention_mask], dim=1)
        extended_mask = self._extended_attention_mask(kv_attention_mask, hidden_states.dtype)

        query_layer = self._transpose_for_scores(
            self_attention.query(hidden_states),
            self_attention.num_attention_heads,
            self_attention.attention_head_size,
        )
        key_layer = self._transpose_for_scores(
            self_attention.key(kv_hidden_states),
            self_attention.num_attention_heads,
            self_attention.attention_head_size,
        )
        value_layer = self._transpose_for_scores(
            self_attention.value(kv_hidden_states),
            self_attention.num_attention_heads,
            self_attention.attention_head_size,
        )

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self_attention.attention_head_size)
        attention_scores = attention_scores + extended_mask

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self_attention.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context_layer.size()[:-2] + (self_attention.all_head_size,)
        context_layer = context_layer.view(new_context_shape)

        attention_output = layer_module.attention.output(context_layer, hidden_states)
        intermediate_output = layer_module.intermediate(attention_output)
        return layer_module.output(intermediate_output, attention_output)

    def _forward_unprompted_layer(
        self,
        layer_module: nn.Module,
        hidden_states: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        self_attention = layer_module.attention.self
        extended_mask = self._extended_attention_mask(attention_mask, hidden_states.dtype)

        query_layer = self._transpose_for_scores(
            self_attention.query(hidden_states),
            self_attention.num_attention_heads,
            self_attention.attention_head_size,
        )
        key_layer = self._transpose_for_scores(
            self_attention.key(hidden_states),
            self_attention.num_attention_heads,
            self_attention.attention_head_size,
        )
        value_layer = self._transpose_for_scores(
            self_attention.value(hidden_states),
            self_attention.num_attention_heads,
            self_attention.attention_head_size,
        )

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self_attention.attention_head_size)
        attention_scores = attention_scores + extended_mask

        attention_probs = nn.functional.softmax(attention_scores, dim=-1)
        attention_probs = self_attention.dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_shape = context_layer.size()[:-2] + (self_attention.all_head_size,)
        context_layer = context_layer.view(new_context_shape)

        attention_output = layer_module.attention.output(context_layer, hidden_states)
        intermediate_output = layer_module.intermediate(attention_output)
        return layer_module.output(intermediate_output, attention_output)

    def _extended_attention_mask(self, attention_mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
        extended = attention_mask[:, None, None, :].to(dtype=dtype, device=attention_mask.device)
        return (1.0 - extended) * torch.finfo(dtype).min

    @staticmethod
    def _transpose_for_scores(
        tensor: torch.Tensor,
        num_attention_heads: int,
        attention_head_size: int,
    ) -> torch.Tensor:
        return transpose_for_scores(tensor, num_attention_heads, attention_head_size)

    @staticmethod
    def _mean_pool(hidden_states: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        mask = attention_mask.unsqueeze(-1).to(hidden_states.dtype)
        summed = (hidden_states * mask).sum(dim=1)
        counts = mask.sum(dim=1).clamp(min=1e-9)
        return summed / counts


def create_embedder(
    retriever_name: str,
    use_prompt: bool,
    prompt_config: Optional[PromptConfig],
    device,
):
    retriever_info = RETRIEVER_MODELS.get(retriever_name)
    query_model_name = retriever_info["query"] if retriever_info else retriever_name
    passage_model_name = retriever_info["passage"] if retriever_info else query_model_name
    return LayerWisePromptEmbedder(
        query_model_name=query_model_name,
        passage_model_name=passage_model_name,
        prompt_config=prompt_config,
        use_prompt=use_prompt,
        device=device,
    )


def _run_single_combo_test(
    base_state_dict,
    use_ilf: bool,
    use_cef: bool,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
) -> torch.Tensor:
    hidden_size = 32
    bert_config = BertConfig(
        vocab_size=97,
        hidden_size=hidden_size,
        num_hidden_layers=4,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
    )
    prompt_config = PromptConfig(
        num_tasks=2,
        prompt_layer=3,
        prompt_len=5,
        use_ilf=use_ilf,
        use_cef=use_cef,
    )
    torch.manual_seed(1234)
    embedder = LayerWisePromptEmbedder.from_random_bert(bert_config, prompt_config)
    embedder.load_state_dict(base_state_dict)
    embedder.set_task(0)
    output = embedder(input_ids=input_ids, attention_mask=attention_mask)
    print(
        f"combo use_ilf={use_ilf}, use_cef={use_cef}: "
        f"output shape {list(output.shape)}"
    )
    assert list(output.shape) == [input_ids.size(0), hidden_size], output.shape
    assert torch.isfinite(output).all(), "Expected finite output values"
    print(f"PASS: combo use_ilf={use_ilf}, use_cef={use_cef}")
    return output


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


class BatchEncodingLite(dict):
    def to(self, device):
        return BatchEncodingLite({key: value.to(device) for key, value in self.items()})


def _run_standalone_test() -> bool:
    batch_size = 3
    seq_len = 11
    hidden_size = 32
    bert_config = BertConfig(
        vocab_size=97,
        hidden_size=hidden_size,
        num_hidden_layers=4,
        num_attention_heads=2,
        intermediate_size=64,
        max_position_embeddings=64,
    )
    prompt_config = PromptConfig(
        num_tasks=2,
        prompt_layer=3,
        prompt_len=5,
    )
    torch.manual_seed(1234)
    embedder = LayerWisePromptEmbedder.from_random_bert(bert_config, prompt_config)
    embedder.set_task(0)

    input_ids = torch.randint(0, bert_config.vocab_size, (batch_size, seq_len))
    attention_mask = torch.ones(batch_size, seq_len, dtype=torch.long)
    output = embedder(input_ids=input_ids, attention_mask=attention_mask)

    encoder_params = sum(param.numel() for param in embedder.model.parameters())
    prompt_params = embedder.prompt_embeddings.numel()
    trainable_params = sum(param.numel() for param in embedder.parameters() if param.requires_grad)
    prompt_pct = 100.0 * prompt_params / encoder_params

    print(f"input_ids shape: {list(input_ids.shape)}")
    print(f"attention_mask shape: {list(attention_mask.shape)}")
    print(f"output shape: {list(output.shape)}")
    print(f"encoder params: {encoder_params:,}")
    print(f"prompt params: {prompt_params:,}")
    print(f"trainable params: {trainable_params:,}")
    print(f"prompt/encoder params: {prompt_pct:.2f}%")

    assert list(output.shape) == [batch_size, hidden_size], output.shape
    assert trainable_params >= prompt_params, (trainable_params, prompt_params)

    for name, param in embedder.named_parameters():
        if name.startswith("query_model.") or name.startswith("passage_model."):
            assert not param.requires_grad, f"{name} should be frozen"
        else:
            assert param.requires_grad, f"{name} should require grad"

    combo_outputs = {}
    base_state_dict = embedder.state_dict()
    for use_ilf in (False, True):
        for use_cef in (False, True):
            combo_outputs[(use_ilf, use_cef)] = _run_single_combo_test(
                base_state_dict,
                use_ilf,
                use_cef,
                input_ids,
                attention_mask,
            )

    ff = combo_outputs[(False, False)]
    ft = combo_outputs[(False, True)]
    tf = combo_outputs[(True, False)]
    tt = combo_outputs[(True, True)]

    ff_ft_match = torch.allclose(ff, ft)
    ff_tf_match = torch.allclose(ff, tf)
    ff_tt_match = torch.allclose(ff, tt)

    print(f"comparison (False, False) vs (False, True): allclose={ff_ft_match}")
    print(f"comparison (False, False) vs (True, False): allclose={ff_tf_match}")
    print(f"comparison (False, False) vs (True, True): allclose={ff_tt_match}")

    assert ff_ft_match, "Expected use_cef=True to have no effect when use_ilf=False"
    assert ff_tf_match, "Expected use_ilf=True alone to have no effect without use_cef"
    assert not ff_tt_match, "Expected combined use_ilf=True and use_cef=True to change output"

    text_tokenizer = TinyTextTokenizer(vocab_size=bert_config.vocab_size)
    state_config = PromptConfig(
        num_tasks=2,
        prompt_layer=2,
        prompt_len=5,
        use_ilf=True,
        use_cef=True,
    )
    torch.manual_seed(4321)
    state_embedder = LayerWisePromptEmbedder(
        model_name="random-bert",
        prompt_config=state_config,
        use_prompt=True,
        query_model=BertModel(bert_config),
        passage_model=BertModel(bert_config),
        query_tokenizer=text_tokenizer,
        passage_tokenizer=text_tokenizer,
    )
    state_embedding, aggregated_state = state_embedder.encode_for_training_with_state(
        ["first text", "second text"],
        normalize=True,
    )
    print(f"state embedding shape: {list(state_embedding.shape)}")
    print(f"aggregated state shape: {None if aggregated_state is None else list(aggregated_state.shape)}")
    print(f"aggregated state requires_grad: {None if aggregated_state is None else aggregated_state.requires_grad}")
    assert aggregated_state is not None, "Expected aggregated state with ILF+CEF enabled"
    assert list(aggregated_state.shape) == [2, hidden_size], aggregated_state.shape
    assert aggregated_state.requires_grad, "Training aggregated state should require grad"

    no_cef_config = PromptConfig(
        num_tasks=2,
        prompt_layer=2,
        prompt_len=5,
        use_ilf=True,
        use_cef=False,
    )
    torch.manual_seed(4321)
    no_cef_embedder = LayerWisePromptEmbedder(
        model_name="random-bert",
        prompt_config=no_cef_config,
        use_prompt=True,
        query_model=BertModel(bert_config),
        passage_model=BertModel(bert_config),
        query_tokenizer=text_tokenizer,
        passage_tokenizer=text_tokenizer,
    )
    _, no_cef_state = no_cef_embedder.encode_for_training_with_state(
        ["first text", "second text"],
        normalize=True,
    )
    print(f"aggregated state with use_cef=False: {no_cef_state}")
    assert no_cef_state is None, "Expected no aggregated state when CEF is disabled"
    print("PASS: encode_with_state standalone checks succeeded")

    return True


if __name__ == "__main__":
    try:
        _run_standalone_test()
        print("PASS: prompt embedder standalone test succeeded")
    except Exception as exc:
        print(f"FAIL: prompt embedder standalone test failed: {exc}")
        raise
