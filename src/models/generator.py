"""Generator model wrapper for FlowRAG supervision and evaluation."""

from __future__ import annotations

from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config, GPT2LMHeadModel


class HFGenerator:
    def __init__(
        self,
        model_name,
        max_new_tokens=64,
        use_vllm=False,
        device="cpu",
        model=None,
        tokenizer=None,
        load_in_4bit: bool = False,
    ):
        self.model_name = model_name
        self.max_new_tokens = max_new_tokens
        self.use_vllm = use_vllm
        self.load_in_4bit = load_in_4bit
        self.device = torch.device(device) if device is not None else torch.device("cpu")
        self.model = model
        self.tokenizer = tokenizer
        self.vllm_model = None

    def load(self):
        if self.use_vllm and not self.load_in_4bit and self.vllm_model is None and self.model is None:
            try:
                from vllm import LLM

                self.vllm_model = LLM(model=self.model_name)
                if self.tokenizer is None:
                    self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
                return self
            except ImportError:
                pass

        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if getattr(self.tokenizer, "pad_token_id", None) is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        if self.model is None:
            if self.load_in_4bit:
                from transformers import BitsAndBytesConfig

                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_quant_type="nf4",
                )
                self.model = AutoModelForCausalLM.from_pretrained(
                    self.model_name,
                    quantization_config=quantization_config,
                    device_map="auto",
                )
            else:
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name)

        if self.load_in_4bit:
            self.device = self.model.device
        else:
            self.model.to(self.device)
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        return self

    @torch.no_grad()
    def generate(self, prompts: List[str]) -> List[str]:
        self.load()
        if self.vllm_model is not None:
            from vllm import SamplingParams

            params = SamplingParams(max_tokens=self.max_new_tokens)
            outputs = self.vllm_model.generate(prompts, params)
            return [out.outputs[0].text.strip() for out in outputs]

        encoded = self.tokenizer(
            prompts,
            padding=True,
            return_tensors="pt",
        ).to(self.device)
        prompt_lengths = encoded["attention_mask"].sum(dim=1)
        output_ids = self.model.generate(
            **encoded,
            max_new_tokens=self.max_new_tokens,
            pad_token_id=self.tokenizer.pad_token_id,
        )

        completions = []
        for row, prompt_len in zip(output_ids, prompt_lengths):
            new_tokens = row[int(prompt_len.item()) :]
            completions.append(self.tokenizer.decode(new_tokens, skip_special_tokens=True).strip())
        return completions

    @torch.no_grad()
    def get_flowrag_score(
        self,
        queries: List[str],
        docs_list: List[List[str]],
        answers: List[str],
    ) -> torch.Tensor:
        self.load()
        if self.model is None:
            raise RuntimeError("get_flowrag_score requires a transformers causal LM")

        batch_scores = []
        for query, docs, answer in zip(queries, docs_list, answers):
            doc_scores = []
            for doc in docs:
                prompt = f"{doc}\n\nQuestion: {query}\nAnswer:"
                doc_scores.append(self._answer_log_likelihood(prompt, answer))
            batch_scores.append(torch.stack(doc_scores))
        return torch.stack(batch_scores, dim=0).detach()

    def _answer_log_likelihood(self, prompt: str, answer: str) -> torch.Tensor:
        prompt_ids = self.tokenizer(prompt, return_tensors="pt")["input_ids"].to(self.device)
        full = self.tokenizer(
            prompt + answer,
            return_tensors="pt",
            padding=True,
        ).to(self.device)
        input_ids = full["input_ids"]
        attention_mask = full["attention_mask"]

        outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :]
        labels = input_ids[:, 1:]
        shifted_mask = attention_mask[:, 1:].bool()

        prompt_len = prompt_ids.size(1)
        positions = torch.arange(labels.size(1), device=self.device).unsqueeze(0)
        answer_mask = positions >= (prompt_len - 1)
        answer_mask = answer_mask & shifted_mask

        log_probs = torch.log_softmax(logits, dim=-1)
        token_log_probs = log_probs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
        return token_log_probs.masked_select(answer_mask).sum()


def create_generator(model_name, max_new_tokens=64, use_vllm=False, device="cpu", load_in_4bit: bool = False):
    return HFGenerator(model_name, max_new_tokens, use_vllm, device, load_in_4bit=load_in_4bit)


class TinyCharTokenizer:
    def __init__(self, vocab_size: int = 100):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1
        self.eos_token = "<eos>"
        self.pad_token = "<pad>"

    def _encode(self, text: str):
        return [2 + (ord(ch) % (self.vocab_size - 2)) for ch in text]

    def __call__(self, texts, padding=False, return_tensors=None, **_):
        if isinstance(texts, str):
            texts = [texts]
        encoded = [self._encode(text) for text in texts]
        max_len = max(len(ids) for ids in encoded)
        if padding:
            padded = [ids + [self.pad_token_id] * (max_len - len(ids)) for ids in encoded]
            masks = [[1] * len(ids) + [0] * (max_len - len(ids)) for ids in encoded]
        else:
            padded = encoded
            masks = [[1] * len(ids) for ids in encoded]
        result = {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }
        return BatchEncodingLite(result)

    def decode(self, ids, skip_special_tokens=True):
        pieces = []
        for idx in ids:
            idx = int(idx)
            if skip_special_tokens and idx in {self.pad_token_id, self.eos_token_id}:
                continue
            pieces.append(chr((idx - 2) % 95 + 32))
        return "".join(pieces)


class BatchEncodingLite(dict):
    def to(self, device):
        return BatchEncodingLite({key: value.to(device) for key, value in self.items()})


def _run_standalone_test() -> bool:
    torch.manual_seed(7)
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
    model = GPT2LMHeadModel(config)
    tokenizer = TinyCharTokenizer(vocab_size=100)
    generator = HFGenerator(
        "random-gpt2",
        max_new_tokens=8,
        use_vllm=False,
        device="cpu",
        model=model,
        tokenizer=tokenizer,
    )

    queries = ["alpha?", "beta?"]
    docs_list = [
        ["doc one", "doc two", "doc three"],
        ["doc four", "doc five", "doc six"],
    ]
    answers = ["yes", "no"]

    scores = generator.get_flowrag_score(queries, docs_list, answers)
    changed_docs = [row[:] for row in docs_list]
    changed_docs[0][0] = "completely changed document text"
    changed_scores = generator.get_flowrag_score(queries, changed_docs, answers)

    print(f"scores shape: {list(scores.shape)}")
    print(f"scores requires_grad: {scores.requires_grad}")
    print(f"finite scores: {bool(torch.isfinite(scores).all())}")
    print(
        "changed doc score differs: "
        f"{not torch.allclose(scores[0, 0], changed_scores[0, 0])}"
    )

    assert list(scores.shape) == [2, 3], scores.shape
    assert torch.isfinite(scores).all(), scores
    assert not scores.requires_grad, "Generator scores must be computed under no_grad"
    assert not torch.allclose(scores[0, 0], changed_scores[0, 0]), (
        "Changing one doc's content should change that doc's score"
    )
    return True


if __name__ == "__main__":
    try:
        _run_standalone_test()
        print("PASS: HFGenerator standalone test succeeded")
    except Exception as exc:
        print(f"FAIL: HFGenerator standalone test failed: {exc}")
        raise
