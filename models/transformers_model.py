"""Local Hugging Face Transformers generator for Colab/evaluator machines."""
from __future__ import annotations

import re
from typing import List

from models.base_model import BaseModel


class TransformersModel(BaseModel):
    """Lazily load an immutable Hugging Face causal-language-model revision."""

    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        self.model_id = config["model"]
        self.revision = config.get("revision")
        if not self.revision or not re.fullmatch(r"[0-9a-f]{40}", self.revision):
            raise ValueError(
                "A full immutable Hugging Face model revision is required"
            )
        self.max_tokens = int(config.get("max_tokens", 96))
        self.temperature = float(config.get("temperature", 1.0))
        self._model = None
        self._tokenizer = None
        self._device = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "Local Transformers generation requires torch and transformers"
            ) from exc

        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.float16 if self._device == "cuda" else torch.float32
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            revision=self.revision,
        )
        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            revision=self.revision,
            dtype=dtype,
            attn_implementation="eager",
        ).to(self._device)
        self._model.eval()

    def generate(self, prompt: str, **kwargs) -> str:
        self._load()
        import torch

        messages = [{"role": "user", "content": prompt}]
        inputs = self._tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._device)
        temperature = float(kwargs.get("temperature", self.temperature))
        generation = {
            "max_new_tokens": int(
                kwargs.get("max_tokens", kwargs.get("max_new_tokens", self.max_tokens))
            ),
            "do_sample": temperature > 0,
            "pad_token_id": self._tokenizer.eos_token_id,
        }
        if temperature > 0:
            generation["temperature"] = temperature
            generation["top_p"] = float(kwargs.get("top_p", 0.9))

        with torch.inference_mode():
            output_ids = self._model.generate(**inputs, **generation)
        new_tokens = output_ids[0, inputs["input_ids"].shape[-1]:]
        text = self._tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
        if not text:
            raise RuntimeError(f"Local model {self.name} returned an empty response")
        return text

    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        return [self.generate(prompt, **kwargs) for prompt in prompts]

    def sample_n(
        self, prompt: str, n: int = 5, temperature: float = 1.0
    ) -> List[str]:
        return [self.generate(prompt, temperature=temperature) for _ in range(n)]
