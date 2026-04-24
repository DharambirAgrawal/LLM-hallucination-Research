"""
Constrained Decoding Reducer
=============================
Same question as baseline, but with tighter generation parameters.

By lowering temperature, top_p, and top_k, we force the model to
pick only its most confident, high-probability tokens. The goal
is to prevent the model from getting creative with its answers,
which often leads to hallucinations.

Parameters (compared to default):
  - temperature: 0.7 -> 0.1  (much less random)
  - top_p:       1.0 -> 0.3  (much narrower probability window)
  - top_k:       40  -> 5    (only top 5 candidate tokens)
"""
from __future__ import annotations

from typing import TYPE_CHECKING

from reducers.base_reducer import BaseReducer

if TYPE_CHECKING:
    from models.base_model import BaseModel
    from data.datasets import BenchmarkSample


class ConstrainedDecodingReducer(BaseReducer):
    """Asks the question with tight generation parameters to reduce randomness."""

    # Default parameters - override via config.yaml
    DEFAULT_TEMPERATURE = 0.1
    DEFAULT_TOP_P       = 0.3
    DEFAULT_TOP_K       = 5

    def __init__(self, config: dict | None = None):
        super().__init__(name="constrained_decoding", config=config)
        cfg = config or {}
        self.temperature = cfg.get("temperature", self.DEFAULT_TEMPERATURE)
        self.top_p       = cfg.get("top_p",       self.DEFAULT_TOP_P)
        self.top_k       = cfg.get("top_k",       self.DEFAULT_TOP_K)

    def generate(self, sample: "BenchmarkSample", model: "BaseModel") -> str:
        """Ask the question with tight parameters."""
        prompt = (
            "Answer the following question concisely and factually.\n\n"
            f"Question: {sample.question}\n\n"
            "Answer:"
        )
        return model.generate(
            prompt,
            temperature=self.temperature,
            top_p=self.top_p,
            top_k=self.top_k,
        )