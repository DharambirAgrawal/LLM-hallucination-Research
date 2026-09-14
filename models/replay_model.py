"""Deterministic model adapter for replaying previously saved responses."""
from __future__ import annotations

from typing import List

from models.base_model import BaseModel


class ReplayModel(BaseModel):
    """Return supplied responses without calling or loading an LLM."""

    def __init__(self, responses: List[str], name: str = "replay"):
        if not responses:
            raise ValueError("ReplayModel requires at least one response")
        super().__init__(name=name, config={"model": name, "provider": "replay"})
        self.responses = responses

    def generate(self, prompt: str, **kwargs) -> str:
        return self.responses[0]

    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        return [self.responses[index % len(self.responses)] for index, _ in enumerate(prompts)]

    def sample_n(
        self, prompt: str, n: int = 5, temperature: float = 1.0
    ) -> List[str]:
        return [self.responses[index % len(self.responses)] for index in range(n)]
