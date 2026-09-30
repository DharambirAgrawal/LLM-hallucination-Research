"""Abstract base class for all models."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List


class BaseModel(ABC):
    """Common interface for every model backend."""

    def __init__(self, name: str, config: dict):
        self.name   = name
        self.config = config

    # ── must implement ────────────────────────────────────────

    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a single response given a prompt."""

    @abstractmethod
    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate responses for a list of prompts."""

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"
