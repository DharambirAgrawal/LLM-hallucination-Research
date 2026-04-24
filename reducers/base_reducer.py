"""Abstract base class for all hallucination reduction methods."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from models.base_model import BaseModel
    from data.datasets import BenchmarkSample


class BaseReducer(ABC):
    """
    Common interface for every reduction method.

    A reducer takes a question (and optionally context) and generates
    an answer using some strategy designed to reduce hallucinations.

    Each reducer must implement:
      - generate(sample, model) -> str
        Takes a BenchmarkSample and a model, returns the generated answer.

    The generated answer is then scored by the detectors in the runner.
    """

    def __init__(self, name: str, config: dict | None = None):
        self.name   = name
        self.config = config or {}

    # ── must implement ────────────────────────────────────────

    @abstractmethod
    def generate(self, sample: "BenchmarkSample", model: "BaseModel") -> str:
        """
        Generate an answer to the sample's question using this reducer's strategy.

        Parameters
        ----------
        sample : BenchmarkSample
            Contains question, context, and reference answers (reference not used
            as input - only the question and context are used by reducers).
        model : BaseModel
            The language model to use for generation.

        Returns
        -------
        str
            The generated answer, which will be scored by detectors.
        """

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name!r})"