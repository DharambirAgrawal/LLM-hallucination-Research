"""Stochastic samples shared by every sampling-based detector.

SelfCheckGPT and the UQLM consistency scorers both judge an answer by its
agreement with other answers the generator gives to the same prompt. The
samples depend only on (generator, question, context), never on the answer
being checked, so they are drawn once and reused: by every sampling-based
detector, for every labeled answer to that question (Stage A), and for every
reduction condition's answer (Stage B). Paired comparisons are therefore made
against identical evidence. Every sample set is exported to JSONL.
"""
from __future__ import annotations

import json
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, List, Tuple

from models.prompts import grounded_prompt


class SampleBank:
    def __init__(self, n_samples: int = 5, temperature: float = 1.0):
        self.n_samples = n_samples
        self.temperature = temperature
        self._samples: Dict[Tuple[str, str], List[str]] = {}

    def get(self, generator, question: str, context: str) -> List[str]:
        prompt = grounded_prompt(question, context)
        key = (getattr(generator, "name", repr(generator)), prompt)
        if key not in self._samples:
            if hasattr(generator, "sample_n"):
                samples = generator.sample_n(prompt, n=self.n_samples, temperature=self.temperature)
            else:
                samples = generator.generate_batch([prompt] * self.n_samples, temperature=self.temperature)
            samples = [s for s in samples if s.strip()]
            if not samples:
                raise RuntimeError("Generator returned no samples")
            self._samples[key] = samples
        return self._samples[key]

    @contextmanager
    def evidence(self, generator, question: str, context: str, samples: List[str]):
        """Score against `samples` instead of the stored ones for one block
        (used for leave-one-out scoring of an answer picked from the samples)."""
        key = (getattr(generator, "name", repr(generator)), grounded_prompt(question, context))
        stored = self._samples.get(key)
        self._samples[key] = samples
        try:
            yield
        finally:
            if stored is None:
                self._samples.pop(key, None)
            else:
                self._samples[key] = stored

    def reset(self) -> None:
        """Forget drawn samples so the next run samples independently."""
        self._samples.clear()

    def export(self, path: Path) -> int:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for (model_name, prompt), samples in self._samples.items():
                handle.write(json.dumps({
                    "model": model_name,
                    "temperature": self.temperature,
                    "n_samples": len(samples),
                    "prompt": prompt,
                    "samples": samples,
                }, ensure_ascii=False) + "\n")
        return len(self._samples)
