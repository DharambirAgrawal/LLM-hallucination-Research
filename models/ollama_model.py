"""
OllamaModel
===========
Unified model backend for ALL Ollama models.

Works with any model from https://ollama.com/library — just put
the model tag in config.yaml and it runs.  No API keys, no GPU
driver setup — Ollama handles everything.

Features
--------
- Auto-pull:  optionally pulls the model if not installed
- Streaming:  uses Ollama's streaming API for faster first-token
- Chat API:   uses /api/chat so chat templates apply automatically
- Stochastic: sample_n() generates N diverse outputs for BERT checker
- Fallback:   graceful retries with exponential back-off
"""
from __future__ import annotations

import time
from typing import List

from loguru import logger

from models.base_model import BaseModel


class OllamaModel(BaseModel):
    """
    Wraps any Ollama model via the ollama Python client.

    Parameters
    ----------
    name     : display name (e.g. "llama3.1-8b")
    config   : dict from config.yaml models entry
      - model:      Ollama model tag  (e.g. "llama3.1:8b")
      - host:       Ollama server URL (default http://localhost:11434)
      - timeout:    request timeout in seconds (default 120)
      - auto_pull:  pull model if not present (default False)
      - temperature: default sampling temperature (default 0.7)
      - max_tokens:  max tokens to generate (default 512)
    """

    def __init__(self, name: str, config: dict, ollama_host: str = "http://localhost:11434"):
        super().__init__(name, config)
        self.model_tag   = config["model"]
        self.host        = config.get("host", ollama_host)
        self.timeout     = config.get("timeout", 120)
        self.auto_pull   = config.get("auto_pull", False)
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens  = config.get("max_tokens", 512)
        self.family      = config.get("family", "unknown")

        # Build ollama client pointed at the right host
        import ollama as _ollama
        self._client = _ollama.Client(host=self.host)

        # Optionally auto-pull
        if self.auto_pull:
            self._pull_if_missing()

    # ── public API ────────────────────────────────────────────

    def generate(self, prompt: str, **kwargs) -> str:
        """Generate a single response."""
        return self._chat(prompt, **kwargs)

    def generate_batch(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate responses for a list of prompts (sequential)."""
        return [self._chat(p, **kwargs) for p in prompts]

    def sample_n(self, prompt: str, n: int = 5, temperature: float = 1.0) -> List[str]:
        """
        Generate N stochastic samples (used by BERT stochastic checker).
        Uses high temperature for diversity.
        """
        return [
            self._chat(prompt, temperature=temperature)
            for _ in range(n)
        ]

    # ── internal ─────────────────────────────────────────────

    def _chat(self, prompt: str, **kwargs) -> str:
        """Single call via Ollama /api/chat endpoint."""
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens  = kwargs.get("max_tokens",  self.max_tokens)

        options = {
            "temperature": temperature,
            "num_predict": max_tokens,
            "stop":        ["\n\nHuman:", "\n\nUser:", "###"],
        }
        if "top_p" in kwargs:
            options["top_p"] = kwargs["top_p"]
        if "top_k" in kwargs:
            options["top_k"] = kwargs["top_k"]

        for attempt in range(3):
            try:
                response = self._client.chat(
                    model=self.model_tag,
                    messages=[{"role": "user", "content": prompt}],
                    options=options,
                    stream=False,
                )
                return response["message"]["content"].strip()

            except Exception as exc:
                wait = 2 ** attempt
                err  = str(exc)

                # Model not found — suggest pull
                if "not found" in err.lower() or "pull" in err.lower():
                    logger.error(
                        f"Model '{self.model_tag}' not found in Ollama.\n"
                        f"  Run:  ollama pull {self.model_tag}"
                    )
                    return ""

                logger.warning(
                    f"[{self.name}] Attempt {attempt+1}/3 failed: {err}. "
                    f"Retrying in {wait}s..."
                )
                time.sleep(wait)

        logger.error(f"[{self.name}] All retries exhausted for prompt.")
        return ""

    def _pull_if_missing(self):
        """Pull the model from Ollama registry if not already present."""
        try:
            models = self._client.list()
            installed = {m["model"] for m in models.get("models", [])}
            # Normalize: "llama3.1:8b" might appear as "llama3.1:8b" or similar
            tag = self.model_tag
            if not any(tag in m for m in installed):
                logger.info(f"  Pulling model: {tag} ...")
                for progress in self._client.pull(tag, stream=True):
                    status = progress.get("status", "")
                    if status in ("success", "pulling manifest"):
                        logger.info(f"    {status}")
                logger.info(f"  ✓ {tag} ready")
            else:
                logger.info(f"  ✓ {tag} already installed")
        except Exception as e:
            logger.warning(f"  Could not check/pull {self.model_tag}: {e}")

    # ── convenience ───────────────────────────────────────────

    @staticmethod
    def list_installed(host: str = "http://localhost:11434") -> List[str]:
        """Return list of all installed Ollama model tags."""
        import ollama as _ollama
        client = _ollama.Client(host=host)
        try:
            result = client.list()
            return [m["model"] for m in result.get("models", [])]
        except Exception as e:
            logger.error(f"Cannot connect to Ollama at {host}: {e}")
            return []

    def __repr__(self) -> str:
        return f"OllamaModel(name={self.name!r}, model={self.model_tag!r}, family={self.family!r})"
