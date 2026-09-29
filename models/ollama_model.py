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
- Chat API:   uses /api/chat so chat templates apply automatically
- Stochastic: sample_n() generates N diverse outputs for SelfCheckGPT
- Reasoning:  `think` is passed through and any <think>...</think> block is
              removed, so only the final answer is scored
- Failures:   retried with exponential back-off, then raised — a failed
              call is never returned as an empty answer
"""
from __future__ import annotations

import re
import time
from typing import Dict, List

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
      - think:       reasoning setting for thinking models (false, true,
                     or "low"/"medium"/"high" for gpt-oss); omitted if unset
    """

    _THINK_BLOCK = re.compile(r"<think>.*?(</think>|$)", re.DOTALL | re.IGNORECASE)

    def __init__(self, name: str, config: dict, ollama_host: str = "http://localhost:11434"):
        super().__init__(name, config)
        self.model_tag   = config["model"]
        self.host        = config.get("host", ollama_host)
        self.timeout     = config.get("timeout", 120)
        self.auto_pull   = config.get("auto_pull", False)
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens  = config.get("max_tokens", 512)
        self.family      = config.get("family", "unknown")
        self.think       = config.get("think")
        self.digest      = config.get("digest")

        # Build ollama client pointed at the right host. `timeout` bounds
        # every HTTP request so a stuck server cannot hang the run forever.
        import ollama as _ollama
        self._client = _ollama.Client(host=self.host, timeout=self.timeout)

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
        Generate N stochastic samples (used by SelfCheckGPT).
        Uses high temperature for diversity.
        """
        return [
            self._chat(prompt, temperature=temperature)
            for _ in range(n)
        ]

    # ── internal ─────────────────────────────────────────────

    def _chat(self, prompt: str, **kwargs) -> str:
        """Single call via Ollama /api/chat endpoint.

        Raises RuntimeError when the model cannot answer, so callers record
        a failure instead of silently scoring an empty string.
        """
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens  = kwargs.get("max_tokens",  self.max_tokens)

        options = {
            "temperature": temperature,
            "num_predict": max_tokens,
            "stop":        ["\n\nHuman:", "\n\nUser:"],
        }
        if "top_p" in kwargs:
            options["top_p"] = kwargs["top_p"]
        if "top_k" in kwargs:
            options["top_k"] = kwargs["top_k"]
        extra = {"think": self.think} if self.think is not None else {}

        last_error = "unknown error"
        for attempt in range(3):
            try:
                response = self._client.chat(
                    model=self.model_tag,
                    messages=[{"role": "user", "content": prompt}],
                    options=options,
                    stream=False,
                    **extra,
                )
            except Exception as exc:
                last_error = str(exc).strip() or type(exc).__name__
                if "not found" in last_error.lower():
                    raise RuntimeError(
                        f"Model '{self.model_tag}' not found in Ollama "
                        f"(run: ollama pull {self.model_tag})"
                    ) from exc
                wait = 2 ** attempt
                logger.debug(
                    f"[{self.name}] attempt {attempt + 1}/3 failed: "
                    f"{last_error}; retrying in {wait}s"
                )
                time.sleep(wait)
                continue

            text = self._THINK_BLOCK.sub("", response["message"]["content"] or "").strip()
            if text:
                return text
            last_error = (
                "empty answer (a reasoning model may have spent all "
                f"{max_tokens} tokens thinking; raise max_tokens or set think)"
            )
            logger.debug(f"[{self.name}] attempt {attempt + 1}/3: {last_error}")

        raise RuntimeError(f"{self.name}: {last_error}")

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
        return list(OllamaModel.installed_details(host))

    @staticmethod
    def installed_details(host: str = "http://localhost:11434") -> Dict[str, dict]:
        """Installed tags mapped to their digest, size and quantization.

        The digest identifies the exact weights that produced a run; it is
        written to run_manifest.json.
        """
        import ollama as _ollama
        client = _ollama.Client(host=host, timeout=10)
        try:
            result = client.list()
        except Exception as e:
            logger.error(f"Cannot connect to Ollama at {host}: {e}")
            return {}
        details = {}
        for m in result.get("models", []):
            info = m.get("details") or {}
            details[m["model"]] = {
                "digest": m.get("digest"),
                "parameter_size": info.get("parameter_size"),
                "quantization_level": info.get("quantization_level"),
            }
        return details

    def __repr__(self) -> str:
        return f"OllamaModel(name={self.name!r}, model={self.model_tag!r}, family={self.family!r})"
