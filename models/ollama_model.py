"""
OllamaModel
===========
Unified model backend for ALL Ollama models.

Works with any model from https://ollama.com/library — just put
the model tag in config.yaml and it runs.  No API keys, no GPU
driver setup — Ollama handles everything.

Features
--------
- Pull:       OllamaModel.pull() downloads a missing model with a progress
              bar (models/model_factory.py decides when)
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
from typing import Callable, Dict, List, Optional

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
      - temperature: default sampling temperature (default 0.7)
      - max_tokens:  max tokens to generate (default 512)
      - think:       reasoning setting for thinking models (false, true,
                     or "low"/"medium"/"high" for gpt-oss); omitted if unset
      - num_ctx:     context window in tokens (ollama.num_ctx). Ollama's
                     default (4096) silently cuts longer prompts; the same
                     value must be used for every request to a model, or
                     Ollama reloads it.
    """

    _THINK_BLOCK = re.compile(r"<think>.*?(</think>|$)", re.DOTALL | re.IGNORECASE)
    # Ollama's wording when a model does not fit in GPU/RAM
    _NO_ROOM = ("resource limitations", "failed to load", "out of memory", "cudamalloc",
                "requires more system memory", "insufficient memory")

    # Set by the benchmark: frees GPU memory held in this process (moves the
    # torch detectors to the CPU) when a model does not fit.
    on_memory_pressure: Optional[Callable[[], None]] = None

    def __init__(self, name: str, config: dict, ollama_host: str = "http://localhost:11434"):
        super().__init__(name, config)
        self.model_tag   = config["model"]
        self.host        = config.get("host", ollama_host)
        self.timeout     = config.get("timeout", 120)
        self.temperature = config.get("temperature", 0.7)
        self.max_tokens  = config.get("max_tokens", 512)
        self.family      = config.get("family", "unknown")
        self.think       = config.get("think")
        self.digest      = config.get("digest")
        self.num_ctx     = config.get("num_ctx")
        self.parallel    = max(1, int(config.get("parallel") or 1))   # concurrent sample requests
        self.truncated   = 0     # prompts that may not have fit num_ctx (see _chat)

        # Build ollama client pointed at the right host. `timeout` bounds
        # every HTTP request so a stuck server cannot hang the run forever.
        import ollama as _ollama
        self._client = _ollama.Client(host=self.host, timeout=self.timeout)

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
        Uses high temperature for diversity. With ollama.parallel > 1 the N
        independent requests are sent concurrently; the Ollama server then
        answers them together if OLLAMA_NUM_PARALLEL allows it (otherwise it
        queues them, as before). Each sample is an independent draw either way.
        """
        if self.parallel < 2 or n < 2:
            return [self._chat(prompt, temperature=temperature) for _ in range(n)]
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=min(self.parallel, n)) as pool:
            return list(pool.map(lambda _: self._chat(prompt, temperature=temperature), range(n)))

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
        if self.num_ctx:
            options["num_ctx"] = int(self.num_ctx)
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
                if any(k in last_error.lower() for k in self._NO_ROOM):
                    self._make_room()
                if attempt == 2:
                    break
                wait = 5 * 2 ** attempt
                logger.debug(
                    f"[{self.name}] attempt {attempt + 1}/3 failed: "
                    f"{last_error}; retrying in {wait}s"
                )
                time.sleep(wait)
                continue

            self._check_fit(response, options["num_predict"])
            text = self._THINK_BLOCK.sub("", response["message"]["content"] or "").strip()
            if text:
                return text
            last_error = (
                "empty answer (a reasoning model may have spent all "
                f"{options['num_predict']} tokens thinking; raise max_tokens or set think)"
            )
            logger.debug(f"[{self.name}] attempt {attempt + 1}/3: {last_error}")
            # A reasoning model that thought until the budget ran out gets a
            # larger budget on the next attempt, instead of failing the same way.
            if self.think is not None:
                options["num_predict"] = int(options["num_predict"]) * 2

        raise RuntimeError(self._explain(last_error))

    def _check_fit(self, response, num_predict: int) -> None:
        """Warn when the prompt may not have fit the context window: Ollama
        then drops part of it without an error. prompt_eval_count leaves out
        a prefix Ollama reused from the previous request, so this can miss a
        cut but never reports one that did not happen."""
        if not self.num_ctx:
            return
        try:
            used = response.get("prompt_eval_count") if hasattr(response, "get") else None
            used = used if used is not None else getattr(response, "prompt_eval_count", None)
        except Exception:
            used = None
        if used and used + num_predict > int(self.num_ctx):
            self.truncated += 1
            logger.warning(f"[{self.name}] prompt of {used} tokens + {num_predict} answer tokens exceeds "
                           f"num_ctx={self.num_ctx}: part of the prompt or answer may be cut; raise "
                           "ollama.num_ctx in config.yaml")

    def prompt_tokens(self, prompt: str) -> Optional[int]:
        """Tokens this model's tokenizer gives `prompt` (one 1-token
        generation; used by the preflight to check the longest prompt fits)."""
        options = {"temperature": 0.0, "num_predict": 1}
        if self.num_ctx:
            options["num_ctx"] = int(self.num_ctx)
        extra = {"think": self.think} if self.think is not None else {}
        response = self._client.chat(model=self.model_tag, messages=[{"role": "user", "content": prompt}],
                                     options=options, stream=False, **extra)
        used = response.get("prompt_eval_count") if hasattr(response, "get") else None
        return used if used is not None else getattr(response, "prompt_eval_count", None)

    def _make_room(self) -> None:
        """The model did not fit: unload every other model Ollama holds (a
        previous generator, the judge) and let the benchmark free the GPU
        memory its detectors use, then the caller retries."""
        logger.warning(f"[{self.name}] {self.model_tag} did not fit in memory; unloading other "
                       "Ollama models and freeing detector GPU memory, then retrying")
        try:
            loaded = self._client.ps().get("models", [])
        except Exception as exc:
            logger.debug(f"[{self.name}] could not list loaded models: {exc}")
            loaded = []
        for entry in loaded:
            tag = entry.get("model") or entry.get("name")
            if tag and tag != self.model_tag:
                try:
                    self._client.generate(model=tag, prompt="", keep_alive=0)
                except Exception as exc:
                    logger.debug(f"[{self.name}] could not unload {tag}: {exc}")
        if self.on_memory_pressure is not None:
            self.on_memory_pressure()

    def _explain(self, error: str) -> str:
        """Turn Ollama's raw error into what happened and what to do."""
        low = error.lower()
        where = f"{self.name} ({self.model_tag}) failed 3 times"
        if "timed out" in low or "timeout" in low:
            return (f"{where}: no answer within {self.timeout}s. Large models can need minutes to load "
                    f"the first time; raise `ollama.timeout` in config.yaml. Last error: {error}")
        if any(k in low for k in self._NO_ROOM) or "memory" in low or "cuda" in low:
            return (f"{where}: out of memory, the model does not fit even after unloading the other "
                    f"Ollama models and moving the detectors to the CPU ({error}). Check `nvidia-smi` "
                    "for other programs using the GPU, or replace this model with a smaller one.")
        if "empty answer" in low:
            return (f"{where}: it returned no answer, even with a doubled token budget ({error}). A "
                    "reasoning model can spend its whole budget thinking on a long or hard prompt; "
                    "raise max_tokens for this model in config.yaml.")
        if "think" in low:
            return (f"{where}: this Ollama version does not accept `think` ({error}). Update Ollama "
                    "(0.9 or newer), or remove `think` from this model in config.yaml.")
        if "connect" in low or "refused" in low:
            return f"{where}: cannot reach Ollama at {self.host} ({error}). Is `ollama serve` running?"
        return f"{where}: {error}"

    def release(self) -> None:
        """Unload the model from Ollama's memory (GPU/RAM) now instead of
        after Ollama's idle timeout, so the next model and the detectors have
        room. Harmless if it fails; the model then unloads on its own."""
        try:
            self._client.generate(model=self.model_tag, prompt="", keep_alive=0)
        except Exception as exc:
            logger.debug(f"[{self.name}] could not unload: {exc}")

    # ── convenience ───────────────────────────────────────────

    @staticmethod
    def list_installed(host: str = "http://localhost:11434") -> List[str]:
        """Return list of all installed Ollama model tags."""
        return list(OllamaModel.installed_details(host))

    @staticmethod
    def server_reachable(host: str) -> bool:
        import ollama as _ollama
        try:
            _ollama.Client(host=host, timeout=10).list()
            return True
        except Exception:
            return False

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
            logger.debug(f"Cannot connect to Ollama at {host}: {e}")
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

    @staticmethod
    def pull(host: str, tag: str) -> None:
        """Download `tag` into the Ollama server with one progress bar.

        Ollama streams per-layer progress; the bar sums every layer so the
        whole model is one bar with one ETA.
        """
        import ollama as _ollama
        from utils import console

        client = _ollama.Client(host=host)
        totals: Dict[str, int] = {}
        done: Dict[str, int] = {}
        bar = console.download_bar(f"ollama pull {tag}", None)
        try:
            for update in client.pull(tag, stream=True):
                digest = update.get("digest")
                if digest and update.get("total"):
                    totals[digest] = update["total"]
                    done[digest] = update.get("completed") or 0
                    bar.total = sum(totals.values())
                    bar.n = sum(done.values())
                    bar.refresh()
                if update.get("status") == "success":
                    break
        finally:
            bar.close()

    def __repr__(self) -> str:
        return f"OllamaModel(name={self.name!r}, model={self.model_tag!r}, family={self.family!r})"


def unload_all(host: str) -> int:
    """Unload every model the Ollama server holds (best effort)."""
    try:
        import ollama as _ollama
        client = _ollama.Client(host=host, timeout=30)
        loaded = client.ps().get("models", [])
    except Exception as exc:
        logger.debug(f"Could not list loaded Ollama models: {exc}")
        return 0
    count = 0
    for entry in loaded:
        tag = entry.get("model") or entry.get("name")
        if not tag:
            continue
        try:
            client.generate(model=tag, prompt="", keep_alive=0)
            count += 1
        except Exception as exc:
            logger.debug(f"Could not unload {tag}: {exc}")
    return count
