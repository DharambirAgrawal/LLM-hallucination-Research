"""
ModelFactory
============
Builds the local Ollama models listed in config.yaml.

- selected_models: []  → every entry under `models:` (skipping Ollama tags
  that are not pulled)
- selected_models: ["llama3.2-3b", "qwen2.5-7b"]  → only those two
- Any model on https://ollama.com/library works; add it under `models:`
"""
from __future__ import annotations

from typing import List

from loguru import logger

from models.base_model import BaseModel
from models.ollama_model import OllamaModel


class ModelFactory:

    @staticmethod
    def active_configs(config: dict) -> List[dict]:
        selected = set(config.get("selected_models") or [])
        return [
            cfg for cfg in config.get("models", [])
            if not selected or cfg.get("name") in selected
        ]

    @staticmethod
    def ensure_ollama_models(config: dict, dry_run: bool = False,
                             include_generators: bool = True, extra_tags: List[str] = ()) -> None:
        """Pull every selected Ollama model that is not on the server yet.

        `ollama.auto_pull` (default true) or a model's own `auto_pull`
        turns this off; a model that is still missing is then skipped by
        build_all with a warning.
        """
        from utils import console

        ollama_cfg = config.get("ollama", {})
        host = ollama_cfg.get("host", "http://localhost:11434")
        wanted = [c for c in ModelFactory.active_configs(config)
                  if c.get("provider", "ollama") == "ollama"] if include_generators else []
        wanted += [{"name": "judge", "model": tag} for tag in extra_tags
                   if tag and tag not in {c["model"] for c in wanted}]
        if not wanted:
            return
        installed = OllamaModel.installed_details(host)
        if not installed and not OllamaModel.server_reachable(host):
            if dry_run:
                console.line(f"? Ollama at {host} is unavailable; model tags cannot be checked offline")
                return
            raise SystemExit(
                f"\nCannot reach Ollama at {host}. Start it (`ollama serve`) or "
                "fix `ollama.host` in the config."
            )
        for cfg in wanted:
            name, tag = cfg["name"], cfg["model"]
            if tag in installed or f"{tag}:latest" in installed:
                if dry_run:
                    info = installed.get(tag) or installed.get(f"{tag}:latest") or {}
                    console.line(f"✓ {name:<16} {tag}  ({info.get('parameter_size') or '?'}) present")
                continue
            if not cfg.get("auto_pull", ollama_cfg.get("auto_pull", True)):
                console.line(f"✗ {name:<16} {tag} is not pulled and auto_pull is off")
                continue
            if dry_run:
                console.line(f"↓ {name:<16} {tag} will be pulled from ollama.com")
                continue
            try:
                OllamaModel.pull(host, tag)
                console.line(f"✓ {name:<16} {tag} pulled")
            except Exception as exc:
                logger.warning(f"Could not pull {tag}: {exc}")

    @staticmethod
    def build_all(config: dict) -> List[BaseModel]:
        """
        Return an adapter for every selected model that the Ollama server has.
        """
        ollama_cfg  = config.get("ollama", {})
        host        = ollama_cfg.get("host", "http://localhost:11434")
        timeout     = ollama_cfg.get("timeout", 120)
        selected    = set(config.get("selected_models") or [])

        model_configs = config.get("models", [])
        active_configs = [
            model_cfg for model_cfg in model_configs
            if not selected or model_cfg.get("name") in selected
        ]
        needs_ollama = any(
            model_cfg.get("provider", "ollama") == "ollama"
            for model_cfg in active_configs
        )

        # Query Ollama only when an Ollama-backed model is configured.
        installed = OllamaModel.installed_details(host) if needs_ollama else {}

        if needs_ollama and not installed:
            logger.warning(
                f"No models found in Ollama at {host} (is `ollama serve` running?)"
            )

        models: List[BaseModel] = []

        for cfg in active_configs:
            name = cfg["name"]
            tag  = cfg["model"]
            provider = cfg.get("provider", "ollama")

            if provider != "ollama":
                logger.error(f"Unsupported model provider '{provider}' for '{name}' "
                             "(only local Ollama models are supported)")
                continue

            # Exact tag match only ("llama3" also matches "llama3:latest").
            # A substring match could register a tag Ollama cannot serve.
            installed_tag = next(
                (t for t in (tag, f"{tag}:latest") if t in installed), None
            )
            if installed_tag is None:
                if installed:
                    logger.warning(
                        f"Skipping {name}: '{tag}' is not pulled in Ollama "
                        f"(run: ollama pull {tag})"
                    )
                continue

            info = installed.get(installed_tag, {})
            cfg = {**cfg, "host": host, "timeout": timeout, "digest": info.get("digest")}
            try:
                m = OllamaModel(name=name, config=cfg, ollama_host=host)
                m.parameter_size = info.get("parameter_size")
                m.quantization = info.get("quantization_level")
                models.append(m)
                logger.debug(f"Registered {name} ({tag}) digest={info.get('digest')}")
            except Exception as e:
                logger.error(f"Cannot register '{name}': {e}")

        return models
