"""
ModelFactory
============
Builds local Ollama or remote OpenAI-compatible models from config.yaml.

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
from models.openai_compatible_model import OpenAICompatibleModel
from models.transformers_model import TransformersModel


class ModelFactory:

    @staticmethod
    def build_all(config: dict) -> List[BaseModel]:
        """
        Return model adapters based on config. Local Ollama models are checked
        for availability; remote endpoints are registered without making a
        network call until generation starts.
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

            if provider == "openai_compatible":
                try:
                    models.append(OpenAICompatibleModel(name=name, config=cfg))
                    logger.debug(f"Registered remote endpoint: {name}")
                except Exception as e:
                    logger.error(f"Cannot register '{name}': {e}")
                continue
            if provider == "transformers":
                try:
                    models.append(TransformersModel(name=name, config=cfg))
                    logger.debug(f"Registered local Transformers model: {name}")
                except Exception as e:
                    logger.error(f"Cannot register '{name}': {e}")
                continue
            if provider != "ollama":
                logger.error(f"Unknown model provider '{provider}' for '{name}'")
                continue

            # Exact tag match only ("llama3" also matches "llama3:latest").
            # A substring match could register a tag Ollama cannot serve.
            installed_tag = next(
                (t for t in (tag, f"{tag}:latest") if t in installed), None
            )
            if installed_tag is None and not cfg.get("auto_pull", False):
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

    # ── interactive helper ────────────────────────────────────

    @staticmethod
    def show_available(host: str = "http://localhost:11434"):
        """Print all installed Ollama models in a rich table."""
        from rich.console import Console
        from rich.table import Table
        from rich import box

        installed = OllamaModel.list_installed(host)
        c = Console()

        if not installed:
            c.print("[red]No Ollama models installed or Ollama is not running.[/red]")
            c.print(f"  Start: [cyan]ollama serve[/cyan]")
            c.print(f"  Pull:  [cyan]ollama pull llama3.2:3b[/cyan]")
            return

        t = Table(title="Installed Ollama Models", box=box.ROUNDED, header_style="bold cyan")
        t.add_column("#",         justify="right", style="dim")
        t.add_column("Model Tag", style="green")
        for i, tag in enumerate(sorted(installed), 1):
            t.add_row(str(i), tag)
        c.print(t)
        c.print(f"\n[dim]Total: {len(installed)} models[/dim]")
