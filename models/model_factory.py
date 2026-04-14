"""
ModelFactory
============
Builds OllamaModel instances from config.yaml.

- selected_models: []  → runs ALL models whose tags are installed in Ollama
- selected_models: ["llama3.2-3b", "mistral-7b"]  → runs only those two
- Any model on https://ollama.com/library works; just add it to config.yaml
"""
from __future__ import annotations

from typing import List

from loguru import logger

from models.base_model import BaseModel
from models.ollama_model import OllamaModel


class ModelFactory:

    @staticmethod
    def build_all(config: dict) -> List[BaseModel]:
        """
        Return a list of OllamaModel instances based on config.
        Skips models that are not installed (unless auto_pull=true).
        """
        ollama_cfg  = config.get("ollama", {})
        host        = ollama_cfg.get("host", "http://localhost:11434")
        timeout     = ollama_cfg.get("timeout", 120)
        selected    = set(config.get("selected_models") or [])

        # What's currently installed in Ollama
        installed      = OllamaModel.list_installed(host)
        installed_tags = set(installed)

        if not installed:
            logger.warning(
                "No models found in Ollama (or Ollama is not running).\n"
                "  Start Ollama:  ollama serve\n"
                "  Pull a model:  ollama pull llama3.2:3b"
            )

        models: List[BaseModel] = []

        for cfg in config.get("models", []):
            name = cfg["name"]
            tag  = cfg["model"]

            # Filter by selected_models if provided
            if selected and name not in selected:
                continue

            # Inject host/timeout into per-model config
            cfg = {**cfg, "host": host, "timeout": timeout}

            # Check if installed (skip unless auto_pull=true)
            tag_installed = any(tag in t or t in tag for t in installed_tags)
            if not tag_installed and not cfg.get("auto_pull", False):
                logger.warning(
                    f"  ⚠ Skipping '{name}' ({tag}) — not installed.\n"
                    f"     Run:  ollama pull {tag}"
                )
                continue

            try:
                m = OllamaModel(name=name, config=cfg, ollama_host=host)
                models.append(m)
                logger.info(f"  ✓ Registered: {name}  ({tag})  [{cfg.get('family','?')}]")
            except Exception as e:
                logger.error(f"  ✗ Failed to register '{name}': {e}")

        logger.info(f"\nModels ready to benchmark: {len(models)}")
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
