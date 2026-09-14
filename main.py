#!/usr/bin/env python3
"""Validate official hallucination detectors on fixed labeled responses."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
import yaml

from benchmark import BenchmarkRunner, DetectorValidator
from data.datasets import DatasetLoader
from models import ModelFactory


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", help="Override benchmark.output_dir")
    parser.add_argument(
        "--detectors",
        nargs="+",
        choices=("selfcheckgpt", "summac", "minicheck", "alignscore"),
        help="Enable only the named official detector adapters",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate configuration and data without importing detector packages",
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config.setdefault("benchmark", {})
    config.setdefault("detectors", {})
    return config


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if args.output:
        config["benchmark"]["output_dir"] = args.output
    if args.detectors:
        selected = set(args.detectors)
        for name in ("selfcheckgpt", "summac", "minicheck", "alignscore"):
            config["detectors"].setdefault(name, {})["enabled"] = name in selected

    datasets = DatasetLoader(
        config, seed=int(config["benchmark"].get("seed", 42))
    ).load_all()
    cases = sum(len(DatasetLoader.detection_cases(value)) for value in datasets.values())
    if not cases:
        raise SystemExit("No labeled detector-validation cases were loaded")

    enabled = [
        name for name, value in config["detectors"].items()
        if isinstance(value, dict) and value.get("enabled", False)
    ]
    print(f"Loaded {cases} fixed labeled cases; enabled detectors: {enabled or 'none'}")
    if args.dry_run:
        print("Dry run complete. No model, detector package, or checkpoint was loaded.")
        return

    generator = None
    if config["detectors"].get("selfcheckgpt", {}).get("enabled", False):
        models = ModelFactory.build_all(config)
        if not models:
            raise SystemExit(
                "SelfCheckGPT needs one reachable generator. Configure remote Ollama "
                "or an OpenAI-compatible endpoint."
            )
        generator = models[0]

    runner = BenchmarkRunner(config, generator=generator)
    raw = runner.validate(datasets)
    score_columns = list(runner.thresholds())
    if not score_columns:
        raise SystemExit("No official detector is enabled")

    usable_columns = [column for column in score_columns if raw[column].notna().any()]
    failed_columns = sorted(set(score_columns) - set(usable_columns))
    if failed_columns:
        print(f"No successful scores for: {', '.join(failed_columns)}")
    if not usable_columns:
        raise SystemExit("All enabled detectors failed; inspect detector_validation_raw.csv")

    summary = DetectorValidator().evaluate_frame(
        raw,
        usable_columns,
        {column: runner.thresholds()[column] for column in usable_columns},
    )
    output_dir = Path(config["benchmark"].get("output_dir", "results/current"))
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "detector_validation_summary.csv"
    summary.to_csv(output, index=False)
    with pd.option_context("display.max_columns", None):
        print(summary.to_string(index=False))
    print(f"Saved: {output}")


if __name__ == "__main__":
    main()
