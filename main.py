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
        "--reduce",
        action="store_true",
        help="Also run the configured reduction.method after detector validation",
    )
    parser.add_argument(
        "--max-samples", type=int,
        help="Override every dataset's max_samples (use a small number for a smoke run)",
    )
    parser.add_argument(
        "--n-samples", type=int,
        help="Override detectors.selfcheckgpt.n_samples (generations per case)",
    )
    parser.add_argument(
        "--max-iterations", type=int,
        help="Override reduction.max_iterations (feedback/refine steps)",
    )
    parser.add_argument(
        "--device", choices=("cpu", "cuda"),
        help="Override device for every torch-based detector (selfcheckgpt nli/bertscore, "
             "summac, alignscore). Ignored by selfcheckgpt's default ngram method, which "
             "needs no GPU. Ollama itself always uses the GPU on its own machine if present "
             "— this flag does not affect Ollama.",
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
    config.setdefault("reduction", {})
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
    if args.reduce:
        config["reduction"]["enabled"] = True
    if args.max_samples is not None:
        for dataset_cfg in config.get("datasets", []):
            dataset_cfg["max_samples"] = args.max_samples
    if args.n_samples is not None:
        config["detectors"].setdefault("selfcheckgpt", {})["n_samples"] = args.n_samples
    if args.max_iterations is not None:
        config["reduction"]["max_iterations"] = args.max_iterations
    if args.device:
        for name in ("selfcheckgpt", "summac", "alignscore"):
            config["detectors"].setdefault(name, {})["device"] = args.device

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

    # Every model listed in `selected_models` is used, not just the first —
    # the runner scores SelfCheckGPT once per generator and tags rows by model.
    generators = []
    if config["detectors"].get("selfcheckgpt", {}).get("enabled", False):
        generators = ModelFactory.build_all(config)
        if not generators:
            raise SystemExit(
                "SelfCheckGPT needs at least one reachable generator. Configure "
                "remote Ollama or an OpenAI-compatible endpoint."
            )
        print(f"Generators registered for SelfCheckGPT: {[g.name for g in generators]}")

    runner = BenchmarkRunner(config, generator=generators[0] if generators else None)
    raw = runner.validate(datasets, generators=generators or None)
    score_columns = list(runner.thresholds())
    if not score_columns:
        raise SystemExit("No official detector is enabled")

    usable_columns = [column for column in score_columns if raw[column].notna().any()]
    failed_columns = sorted(set(score_columns) - set(usable_columns))
    if failed_columns:
        print(f"No successful scores for: {', '.join(failed_columns)}")
    if not usable_columns:
        raise SystemExit("All enabled detectors failed; inspect detector_validation_raw.csv")

    thresholds = runner.thresholds()
    has_model_column = "model" in raw.columns and raw["model"].notna().any()
    summary_frames = []

    # SelfCheckGPT varies by generator, so report it once per model.
    if "selfcheckgpt_score" in usable_columns and has_model_column:
        for model_name, group in raw.groupby("model"):
            per_model = DetectorValidator().evaluate_frame(
                group, ["selfcheckgpt_score"],
                {"selfcheckgpt_score": thresholds["selfcheckgpt_score"]},
            )
            per_model["model"] = model_name
            summary_frames.append(per_model)
        remaining_columns = [c for c in usable_columns if c != "selfcheckgpt_score"]
    else:
        remaining_columns = usable_columns

    # Model-independent detectors (SummaC/MiniCheck/AlignScore) are duplicated
    # across model rows in `raw`; de-duplicate by case before scoring them once.
    if remaining_columns:
        dedup = raw.drop_duplicates(subset="case_id") if has_model_column else raw
        shared = DetectorValidator().evaluate_frame(
            dedup, remaining_columns, {c: thresholds[c] for c in remaining_columns},
        )
        shared["model"] = "n/a (model-independent detector)"
        summary_frames.append(shared)

    summary = pd.concat(summary_frames, ignore_index=True)
    output_dir = Path(config["benchmark"].get("output_dir", "results/current"))
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / "detector_validation_summary.csv"
    summary.to_csv(output, index=False)
    with pd.option_context("display.max_columns", None):
        print(summary.to_string(index=False))
    print(f"Saved: {output}")

    reduction_cfg = config.get("reduction", {})
    if reduction_cfg.get("enabled", False):
        detector_name = reduction_cfg.get("detector", "selfcheckgpt")
        if detector_name != "selfcheckgpt":
            raise SystemExit(
                "Reduction stage currently supports only detector: 'selfcheckgpt'"
            )
        detector = runner.detectors.get(detector_name)
        if detector is None:
            raise SystemExit(
                f"Reduction stage needs detector '{detector_name}' enabled in `detectors:`"
            )
        if not generators:
            raise SystemExit("Reduction stage needs at least one generator model")

        from benchmark.reduction_runner import ReductionRunner

        reduction_frame = ReductionRunner(config, detector).run(datasets, generators)
        reduction_output = output_dir / "reduction_comparison.csv"
        print(
            f"Reduction comparison: {len(reduction_frame)} paired rows across "
            f"{len(generators)} model(s). See {reduction_output}"
        )


if __name__ == "__main__":
    main()
