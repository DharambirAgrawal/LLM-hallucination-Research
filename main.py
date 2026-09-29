#!/usr/bin/env python3
"""Validate official hallucination detectors on fixed labeled responses."""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import yaml
from loguru import logger

from utils import console

DETECTOR_NAMES = ("selfcheckgpt", "summac", "minicheck", "alignscore")
MODEL_INDEPENDENT = "n/a (model-independent detector)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", help="Override benchmark.output_dir")
    parser.add_argument(
        "--detectors",
        nargs="+",
        choices=DETECTOR_NAMES,
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


def apply_overrides(config: dict, args: argparse.Namespace) -> None:
    if args.output:
        config["benchmark"]["output_dir"] = args.output
    if args.detectors:
        selected = set(args.detectors)
        for name in DETECTOR_NAMES:
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


def describe_detector(name: str, cfg: dict) -> str:
    if name == "selfcheckgpt":
        return (f"{cfg.get('method', 'ngram')} · n_samples={cfg.get('n_samples', 5)} · "
                f"temperature={cfg.get('temperature', 1.0)} · threshold={cfg.get('threshold', 0.5)}")
    parts = [f"model={cfg.get('model_name', 'default')}", f"threshold={cfg.get('threshold', 0.5)}"]
    if "device" in cfg:
        parts.append(f"device={cfg['device']}")
    return " · ".join(parts)


def print_table(frame: pd.DataFrame) -> None:
    text = frame.to_string(index=False, float_format=lambda v: f"{v:.3f}", na_rep="—")
    for row in text.splitlines():
        console.line(row)


def summarise(raw: pd.DataFrame, runner) -> pd.DataFrame:
    from benchmark import DetectorValidator

    thresholds = runner.thresholds()
    score_columns = list(thresholds)
    has_model_column = "model" in raw.columns and raw["model"].notna().any()
    frames = []

    # SelfCheckGPT varies by generator, so report it once per model.
    if "selfcheckgpt_score" in score_columns and has_model_column:
        for model_name, group in raw.groupby("model", sort=False):
            if group["selfcheckgpt_score"].notna().any():
                per_model = DetectorValidator().evaluate_frame(
                    group, ["selfcheckgpt_score"],
                    {"selfcheckgpt_score": thresholds["selfcheckgpt_score"]},
                )
                per_model["model"] = model_name
                frames.append(per_model)
    other = [c for c in score_columns if c != "selfcheckgpt_score" and raw[c].notna().any()]

    # Model-independent detectors are duplicated across model rows in `raw`;
    # de-duplicate by case before scoring them once.
    if other:
        dedup = raw.drop_duplicates(subset="case_id") if has_model_column else raw
        shared = DetectorValidator().evaluate_frame(
            dedup, other, {c: thresholds[c] for c in other},
        )
        shared["model"] = MODEL_INDEPENDENT
        frames.append(shared)

    if not frames:
        return pd.DataFrame()
    summary = pd.concat(frames, ignore_index=True)
    leading = ["detector", "model", "n_cases", "n_failed", "threshold"]
    return summary[leading + [c for c in summary.columns if c not in leading]]


def main() -> None:
    args = parse_args()
    started_at = datetime.now(timezone.utc)
    started = time.monotonic()
    stage_seconds: dict[str, float] = {}

    config = load_config(args.config)
    apply_overrides(config, args)
    output_dir = Path(config["benchmark"].get("output_dir", "results/current"))
    log_file = None if args.dry_run else output_dir / "run.log"
    console.setup_logging(config.get("logging", {}).get("level", "INFO"), log_file)
    logger.debug("Command: {}", " ".join(sys.argv))

    enabled = [
        name for name in DETECTOR_NAMES
        if config["detectors"].get(name, {}).get("enabled", False)
    ]
    selfcheck_on = "selfcheckgpt" in enabled
    reduce_on = bool(config["reduction"].get("enabled", False))

    console.header("LLM hallucination benchmark" + (" · dry run" if args.dry_run else ""))
    console.kv("config", args.config)
    if not args.dry_run:
        console.kv("output", output_dir)
        console.kv("full log", log_file)
    console.kv("stages", "detector validation" + (" → reduction (self_refine_adapted)" if reduce_on else ""))

    # ── Data ────────────────────────────────────────────────────────────
    from data.datasets import DatasetLoader

    console.section("Data")
    seed = int(config["benchmark"].get("seed", 42))
    datasets = DatasetLoader(config, seed=seed).load_all()
    wanted = [d["name"] for d in config.get("datasets", []) if d.get("enabled", True)]
    missing = [name for name in wanted if name not in datasets]
    if missing:
        raise SystemExit(
            f"\nDataset(s) failed to load: {', '.join(missing)} (reason above). "
            "Fix them or set `enabled: false` in the config; a run on partial "
            "data would be mislabeled."
        )
    n_samples = n_cases = 0
    for name, samples in datasets.items():
        cases = DatasetLoader.detection_cases(samples)
        n_samples += len(samples)
        n_cases += len(cases)
        console.kv(name, f"{len(samples):>4} samples → {len(cases):>4} labeled cases", width=24)
    console.kv("total", f"{n_samples:>4} samples → {n_cases:>4} labeled cases "
               f"(half factual, half hallucinated) · seed {seed}", width=24)
    if not n_cases:
        raise SystemExit("No labeled detector-validation cases were loaded")

    # ── Detectors ───────────────────────────────────────────────────────
    console.section("Detectors")
    if not enabled:
        console.line("none enabled (use --detectors ...)")
    for name in enabled:
        console.kv(name, describe_detector(name, config["detectors"][name]))

    if args.dry_run:
        console.section("Dry run complete")
        console.line("No model, detector package, or checkpoint was loaded.")
        return
    if not enabled:
        raise SystemExit("No official detector is enabled")

    if any(name in enabled for name in ("minicheck", "summac", "alignscore")):
        from detectors.nltk_resources import ensure_sentence_tokenizer
        ensure_sentence_tokenizer()

    # ── Generators ──────────────────────────────────────────────────────
    generators = []
    if selfcheck_on:
        from models import ModelFactory

        console.section("Generators (SelfCheckGPT sampling" + (" + reduction)" if reduce_on else ")"))
        generators = ModelFactory.build_all(config)
        if not generators:
            raise SystemExit(
                "SelfCheckGPT needs at least one reachable generator. Configure "
                "remote Ollama or an OpenAI-compatible endpoint."
            )
        for g in generators:
            extras = [x for x in (getattr(g, "parameter_size", None),
                                  getattr(g, "quantization", None)) if x]
            if g.config.get("think") is not None:
                extras.append(f"think={g.config['think']}")
            console.kv(g.name, f"{g.config.get('model')}" + (f"  ({', '.join(extras)})" if extras else ""),
                       width=18)
        requested = len(config.get("selected_models") or config.get("models", []))
        if len(generators) < requested:
            console.line(f"⚠ {len(generators)} of {requested} selected models are available")

        n_per_prompt = int(config["detectors"]["selfcheckgpt"].get("n_samples", 5))
        console.section("Workload")
        console.kv("validation", f"{n_samples} prompts × {n_per_prompt} samples × "
                   f"{len(generators)} models = {n_samples * n_per_prompt * len(generators):,} generations")
        if reduce_on:
            iters = int(config["reduction"].get("max_iterations", 3))
            upper = n_samples * len(generators) * (1 + 2 * iters)
            console.kv("reduction", f"{n_samples} samples × {len(generators)} models × "
                       f"2–{1 + 2 * iters} calls = up to {upper:,} generations "
                       "(scoring reuses the validation samples)")
        console.line("Each progress bar shows elapsed<remaining time for that model.")

    # ── Stage 1: detector validation ────────────────────────────────────
    from benchmark import BenchmarkRunner

    total_stages = 2 if reduce_on else 1
    console.section(f"Stage 1/{total_stages} · Detector validation")
    stage_start = time.monotonic()
    runner = BenchmarkRunner(config, generator=generators[0] if generators else None)
    raw = runner.validate(datasets, generators=generators or None)
    stage_seconds["detector_validation"] = time.monotonic() - stage_start

    summary = summarise(raw, runner)
    if summary.empty:
        raise SystemExit(f"All enabled detectors failed; see {log_file} and detector_validation_raw.csv")
    summary_path = output_dir / "detector_validation_summary.csv"
    summary.to_csv(summary_path, index=False)

    console.section("Detector validation results")
    shown = summary.rename(columns={"average_precision": "auprc", "roc_auc": "auroc"})
    shown["model"] = shown["model"].replace(MODEL_INDEPENDENT, "—")
    print_table(shown[["detector", "model", "n_cases", "n_failed", "auroc", "auprc",
                       "accuracy", "precision", "recall", "f1", "threshold"]])
    console.line("auroc/auprc are threshold-free; the other columns use the threshold shown.")

    # ── Stage 2: reduction ──────────────────────────────────────────────
    selfcheck = runner.detectors.get("selfcheckgpt")
    if reduce_on:
        detector_name = config["reduction"].get("detector", "selfcheckgpt")
        if detector_name != "selfcheckgpt":
            raise SystemExit("Reduction stage currently supports only detector: 'selfcheckgpt'")
        if selfcheck is None:
            raise SystemExit(
                f"Reduction stage needs detector '{detector_name}' enabled in `detectors:`"
            )

        from benchmark.reduction_runner import ReductionRunner

        console.section(f"Stage 2/2 · Reduction (self_refine_adapted, "
                        f"max_iterations={config['reduction'].get('max_iterations', 3)})")
        stage_start = time.monotonic()
        reduction = ReductionRunner(config, selfcheck).run(datasets, generators)
        stage_seconds["reduction"] = time.monotonic() - stage_start

        ok = reduction[reduction["error"].isna()] if "error" in reduction else reduction.iloc[0:0]
        console.section("Reduction results (SelfCheckGPT score, lower = less hallucinated)")
        if ok.empty:
            logger.warning("No reduction row succeeded; see run.log")
        else:
            table = ok.groupby("model", sort=False).agg(
                n_ok=("score_delta", "count"),
                baseline=("baseline_score", "mean"),
                refined=("refined_score", "mean"),
                mean_delta=("score_delta", "mean"),
                win_rate=("score_delta", lambda s: (s < 0).mean()),
            ).reset_index()
            table.insert(2, "n_failed", [
                int((reduction["model"] == m).sum()) - int(n) for m, n in zip(table["model"], table["n_ok"])
            ])
            print_table(table)
            console.line("win_rate = share of samples whose refined answer scored lower than the baseline.")

    # ── Archive + report ────────────────────────────────────────────────
    # Same outputs for a 2-sample smoke run and a full run: only the data
    # size differs.
    if selfcheck is not None:
        selfcheck.export_samples(output_dir / "selfcheckgpt_samples.jsonl")
    from utils.run_manifest import write_run_files
    write_run_files(output_dir, config, sys.argv, datasets, generators, started_at, stage_seconds)

    from scripts.generate_report import generate
    try:
        produced = generate(output_dir)
    except Exception as exc:
        logger.opt(exception=exc).error(f"Report generation failed: {exc}")
        produced = {}

    console.section(f"Done in {console.duration(time.monotonic() - started)}")
    console.line(f"All files are in {output_dir}/")
    console.kv("report", produced.get("report", "not written (see run.log)"))
    console.kv("charts", produced.get("charts", "—"))
    console.kv("summary", summary_path)
    console.kv("raw scores", output_dir / "detector_validation_raw.csv")
    if reduce_on:
        console.kv("reduction", output_dir / "reduction_comparison.csv")
    for label in ("per-dataset", "reduction summary"):
        if label in produced:
            console.kv(label, produced[label])
    if selfcheck is not None:
        console.kv("samples", output_dir / "selfcheckgpt_samples.jsonl")
    console.kv("manifest", output_dir / "run_manifest.json")
    console.kv("full log", log_file)


if __name__ == "__main__":
    main()
