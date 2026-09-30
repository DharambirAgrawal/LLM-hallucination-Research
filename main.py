#!/usr/bin/env python3
"""Run the benchmark in one Python environment.

Setup (download + verify everything) → preflight (one real question through
every detector, model and reduction method) → run_01 … run_N (Stage A:
detector validation on labeled answers; Stage B: reduction methods vs. the
baseline answer) → per-run and combined reports. scripts/run_full.py runs
this once per detector environment. Flow and diagrams: docs/ARCHITECTURE.md.
"""
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

DETECTOR_NAMES = ("selfcheckgpt", "uqlm", "uqlm_judge", "minicheck", "summac", "alignscore")
SAMPLING_DETECTORS = ("selfcheckgpt", "uqlm")      # need generator samples
JUDGE_DETECTORS = ("uqlm_judge",)                  # need the judge model
MODEL_INDEPENDENT = "n/a (model-independent detector)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--output", help="Override benchmark.output_dir")
    parser.add_argument(
        "--detectors",
        nargs="+",
        choices=DETECTOR_NAMES,
        help="Detectors to run (run.detectors)",
    )
    parser.add_argument(
        "--runs", type=int,
        help="Independent repeats (run.runs): run_01 … run_N, then combined/ with mean ± std",
    )
    parser.add_argument(
        "--reduce", action="store_true",
        help="Run the reduction stage (default: run.reduce)",
    )
    parser.add_argument(
        "--no-reduce", action="store_true",
        help="Skip the reduction stage even if run.reduce is true",
    )
    parser.add_argument(
        "--max-samples", type=int,
        help="Samples per dataset for every dataset (run.samples_per_dataset); small for a smoke run",
    )
    parser.add_argument(
        "--n-samples", type=int,
        help="SelfCheckGPT samples per question (run.selfcheckgpt_samples)",
    )
    parser.add_argument(
        "--max-iterations", type=int,
        help="Max feedback → refine rounds (run.reduction_iterations)",
    )
    parser.add_argument(
        "--device", choices=("cpu", "cuda"),
        help="Device for the torch-based detectors: SelfCheckGPT (BERTScore, NLI), UQLM "
             "(NLI, BERTScore, embeddings, best-response), SummaC, AlignScore. MiniCheck picks "
             "the GPU itself when one is present. Ollama always uses its own GPU; this flag "
             "does not affect it.",
    )
    parser.add_argument(
        "--score-reduction-from",
        help="Folder of a sampling-detector run (with run_XX/reduction_comparison.csv): "
             "also score its reduction answers with this run's detectors. Used by "
             "scripts/run_full.py for detectors that live in their own venv.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Report what is present and what would be downloaded; download "
             "and load nothing",
    )
    parser.add_argument(
        "--preflight",
        action="store_true",
        help="Download everything, run every check on one real case, print "
             "the time estimate, then stop (no long stage is started)",
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config.setdefault("benchmark", {})
    config.setdefault("detectors", {})
    config.setdefault("reduction", {})
    return config


def apply_plan(config: dict, args: argparse.Namespace) -> dict:
    """Resolve the run plan. Precedence:
      detectors, reduce, runs     flag > `run:` block
      questions per dataset       flag > the dataset's own max_samples > run.samples_per_dataset
      samples, Self-Refine rounds flag > their own section > `run:` block
    Writes the resolved values back into the sections the code reads, and
    returns the plan that is printed and saved."""
    plan = config.setdefault("run", {})
    detectors_cfg = config["detectors"]
    if args.output:
        config["benchmark"]["output_dir"] = args.output

    selected = (args.detectors or plan.get("detectors")
                or [n for n in DETECTOR_NAMES if detectors_cfg.get(n, {}).get("enabled")])
    for name in DETECTOR_NAMES:
        detectors_cfg.setdefault(name, {})["enabled"] = name in selected

    for ds in config.get("datasets", []):
        if args.max_samples is not None:
            ds["max_samples"] = args.max_samples
        else:
            ds.setdefault("max_samples", plan.get("samples_per_dataset", 50))

    sc = detectors_cfg["selfcheckgpt"]
    if args.n_samples is not None:
        sc["n_samples"] = args.n_samples
    sc.setdefault("n_samples", plan.get("selfcheckgpt_samples", 5))

    red = config["reduction"]
    if args.max_iterations is not None:
        red["max_iterations"] = args.max_iterations
    red.setdefault("max_iterations", plan.get("reduction_iterations", 3))
    # Reduction needs generators, which only the sampling-based detectors bring.
    red["enabled"] = (any(n in selected for n in SAMPLING_DETECTORS) and not args.no_reduce
                      and bool(args.reduce or plan.get("reduce", False)))

    if args.device:
        for name in ("selfcheckgpt", "uqlm", "summac", "alignscore"):
            detectors_cfg[name]["device"] = args.device

    resolved = {
        "runs": max(1, int(args.runs if args.runs is not None else plan.get("runs", 1))),
        "detectors": [n for n in DETECTOR_NAMES if n in selected],
        "samples_per_dataset": {ds["name"]: ds["max_samples"] for ds in config.get("datasets", [])
                                if ds.get("enabled", True)},
        "selfcheckgpt_samples": sc["n_samples"],
        "reduce": red["enabled"],
        "reduction_methods": list(red.get("methods") or []) if red["enabled"] else [],
        "reduction_iterations": red["max_iterations"],
    }
    config["run"] = resolved
    return resolved


def describe_detector(name: str, cfg: dict, judge: dict) -> str:
    if name == "selfcheckgpt":
        methods = cfg.get("methods") or [cfg.get("method", "ngram")]
        text = f"{', '.join(methods)} · temperature={cfg.get('temperature', 1.0)}"
        return text + (f" · prompt judge={judge.get('model')}" if "prompt" in methods else "")
    if name == "uqlm":
        return ", ".join(cfg.get("scorers") or ["default scorers"])
    if name == "uqlm_judge":
        return f"judge={judge.get('model')} · template={cfg.get('template', 'true_false_uncertain')}"
    parts = [f"model={cfg.get('model_name', 'default')}", f"threshold={cfg.get('threshold', 0.5)}"]
    if "device" in cfg:
        parts.append(f"device={cfg['device']}")
    return " · ".join(parts)


def print_table(frame: pd.DataFrame) -> None:
    text = frame.to_string(index=False, float_format=lambda v: f"{v:.3f}", na_rep="—")
    for row in text.splitlines():
        console.line(row)


def summarise(raw: pd.DataFrame, runner) -> pd.DataFrame:
    """Metrics per score column: per model for generator-dependent
    detectors, once per case for the others. A detector (or a detector for
    one model) that failed on every case still gets a row with n_cases 0 and
    no metrics, so nothing can silently disappear from the results."""
    from benchmark import DetectorValidator

    thresholds = runner.thresholds()
    gen_cols = [c for c in runner.generator_columns() if c in raw.columns]
    shared = [c for c in thresholds if c not in runner.generator_columns() and c in raw.columns]
    has_model_column = "model" in raw.columns and raw["model"].notna().any()

    def failed_row(column, model, n):
        return pd.DataFrame([{"detector": column.removesuffix("_score"), "model": model,
                              "n_cases": 0, "n_failed": int(n), "threshold": thresholds[column]}])

    frames = []
    if gen_cols and has_model_column:
        for model_name, group in raw.groupby("model", sort=False):
            usable = [c for c in gen_cols if group[c].notna().any()]
            if usable:
                part = DetectorValidator().evaluate_frame(group, usable, {c: thresholds[c] for c in usable})
                part["model"] = model_name
                frames.append(part)
            frames += [failed_row(c, model_name, len(group)) for c in gen_cols if c not in usable]
    # Generator-independent scores repeat on every model's rows in `raw`;
    # de-duplicate by case before scoring them once.
    if shared:
        dedup = raw.drop_duplicates(subset="case_id") if has_model_column else raw
        usable = [c for c in shared if dedup[c].notna().any()]
        if usable:
            part = DetectorValidator().evaluate_frame(dedup, usable, {c: thresholds[c] for c in usable})
            part["model"] = MODEL_INDEPENDENT
            frames.append(part)
        frames += [failed_row(c, MODEL_INDEPENDENT, len(dedup)) for c in shared if c not in usable]
    if not frames:
        return pd.DataFrame()
    summary = pd.concat(frames, ignore_index=True)
    leading = ["detector", "model", "n_cases", "n_failed", "threshold"]
    return summary[leading + [c for c in summary.columns if c not in leading]]


def print_reduction(reduction: pd.DataFrame, runner) -> None:
    """Condition × detector: mean paired change vs. the baseline answer."""
    from benchmark.reduction_runner import paired_deltas

    cols = [c for c in runner.thresholds() if c in reduction.columns]
    deltas = paired_deltas(reduction, cols)
    if deltas.empty:
        logger.warning("No reduction condition could be compared with its baseline; see run.log")
        return
    table = deltas.pivot_table(index="condition", columns="detector", values="delta",
                               aggfunc="mean", sort=False).reset_index()
    short = {c: c.replace("selfcheckgpt_", "sc_").replace("uqlm_", "uq_")
             .replace("semantic_negentropy", "sne").replace("noncontradiction", "noncon")
             .replace("entailment", "entail").replace("cosine_sim", "cos")
             .replace("exact_match", "exact").replace("bert_score", "bert")
             for c in table.columns}
    print_table(table.rename(columns=short))
    console.line("mean change in each detector's risk score vs. the model's baseline answer on the same")
    console.line("question (negative = looks less hallucinated). Per model, per dataset: see the report.")


def print_plan(plan: dict, config: dict) -> None:
    runs = plan["runs"]
    console.section("Run plan (config.yaml `run:` + command-line flags)")
    console.kv("runs", f"{runs}  → " + (" · ".join(f"run_{i:02d}" for i in range(1, min(runs, 3) + 1))
                                       + (" … " if runs > 3 else "") + " + combined/ (mean ± std)"))
    console.kv("questions", " · ".join(f"{k} {v}" for k, v in plan["samples_per_dataset"].items())
               + "  (per dataset)")
    console.kv("detectors", ", ".join(plan["detectors"]) or "none")
    if any(n in plan["detectors"] for n in SAMPLING_DETECTORS):
        console.kv("samples", f"{plan['selfcheckgpt_samples']} per question per model "
                   "(shared by every sampling-based detector)")
    console.kv("reduction", (", ".join(plan["reduction_methods"]) + " vs. baseline"
                             + f" · self_refine up to {plan['reduction_iterations']} rounds")
               if plan["reduce"] else "off")
    if any(n in plan["detectors"] for n in JUDGE_DETECTORS) or "prompt" in (
            config["detectors"]["selfcheckgpt"].get("methods") or []) and "selfcheckgpt" in plan["detectors"]:
        console.kv("judge", config.get("judge", {}).get("model"))
    if any(n in plan["detectors"] for n in SAMPLING_DETECTORS):
        console.kv("models", ", ".join(config.get("selected_models") or
                                       [m["name"] for m in config.get("models", [])]))


def run_once(run_dir: Path, run_label: str, config: dict, runner, datasets, generators,
             reduce_on: bool, score_reduction_from: Path | None = None) -> dict:
    """One complete repetition into `run_dir`; returns its stage timings."""
    from benchmark.reduction_runner import ReductionRunner
    from reporting import generate
    from utils.run_manifest import write_run_files

    started_at = datetime.now(timezone.utc)
    stage_seconds: dict[str, float] = {}
    run_dir.mkdir(parents=True, exist_ok=True)
    runner.output_dir = run_dir
    run_config = {**config, "benchmark": {**config["benchmark"], "output_dir": str(run_dir)}}

    total_stages = 2 if (reduce_on or score_reduction_from) else 1
    console.section(f"{run_label} · Stage 1/{total_stages} · Detector validation")
    stage_start = time.monotonic()
    raw = runner.validate(datasets, generators=generators or None)
    stage_seconds["detector_validation"] = time.monotonic() - stage_start

    summary = summarise(raw, runner)
    if summary.empty or not (summary["n_cases"] > 0).any():
        raise SystemExit(f"All enabled detectors failed on every case in {run_label}; see run.log")
    summary.to_csv(run_dir / "detector_validation_summary.csv", index=False)
    console.section(f"{run_label} · Detector validation results")
    shown = summary.rename(columns={"average_precision": "auprc", "roc_auc": "auroc"})
    shown["model"] = shown["model"].replace(MODEL_INDEPENDENT, "—")
    print_table(shown.reindex(columns=["detector", "model", "n_cases", "n_failed", "auroc", "auprc",
                                       "accuracy", "precision", "recall", "f1", "threshold"]))
    console.line("auroc/auprc are threshold-free; the other columns use the threshold shown.")

    if reduce_on:
        console.section(f"{run_label} · Stage 2/2 · Reduction "
                        f"(baseline vs. {', '.join(config['reduction'].get('methods', []))})")
        stage_start = time.monotonic()
        reduction = ReductionRunner(run_config, runner).run(datasets, generators)
        stage_seconds["reduction"] = time.monotonic() - stage_start
        console.section(f"{run_label} · Reduction results")
        print_reduction(reduction, runner)
    elif score_reduction_from:
        source = score_reduction_from / run_dir.name / "reduction_comparison.csv"
        console.section(f"{run_label} · Stage 2/2 · Scoring the reduction answers of {source.parent}")
        stage_start = time.monotonic()
        score_reduction_answers(source, run_dir / "reduction_scores.csv", runner, datasets)
        stage_seconds["reduction_scoring"] = time.monotonic() - stage_start

    if runner.generator_columns():
        runner.bank.export(run_dir / "selfcheckgpt_samples.jsonl")
    write_run_files(run_dir, run_config, sys.argv, datasets, generators, started_at, stage_seconds)
    try:
        generate(run_dir, title=f"{run_dir.parent.name}/{run_dir.name}")
    except Exception as exc:
        logger.opt(exception=exc).error(f"Report for {run_label} failed: {exc}")
    return stage_seconds


def score_reduction_answers(source: Path, output: Path, runner, datasets) -> None:
    """Score another environment's reduction answers with this run's
    (generator-independent) detectors, keyed by sample, model, condition."""
    if not source.is_file():
        raise SystemExit(f"{source} not found: run the sampling-detector environment first")
    answers = pd.read_csv(source)
    by_id = {s.sample_id: s for group in datasets.values() for s in group}
    rows = []
    usable = answers[answers["answer"].notna()] if "answer" in answers else answers.iloc[0:0]
    bar = console.progress(list(usable.itertuples(index=False)), desc=f"{'reduction answers':<26}",
                           total=len(usable), unit="answer")
    for item in bar:
        sample = by_id.get(item.sample_id)
        key = {"sample_id": item.sample_id, "model": item.model, "condition": item.condition}
        if sample is None:
            rows.append({**key, "error": "sample not loaded in this environment"})
            continue
        case = {"case_id": f"{item.sample_id}:{item.condition}", "question": sample.question,
                "context": sample.context, "answer": item.answer}
        rows.append({**key, **runner.score_answer(case, None)})
    bar.close()
    pd.DataFrame(rows).to_csv(output, index=False)
    console.line(f"✓ scored {len(rows)} reduction answers → {output.name}")


def main() -> None:
    args = parse_args()
    started = time.monotonic()

    config = load_config(args.config)
    plan = apply_plan(config, args)
    output_dir = Path(config["benchmark"].get("output_dir", "results/current"))
    old_runs = sorted(output_dir.glob("run_[0-9]*")) if output_dir.is_dir() else []
    if old_runs and not (args.dry_run or args.preflight):
        raise SystemExit(
            f"{output_dir} already holds results ({', '.join(p.name for p in old_runs)}). "
            "Choose a new --output (or move the old folder): mixing runs would make the "
            "combined report wrong."
        )
    log_file = None if args.dry_run else output_dir / "run.log"
    console.setup_logging(config.get("logging", {}).get("level", "INFO"), log_file)
    logger.debug("Command: {}", " ".join(sys.argv))

    enabled = plan["detectors"]
    selfcheck_on = any(n in enabled for n in SAMPLING_DETECTORS)   # needs generators
    judge_on = "uqlm_judge" in enabled or (
        "selfcheckgpt" in enabled and "prompt" in (config["detectors"]["selfcheckgpt"].get("methods") or []))
    score_from = Path(args.score_reduction_from) if args.score_reduction_from else None
    reduce_on = plan["reduce"]
    n_runs = plan["runs"]

    mode = " · dry run" if args.dry_run else " · preflight only" if args.preflight else ""
    console.header("LLM hallucination benchmark" + mode)
    console.kv("config", args.config)
    if not args.dry_run:
        console.kv("output", output_dir)
        console.kv("full log", log_file)
    print_plan(plan, config)

    # ── Setup: fetch anything missing ───────────────────────────────────
    from utils.resources import prepare

    console.section("Setup" + (" (dry run: nothing is downloaded)" if args.dry_run
                               else " (anything missing is downloaded and verified)"))
    prepare(config, dry_run=args.dry_run)
    if selfcheck_on or judge_on:
        from models import ModelFactory
        ModelFactory.ensure_ollama_models(
            config, dry_run=args.dry_run, include_generators=selfcheck_on,
            extra_tags=[config["judge"]["model"]] if judge_on else [])

    # ── Data ────────────────────────────────────────────────────────────
    from data.datasets import DatasetLoader

    console.section("Data (the same subset is used in every run)")
    seed = int(config["benchmark"].get("seed", 42))
    data_config = config
    pending_download = []
    if args.dry_run:
        # Files the Setup step said it would download are not loaded here.
        data_config = {**config, "datasets": []}
        for d in config.get("datasets", []):
            if d.get("source") == "json" and d.get("path") and not Path(d["path"]).is_file():
                pending_download.append(d["name"])
            else:
                data_config["datasets"].append(d)
    datasets = DatasetLoader(data_config, seed=seed).load_all()
    for name in pending_download:
        console.kv(name, "downloaded on the real run", width=24)
    wanted = [d["name"] for d in data_config.get("datasets", []) if d.get("enabled", True)]
    missing = [name for name in wanted if name not in datasets]
    if missing:
        raise SystemExit(
            f"\nDataset(s) failed to load: {', '.join(missing)} (reason above). "
            "Fix them or set `enabled: false` in the config; a run on partial "
            "data would be mislabeled."
        )
    n_samples = n_cases = n_hallucinated = 0
    for name, samples in datasets.items():
        cases = DatasetLoader.detection_cases(samples)
        bad = sum(c["label"] for c in cases)
        n_samples += len(samples)
        n_cases += len(cases)
        n_hallucinated += bad
        console.kv(name, f"{len(samples):>4} questions → {len(cases):>4} labeled answers "
                   f"({len(cases) - bad} faithful / {bad} hallucinated)", width=24)
    console.kv("total", f"{n_samples:>4} questions → {n_cases:>4} labeled answers "
               f"({n_cases - n_hallucinated} faithful / {n_hallucinated} hallucinated) · seed {seed}", width=24)
    if not n_cases and not pending_download:
        raise SystemExit("No labeled detector-validation cases were loaded")

    # ── Detectors ───────────────────────────────────────────────────────
    console.section("Detectors")
    if not enabled:
        console.line("none selected (run.detectors or --detectors)")
    for name in enabled:
        console.kv(name, describe_detector(name, config["detectors"][name], config.get("judge", {})))

    if args.dry_run:
        console.section("Dry run complete")
        console.line("Nothing was downloaded or loaded. A real run fetches every ↓ item above first.")
        return
    if not enabled:
        raise SystemExit("No detector selected (run.detectors or --detectors)")

    # ── Generators ──────────────────────────────────────────────────────
    generators = []
    if selfcheck_on:
        from models import ModelFactory

        console.section("Generators (samples for SelfCheckGPT/UQLM" + (" + reduction)" if reduce_on else ")"))
        generators = ModelFactory.build_all(config)
        if not generators:
            raise SystemExit(
                "The sampling-based detectors need at least one generator model: start "
                "Ollama (`ollama serve`) and check selected_models in the config."
            )
        for g in generators:
            extras = [x for x in (getattr(g, "parameter_size", None),
                                  getattr(g, "quantization", None)) if x]
            if g.config.get("think") is not None:
                extras.append(f"think={g.config['think']}")
            console.kv(g.name, f"{g.config.get('model')}" + (f"  ({', '.join(extras)})" if extras else ""),
                       width=18)
        requested = [c["name"] for c in ModelFactory.active_configs(config)]
        missing_models = [n for n in requested if n not in {g.name for g in generators}]
        if missing_models:
            raise SystemExit(
                f"\nSelected model(s) not available: {', '.join(missing_models)} "
                "(reason above). Pull them, or remove them from selected_models."
            )

        n_per_prompt = int(config["detectors"]["selfcheckgpt"]["n_samples"])
        per_run = n_samples * n_per_prompt * len(generators)
        console.section("Workload")
        console.kv("validation", f"{n_samples} questions × {n_per_prompt} samples × "
                   f"{len(generators)} models = {per_run:,} generations per run")
        if reduce_on:
            iters = int(config["reduction"]["max_iterations"])
            methods = config["reduction"].get("methods", [])
            per_q = 1 + ("closed_book" in methods) + ("greedy" in methods) \
                + (2 * iters if "self_refine_adapted" in methods else 0) + (7 if "cove_adapted" in methods else 0)
            console.kv("reduction", f"{n_samples} questions × {len(generators)} models × up to {per_q} calls "
                       f"= up to {n_samples * len(generators) * per_q:,} generations per run "
                       f"({len(methods) + 1} answers per question, each scored by every detector)")
        console.kv("runs", f"× {n_runs}")

    # ── Preflight: every piece once, on one real case ───────────────────
    from benchmark import BenchmarkRunner
    from benchmark.preflight import run_preflight

    console.section("Preflight (one real case through every detector, model and stage)")
    runner = BenchmarkRunner(config, generator=generators[0] if generators else None)
    estimate = run_preflight(config, runner, datasets, generators, reduce_on)
    per_run_seconds = sum(estimate.values())
    if per_run_seconds:
        parts = [f"{stage.replace('_', ' ')} ~{console.duration(sec)}"
                 for stage, sec in estimate.items() if sec]
        console.line(f"Estimated time per run: {' · '.join(parts)}")
        console.line(f"Estimated total: {n_runs} run(s) × ~{console.duration(per_run_seconds)} "
                     f"= ~{console.duration(per_run_seconds * n_runs)}")
    if args.preflight:
        console.section("Preflight passed")
        console.line("Everything needed is downloaded and working. Run again without "
                     "--preflight to start.")
        return
    console.line("All checks passed · starting. Each progress bar shows that model's "
                 "elapsed<remaining time.")

    # ── Runs ────────────────────────────────────────────────────────────
    for index in range(1, n_runs + 1):
        run_label = f"Run {index}/{n_runs}"
        console.header(f"{run_label} → {output_dir / f'run_{index:02d}'}")
        if index > 1:
            runner.bank.reset()  # every run draws its own samples
        run_once(output_dir / f"run_{index:02d}", run_label, config, runner,
                 datasets, generators, reduce_on, score_from)

    # ── Combined ────────────────────────────────────────────────────────
    from reporting import generate

    console.header(f"Combined · {n_runs} run(s) → {output_dir / 'combined'}")
    produced = {}
    try:
        produced = generate(output_dir, out_dir=output_dir / "combined")
    except Exception as exc:
        logger.opt(exception=exc).error(f"Combined report failed: {exc}")
    takeaways = output_dir / "combined" / "takeaways.md"
    if takeaways.is_file():
        for line in takeaways.read_text(encoding="utf-8").splitlines():
            if line.startswith("- "):
                console.line("• " + line[2:])

    console.section(f"Done in {console.duration(time.monotonic() - started)}")
    console.line(f"{output_dir}/")
    for index in range(1, n_runs + 1):
        console.line(f"  run_{index:02d}/     REPORT.md · report.html · report.docx · charts/ · tables/ · raw CSVs")
    console.line("  combined/   REPORT.md · report.html · report.docx · takeaways.md · charts/ · tables/"
                 "   ← start here")
    console.line("  run.log     full log with tracebacks")
    if "report (docx)" in produced:
        console.kv("open", produced["report (docx)"])


if __name__ == "__main__":
    main()
