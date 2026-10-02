#!/usr/bin/env python3
"""Run the benchmark in one Python environment.

Setup (download + verify everything) → preflight (one real question through
every detector, model and reduction method) → run_01 … run_N (Stage A:
detector validation on labeled answers; Stage B: reduction methods vs. the
baseline answer) → per-run and combined reports. scripts/run_full.py starts
a worker per detector group per run. Flow and diagrams: docs/ARCHITECTURE.md.
"""
from __future__ import annotations

import argparse
import json
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
    smoke_group = parser.add_mutually_exclusive_group()
    smoke_group.add_argument(
        "--smoke", action="store_true",
        help="Quick end-to-end test: 2 runs, 2 questions per dataset, 2 samples, 1 refine round "
             "(any of these given explicitly still wins)",
    )
    smoke_group.add_argument(
        "--smoke-2q", action="store_true",
        help="Two questions total from the first enabled dataset; 2 runs, every detector, "
             "every configured reduction method, and every selected generator model",
    )
    parser.add_argument(
        "--part", help=argparse.SUPPRESS,   # used by scripts/run_full.py: write run_XX/<part>/
    )
    parser.add_argument("--run-index", type=int, help=argparse.SUPPRESS)
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
        "--device", choices=("auto", "cpu", "cuda"),
        help="Device for the torch-based detectors: SelfCheckGPT (BERTScore, NLI), UQLM "
             "(NLI, BERTScore, embeddings, best-response), SummaC, AlignScore. Default "
             "`auto` (config): the GPU when one is present, else the CPU. MiniCheck picks "
             "the GPU itself. Ollama uses its own GPU; this flag does not affect it.",
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
    parser.add_argument("--skip-preflight", action="store_true", help=argparse.SUPPRESS)
    return parser.parse_args()


def load_config(path: str) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        config = yaml.safe_load(handle) or {}
    config.setdefault("benchmark", {})
    config.setdefault("detectors", {})
    config.setdefault("reduction", {})
    return config


# why `auto` chose what it chose; printed in the Detectors section
DEVICE_NOTE: list[str] = []


# detectors that run torch models (MiniCheck picks its device by itself)
TORCH_DETECTORS = ("selfcheckgpt", "uqlm", "summac", "alignscore", "minicheck")


def _resolve_device(device: str) -> str:
    """`auto`: the GPU only when it is large enough to hold the detectors next
    to Ollama's models (utils/gpu.py); otherwise the GPU is left to Ollama."""
    if device != "auto":
        return device
    from utils.gpu import AUTO_GPU_MIN_GIB, nvidia_gpu
    gpu = nvidia_gpu()
    if gpu is None:
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"
    name, total = gpu
    if total < AUTO_GPU_MIN_GIB:
        if not DEVICE_NOTE:
            DEVICE_NOTE.append(
                f"{name} has {total:.1f} GB (< {AUTO_GPU_MIN_GIB} GB): the GPU is left to Ollama, "
                "detectors run on the CPU (same scores, slower; --device cuda to override)")
        return "cpu"
    return "cuda"


SMOKE = {"runs": 2, "max_samples": 2, "n_samples": 2, "max_iterations": 1}


def apply_smoke(args: argparse.Namespace) -> None:
    """Smoke modes fill in test numbers that were not given explicitly."""
    if getattr(args, "smoke", False) or getattr(args, "smoke_2q", False):
        for key, value in SMOKE.items():
            if getattr(args, key, None) is None:
                setattr(args, key, value)


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

    # "auto" (the config default): see _resolve_device
    for name in TORCH_DETECTORS:
        wanted = args.device or detectors_cfg[name].get("device", "auto")
        detectors_cfg[name]["device"] = _resolve_device(wanted)
    # Detectors on the CPU: hide the GPU from this process, so libraries that
    # pick the GPU by themselves (UQLM's cosine model, MiniCheck) cannot take
    # memory Ollama needs. Ollama is a separate process and keeps the GPU.
    if all(detectors_cfg[n]["device"] == "cpu" for n in TORCH_DETECTORS if n in selected):
        import os
        from utils.gpu import nvidia_gpu
        if nvidia_gpu() is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = ""

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
        text = f"{', '.join(methods)} · temperature={cfg.get('temperature', 1.0)} · device={cfg.get('device')}"
        return text + (f" · prompt judge={judge.get('model')}" if "prompt" in methods else "")
    if name == "uqlm":
        return ", ".join(cfg.get("scorers") or ["default scorers"]) + f" · device={cfg.get('device')}"
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
             reduce_on: bool, score_reduction_from: Path | None = None,
             run_name: str = "", make_report: bool = True) -> dict:
    """One complete repetition into `run_dir`; returns its stage timings."""
    from benchmark.reduction_runner import ReductionRunner
    from data.datasets import DatasetLoader
    from reporting import generate
    from utils.run_manifest import write_run_files
    from utils.completeness import inspect_scores

    started_at = datetime.now(timezone.utc)
    stage_seconds: dict[str, float] = {}
    coverage = {}
    run_dir.mkdir(parents=True, exist_ok=True)
    runner.output_dir = run_dir
    run_config = {**config, "benchmark": {**config["benchmark"], "output_dir": str(run_dir)}}

    total_stages = 2 if (reduce_on or score_reduction_from) else 1
    console.section(f"{run_label} · Stage 1/{total_stages} · Detector validation")
    stage_start = time.monotonic()
    raw = runner.validate(datasets, generators=generators or None)
    stage_seconds["detector_validation"] = time.monotonic() - stage_start
    score_columns = [f"{col}_score" for fam in runner.families.values() for col in fam.columns]
    cases = [case for group in datasets.values() for case in DatasetLoader.detection_cases(group)]
    n_cases = len(cases)
    n_validation_models = len(generators) if runner.generator_columns() else 1
    validation_keys = ([(case['case_id'], g.name) for case in cases for g in generators]
                       if runner.generator_columns() else [(case['case_id'],) for case in cases])
    validation_key_columns = ("case_id", "model") if runner.generator_columns() else ("case_id",)
    coverage["detector_validation"] = inspect_scores(raw.to_dict("records"), score_columns,
                                                     n_cases * n_validation_models,
                                                     validation_key_columns, validation_keys)

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
        reducer = ReductionRunner(run_config, runner)
        reduction = reducer.run(datasets, generators)
        n_conditions = len(reducer.conditions())
        n_questions = sum(len(group) for group in datasets.values())
        coverage["reduction"] = inspect_scores(reduction.to_dict("records"), score_columns,
                                               n_questions * len(generators) * n_conditions,
                                               ("sample_id", "model", "condition"),
                                               [(sample.sample_id, g.name, cond) for group in datasets.values()
                                                for sample in group for g in generators for cond in reducer.conditions()])
        stage_seconds["reduction"] = time.monotonic() - stage_start
        console.section(f"{run_label} · Reduction results")
        print_reduction(reduction, runner)
    elif score_reduction_from:
        source = score_reduction_from / run_name / "core" / "reduction_comparison.csv"
        if not source.is_file():   # a single-environment output folder
            source = score_reduction_from / run_name / "reduction_comparison.csv"
        console.section(f"{run_label} · Stage 2/2 · Scoring the reduction answers of {source.parent}")
        stage_start = time.monotonic()
        scored = score_reduction_answers(source, run_dir / "reduction_scores.csv", runner, datasets)
        source_answers = pd.read_csv(source)
        coverage["reduction_scoring"] = inspect_scores(scored.to_dict("records"), score_columns,
                                                       len(source_answers), ("sample_id", "model", "condition"),
                                                       list(source_answers[["sample_id", "model", "condition"]].itertuples(index=False, name=None)))
        stage_seconds["reduction_scoring"] = time.monotonic() - stage_start

    if runner.generator_columns():
        runner.bank.export(run_dir / "selfcheckgpt_samples.jsonl")
    coverage["passed"] = all(stage["passed"] for stage in coverage.values())
    (run_dir / "score_completeness.json").write_text(json.dumps(coverage, indent=2), encoding="utf-8")
    write_run_files(run_dir, run_config, sys.argv, datasets, generators, started_at, stage_seconds)
    if make_report:
        produced = generate(run_dir, title=f"{run_dir.parent.name}/{run_dir.name}")
        if "report (docx)" not in produced:
            raise RuntimeError(f"Report for {run_label} did not produce report.docx")
    if not coverage["passed"]:
        raise SystemExit(f"Incomplete scores in {run_label}; this run FAILED. "
                         f"See {run_dir / 'score_completeness.json'} and run.log. "
                         "Recorded results are retained for diagnosis.")
    console.line(f"✓ {run_label} · every expected score is present and finite")
    return stage_seconds


def score_reduction_answers(source: Path, output: Path, runner, datasets) -> pd.DataFrame:
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
    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False)
    console.line(f"✓ scored {len(rows)} reduction answers → {output.name}")
    return frame


def main() -> None:
    args = parse_args()
    started = time.monotonic()

    config = load_config(args.config)
    if args.smoke_2q:
        if (args.max_samples is not None or (args.detectors and not args.part) or
                args.no_reduce or args.runs not in (None, 2)):
            raise SystemExit("--smoke-2q fixes two questions, two runs, all detectors and reducers; "
                             "omit --max-samples, --detectors, and --no-reduce, "
                             "and do not override --runs")
        from utils.smoke import configure_two_question_smoke
        chosen = configure_two_question_smoke(config, DETECTOR_NAMES)
        console.line(f"Two-question smoke dataset: {chosen}")
    apply_smoke(args)
    plan = apply_plan(config, args)
    output_dir = Path(config["benchmark"].get("output_dir", "results/current"))
    part = args.part   # set by scripts/run_full.py: this process fills run_XX/<part>/
    if args.run_index is not None and (not part or not 1 <= args.run_index <= plan["runs"]):
        raise SystemExit("--run-index requires --part and an index between 1 and --runs")
    pattern = (f"run_{args.run_index:02d}/{part}" if args.run_index is not None else
               f"run_[0-9]*/{part}" if part else "run_[0-9]*")
    old_runs = sorted(output_dir.glob(pattern)) if output_dir.is_dir() else []
    if old_runs and not (args.dry_run or args.preflight):
        raise SystemExit(
            f"{output_dir} already holds results ({', '.join(str(p.relative_to(output_dir)) for p in old_runs)}). "
            "Choose a new --output (or move the old folder): mixing runs would make the "
            "combined report wrong."
        )
    log_file = None if args.dry_run else (output_dir / "logs" / f"{part}.log" if part else output_dir / "run.log")
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
        # A dry run must not ask any loader to fetch a missing file, including
        # RAGTruth and HaluBench, or invoke the Hugging Face remote loader.
        from utils.offline_plan import local_datasets
        data_config, pending_download = local_datasets(config)
    datasets = DatasetLoader(data_config, seed=seed).load_all()
    for name in pending_download:
        console.kv(name, "not loaded in dry run; checked on a real run", width=24)
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
    if args.smoke_2q and not args.dry_run and n_samples != 2:
        raise SystemExit(f"--smoke-2q needs exactly two usable questions in {chosen}; "
                         f"loaded {n_samples}")
    if not n_cases and not pending_download:
        raise SystemExit("No labeled detector-validation cases were loaded")

    # ── Detectors ───────────────────────────────────────────────────────
    console.section("Detectors")
    if not enabled:
        console.line("none selected (run.detectors or --detectors)")
    for name in enabled:
        console.kv(name, describe_detector(name, config["detectors"][name], config.get("judge", {})))
    if DEVICE_NOTE and any(n in enabled for n in TORCH_DETECTORS):
        console.kv("device", DEVICE_NOTE[0])

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

    runner = BenchmarkRunner(config, generator=generators[0] if generators else None)
    runner.attach(generators)   # a model that does not fit can take the detectors' GPU memory
    if not args.skip_preflight:
        console.section("Preflight (one real case through every detector, model and stage)")
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
    indices = [args.run_index] if args.run_index is not None else range(1, n_runs + 1)
    for index in indices:
        run_name = f"run_{index:02d}"
        run_dir = output_dir / run_name / part if part else output_dir / run_name
        run_label = f"Run {index}/{n_runs}"
        console.header(f"{run_label} → {run_dir}")
        runner.bank.reset()  # fresh samples, including after a standalone preflight
        run_once(run_dir, run_label, config, runner, datasets, generators, reduce_on, score_from,
                 run_name=run_name, make_report=not part)

    if part:   # scripts/run_full.py builds each run's report and the combined one
        console.section(f"Done in {console.duration(time.monotonic() - started)}")
        console.line(f"{part}: {len(indices)} run(s) written to {output_dir}/"
                     + (f"run_{args.run_index:02d}/{part}/" if args.run_index is not None else f"run_XX/{part}/"))
        return

    # ── Combined ────────────────────────────────────────────────────────
    from reporting import generate

    console.header(f"Combined · {n_runs} run(s) → {output_dir / 'combined'}")
    produced = generate(output_dir, out_dir=output_dir / "combined")
    if "report (docx)" not in produced:
        raise RuntimeError("Combined report did not produce report.docx")
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
