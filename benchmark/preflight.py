"""Fail-fast checks that run before any long stage.

Every part of the pipeline is exercised once on one real question from the
loaded data: every generator-independent detector loads (downloading its
weights on first use) and scores two answers; every generator answers,
draws its samples, and every sampling-based detector scores two answers
with them; with the reduction stage on, every reduction method runs once per
model and all its answers are scored. If anything fails, the run stops here
with the reason for each failure, instead of hours later.

The timings give the estimate printed before the long part. Preflight is a
separate check: main.py discards its samples before the measured runs.
"""
from __future__ import annotations

import tempfile
import time
from pathlib import Path
from typing import Callable, Dict, List

from loguru import logger

from data.datasets import BenchmarkSample, DatasetLoader
from utils import console
from utils.completeness import require_scores


class Preflight:
    def __init__(self) -> None:
        self.failures: List[str] = []

    def check(self, label: str, action: Callable[[], str]) -> float | None:
        """Run `action`; print ✓/✗ with its detail. Returns seconds taken,
        or None if it failed."""
        started = time.monotonic()
        try:
            detail = action()
        except Exception as exc:
            reason = str(exc).strip() or type(exc).__name__
            logger.opt(exception=exc).debug(f"Preflight failed: {label}: {reason}")
            console.line(f"✗ {label:<34} {type(exc).__name__}: {reason[:400]}")
            self.failures.append(f"{label}: {reason[:400]}")
            return None
        seconds = time.monotonic() - started
        console.line(f"✓ {label:<34} {detail} ({console.duration(seconds)})")
        return seconds


def _fmt(scores: Dict[str, object]) -> str:
    return " ".join(f"{k.replace('selfcheckgpt_', 'sc_').replace('uqlm_', 'uq_')}={v:.2f}"
                    for k, v in scores.items() if not k.startswith("__") and v is not None)


def run_preflight(
    config: dict,
    runner,
    datasets: Dict[str, List[BenchmarkSample]],
    generators: list,
    reduce_on: bool,
    *,
    reduction_answers_per_question: int = 0,
) -> Dict[str, float]:
    """Exercise the whole pipeline once. Raises SystemExit listing every
    failure; otherwise returns estimated seconds per stage for one run."""
    pf = Preflight()
    output_dir = Path(config["benchmark"].get("output_dir", "results/current"))

    def writable() -> str:
        output_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=output_dir):
            pass
        return "ok"

    def report_libraries() -> str:
        import docx  # noqa: F401  (report.docx)
        import matplotlib  # noqa: F401
        import sklearn  # noqa: F401
        return "matplotlib, scikit-learn, python-docx"

    pf.check("output folder writable", writable)
    pf.check("report libraries", report_libraries)

    samples = [s for group in datasets.values() for s in group]
    all_cases = [c for g in datasets.values() for c in DatasetLoader.detection_cases(g)]
    # A median-length input is a less misleading timing probe than whichever
    # dataset happens to come first. This adds no model calls to preflight.
    first = sorted(samples, key=lambda s: len(s.question) + len(s.context))[len(samples) // 2]
    cases = DatasetLoader.detection_cases([first])
    case_a = cases[0]
    case_b = cases[1] if len(cases) > 1 else cases[0]
    n_questions, n_cases = len(samples), len(all_cases)
    estimate = {"detector_validation": 0.0, "reduction": 0.0}
    fixed_seconds = 0.0
    fixed_startup = 0.0
    generator_loading = 0.0
    sampling_startup = 0.0
    probe_timings = {}
    reducer = None

    # Generator-independent detectors: the first call loads (and downloads)
    # the model, the second gives the per-case time.
    fixed = [f for f in runner.families.values() if not f.needs_generator]
    for fam in fixed:
        loaded = pf.check(f"{fam.name} loads + scores", lambda f=fam: _fmt(require_scores(f.score(case_a, None), f.columns)))
        if loaded is not None:
            per_case = pf.check(f"{fam.name} timing", lambda f=fam: _fmt(require_scores(f.score(case_b, None), f.columns)))
            estimate["detector_validation"] += (per_case or 0) * n_cases
            fixed_seconds += per_case or 0
            fixed_startup += max(0.0, loaded - (per_case or 0))
            probe_timings[fam.name] = {"first_score_seconds": loaded, "warm_score_seconds": per_case}

    sampling = [f for f in runner.families.values() if f.needs_generator]
    for g in generators:
        answered = pf.check(f"{g.name} answers",
                            lambda g=g: f"{len(g.generate('Reply with one short sentence: what is 2 + 2?'))} chars")
        if answered is None:
            continue
        # The run releases each generator after validation and again after
        # reduction. Use its first answer as a rough restart allowance rather
        # than multiplying one cold load by every question.
        generator_loading += answered * (2 if reduce_on else 1)
        drawn = pf.check(f"{g.name} samples",
                         lambda g=g: f"{len(runner.bank.get(g, first.question, first.context))} samples")
        if drawn is None or not sampling:
            continue
        scored = []
        for fam in sampling:
            def run_case(case, f=fam, g=g):
                values = require_scores(f.score(case, g), f.columns)
                return _fmt(values)
            t1 = pf.check(f"{g.name} · {fam.name}", lambda: run_case(case_a))
            t2 = pf.check(f"{g.name} · {fam.name} timing", lambda: run_case(case_b)) \
                if t1 is not None else None
            scored.append(t2 or 0)
            if fam.name not in probe_timings:
                sampling_startup += max(0.0, (t1 or 0) - (t2 or 0))
                probe_timings[fam.name] = {"first_score_seconds": t1, "warm_score_seconds": t2}
        # Per question: draw samples once; per labeled case: every scorer.
        estimate["detector_validation"] += drawn * n_questions + sum(scored) * n_cases

        if reduce_on and not pf.failures:
            from benchmark.reduction_runner import ReductionRunner

            if reducer is None:
                def load_reducer() -> str:
                    nonlocal reducer
                    reducer = ReductionRunner(config, runner)
                    best = getattr(reducer, "_best", None)
                    if best is not None:
                        best._load()  # separate lazy model loading from per-question inference
                    return "ready (loading counted once per run)"

                startup = pf.check("reduction tools load", load_reducer)
                estimate["reducer_loading"] = startup or 0
            if pf.failures:
                if hasattr(g, "release"):
                    g.release()
                continue

            def reduction_round(g=g) -> str:
                answers = reducer._produce(g, first)
                errors = {k: v["error"] for k, v in answers.items() if v["error"]}
                if errors:
                    raise RuntimeError("; ".join(f"{k}: {v}" for k, v in errors.items()))
                for cond, item in answers.items():
                    # the same scoring path as the run, incl. leave-one-out
                    row = reducer._score({"case_id": f"preflight:{cond}", "question": first.question,
                                          "context": first.context, "answer": item["answer"]},
                                         g, cond, item, answers)
                    failed = {k: v for k, v in row.items()
                              if k.endswith("_error") and v}
                    if failed:
                        raise RuntimeError(f"scoring {cond}: {failed}")
                    for fam in runner.families.values():
                        require_scores({col: row.get(f"{col}_score") for col in fam.columns}, fam.columns)
                return f"{len(answers)} answers ({', '.join(answers)}) produced and scored"

            reduced = pf.check(f"{g.name} reduction round", reduction_round)
            estimate["reduction"] += (reduced or 0) * n_questions
        # free Ollama's memory for the next model and the detectors
        if hasattr(g, "release"):
            g.release()

    if pf.failures:
        console.section("Preflight failed · nothing long was started")
        for failure in pf.failures:
            console.line(f"✗ {failure}")
        console.line("Full tracebacks are in run.log. Fix these (or remove the model/detector")
        console.line("from the config) and run again.")
        raise SystemExit(1)
    if reduction_answers_per_question:
        estimate["reduction_scoring"] = fixed_seconds * n_questions * reduction_answers_per_question
    if generator_loading:
        estimate["generator_restarts"] = generator_loading
    if sampling_startup:
        estimate["sampling_detector_loading"] = sampling_startup
    # Separate worker processes must reload fixed detectors; a standalone
    # main.py run keeps them loaded after preflight.
    runner.preflight_timing = {
        "questions": n_questions, "labeled_cases": n_cases,
        "generator_models": len(generators),
        "generator_names": [g.name for g in generators],
        "detectors": list(runner.families),
        "sample_count": config.get("detectors", {}).get("selfcheckgpt", {}).get("n_samples"),
        "reduction_answers_per_question": reduction_answers_per_question,
        "fixed_detector_startup_seconds": fixed_startup,
        "detector_probes": probe_timings,
        "probe": {"sample_id": first.sample_id, "dataset": first.dataset,
                  "question_chars": len(first.question), "context_chars": len(first.context)},
    }
    return estimate
