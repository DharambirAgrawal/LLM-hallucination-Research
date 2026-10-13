"""Stage A and Stage B of one run, model by model, in three phases.

Before, every question went generator → detectors → judge → next question.
On one GPU that meant the generator, the judge (both in Ollama) and the
detectors' torch models (in this process) kept pushing each other out of GPU
memory: Ollama reloaded models thousands of times per run, and the torch
detectors had to run on the CPU. The first full run spent ~37 of its ~51 core
hours per run scoring on the CPU.

Now, for each generator model:

  1. generate   (Ollama: this model only; detector models released)
                samples for every question, and every reduction answer that
                needs the model (baseline, closed_book, greedy, self-refine,
                CoVe). Then the model is unloaded.
  2. detectors  (GPU: torch models only; Ollama empty)
                UQLM best-response selection, then the SelfCheckGPT
                n-gram/BERTScore/NLI and UQLM scores of the labeled answers
                and of every reduction answer.
  3. judge      (Ollama: the judge model only; detector models released)
                SelfCheckGPT's prompt scorer and UQLM's LLM judge on the
                same answers. Then the judge is unloaded.

The judge's verdicts on the labeled answers (model-independent) are made
once, first. The outputs are the same files with the same columns as before
(detector_validation_raw.csv, reduction_comparison.csv); only the order of
the work changed. Every answer is scored against the same shared samples, and
a uqlm_best_response pick is still scored leave-one-out.
"""
from __future__ import annotations

import time
from typing import Dict, List, Optional, Tuple

import pandas as pd
from loguru import logger

from benchmark.runner import MAX_CONSECUTIVE_FAILURES
from data.datasets import BenchmarkSample, DatasetLoader
from utils import console


def _generate(runner, reducer, model, samples: List[BenchmarkSample], label: str) -> Dict[str, Dict[str, dict]]:
    """Phase 1: everything that needs the generator, question by question."""
    answers: Dict[str, Dict[str, dict]] = {}
    started = time.monotonic()
    failures = consecutive = 0
    first_error = None
    sampling = bool(runner.generator_columns())
    bar = console.progress(samples, desc=f"{label + ' · generate':<26}", total=len(samples), unit="question")
    for sample in bar:
        if consecutive >= MAX_CONSECUTIVE_FAILURES:
            reason = f"skipped after {MAX_CONSECUTIVE_FAILURES} consecutive failures"
            runner.bank.mark_failed(model, sample.question, sample.context, reason)
            if reducer is not None:
                answers[sample.sample_id] = {cond: {"answer": None, "n_calls": 0, "details": None,
                                                    "latency_seconds": 0.0, "error": reason}
                                             for cond in reducer.conditions()}
            failures += 1
            continue
        failed = False
        if sampling:
            try:
                runner.bank.get(model, sample.question, sample.context)
            except Exception as exc:
                failed = True
                first_error = first_error or f"samples: {exc}"
        if reducer is not None:
            produced = reducer.produce_generated(model, sample)
            answers[sample.sample_id] = produced
            if produced["baseline"]["error"]:
                failed = True
                first_error = first_error or f"baseline: {produced['baseline']['error']}"
        if failed:
            failures += 1
            consecutive += 1
            bar.set_postfix_str(f"failed={failures}")
        else:
            consecutive = 0
    bar.close()
    elapsed = console.duration(time.monotonic() - started)
    if failures:
        logger.warning(f"{label} · generate: {failures}/{len(samples)} questions had a failure in {elapsed}"
                       f" · first: {(first_error or '')[:200]}")
    else:
        console.line(f"✓ {label} · generate · {len(samples)} questions in {elapsed}")
    return answers


def _select_best(reducer, model, samples: List[BenchmarkSample], answers, label: str) -> None:
    if reducer is None or "uqlm_best_response" not in reducer.methods:
        return
    started = time.monotonic()
    bar = console.progress(samples, desc=f"{label + ' · best response':<26}", total=len(samples), unit="question")
    for sample in bar:
        reducer.select_best(model, sample, answers[sample.sample_id])
    bar.close()
    failed = sum(1 for s in samples if answers[s.sample_id]["uqlm_best_response"]["error"])
    elapsed = console.duration(time.monotonic() - started)
    if failed:
        logger.warning(f"{label} · best response: {failed}/{len(samples)} not selected in {elapsed}")
    else:
        console.line(f"✓ {label} · best response · {len(samples)} questions in {elapsed}")


def run_phased(runner, datasets: Dict[str, List[BenchmarkSample]], generators: list,
               reducer=None) -> Tuple[pd.DataFrame, Optional[pd.DataFrame], Dict[str, float]]:
    """Run Stage A (and Stage B when `reducer` is given) for every model.
    Returns (validation frame, reduction frame or None, seconds per phase)."""
    samples = [s for group in datasets.values() for s in group]
    by_id = {s.sample_id: s for s in samples}
    cases = [c for group in datasets.values() for c in DatasetLoader.detection_cases(group)]
    base_rows = [dict(case) for case in cases]
    fixed = [f for f in runner.families.values() if not f.needs_generator]
    sampling = [f for f in runner.families.values() if f.needs_generator]
    seconds = {"generation": 0.0, "detector_scoring": 0.0, "judge_scoring": 0.0}
    # Reuse of identical inputs is only ever within one run: every run
    # recomputes all scores (run_protocol in run_manifest.json).
    runner._answer_cache = {}
    for detector in getattr(runner, "detectors", {}).values():
        if hasattr(detector, "_judge_cache"):
            detector._judge_cache.clear()

    def timed(phase, fn, *args):
        start = time.monotonic()
        try:
            return fn(*args)
        finally:
            seconds[phase] += time.monotonic() - start

    # Anything already loaded on the GPU (e.g. UQLM's models, loaded when the
    # reduction tools were created) is released until its turn.
    runner.park()
    # Model-independent families on the labeled answers (the judge, in core).
    for part, phase in (("torch", "detector_scoring"), ("judge", "judge_scoring")):
        fams = [f for f in fixed if part in f.parts]
        if not fams:
            continue
        if part == "torch":
            runner.unpark()
        timed(phase, runner._score_all, base_rows, fams, f"labeled answers · {part}", None, part)
        runner.park()
    runner.release_judge()

    validation_rows, reduction_rows = [], []
    total = len(generators)
    for index, model in enumerate(generators, 1):
        label = f"[{index}/{total}] {model.name}"
        console.line(f"── {label}: generate → detectors (GPU) → judge")

        # 1. generate: Ollama holds only this model
        runner.park()
        answers = timed("generation", _generate, runner, reducer, model, samples, label)
        if hasattr(model, "release"):
            model.release()

        model_rows = [{**row, "model": model.name} for row in base_rows]
        rows = []

        # 2. detectors: the torch models, alone on the GPU
        runner.unpark()
        timed("detector_scoring", _select_best, reducer, model, samples, answers, label)
        if reducer is not None:
            for sample in samples:
                rows.extend(reducer.rows_for(model, sample, answers[sample.sample_id]))
        timed("detector_scoring", runner._score_all, model_rows, sampling, f"{label} · detectors", model, "torch")
        if reducer is not None:
            timed("detector_scoring", reducer.score_rows, rows, model, by_id, answers, "torch",
                  f"{label} · reduction · detectors")
        runner.park()

        # 3. judge: Ollama holds only the judge model
        timed("judge_scoring", runner._score_all, model_rows, sampling, f"{label} · judge", model, "judge")
        if reducer is not None:
            timed("judge_scoring", reducer.score_rows, rows, model, by_id, answers, "judge",
                  f"{label} · reduction · judge")
        runner.release_judge()

        validation_rows.extend(model_rows)
        reduction_rows.extend(rows)

    frame = pd.DataFrame(validation_rows if generators else base_rows)
    output = runner.output_dir / "detector_validation_raw.csv"
    frame.to_csv(output, index=False)
    logger.debug("Saved raw detector validation to {}", output)
    reduction = None
    if reducer is not None:
        reduction = pd.DataFrame(reduction_rows)
        out = reducer.output_dir / "reduction_comparison.csv"
        reduction.to_csv(out, index=False)
        logger.debug("Saved reduction comparison to {}", out)
    return frame, reduction, seconds
