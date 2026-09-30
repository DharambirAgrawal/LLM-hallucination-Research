"""Stage B: every reduction method on the same questions, scored the same way.

For each model and question, the model's grounded baseline answer is
produced once; every reduction method then produces its own answer (the
self-correction methods start from that same baseline), and every enabled
detector scores every answer against the same shared samples. One row per
(question, model, condition) goes to reduction_comparison.csv; the report
compares each condition with the baseline, paired by question.

Conditions (the `reproduction_status` column says what each one is):

  baseline             grounded answer (context in the prompt, the RAG
                       setting), the model's default sampling
  closed_book          same question, no context: the gap to `baseline` is
                       what retrieval-augmented generation (Lewis et al.,
                       2020) adds, with the dataset's own passage as oracle
                       retrieval
  greedy               grounded answer at temperature 0 (a decoding
                       setting, not a published method)
  self_refine_adapted  Self-Refine loop (Madaan et al., 2023), local
                       inspired adaptation; see reducers/self_refine.py
  cove_adapted         Chain-of-Verification, factored (Dhuliawala et al.,
                       2023); local implementation, no official code exists
  uqlm_best_response   UQLM's official semantic-entropy best-response
                       selection over the baseline + its samples; scored
                       leave-one-out (never against itself), see _score

This is an engineering harness for the Stage B protocol in
docs/REPRODUCIBILITY.md, not its full statistical analysis.
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List

import pandas as pd
from loguru import logger

from benchmark.runner import MAX_CONSECUTIVE_FAILURES
from data.datasets import BenchmarkSample
from models.base_model import BaseModel
from models.prompts import closed_book_prompt, grounded_prompt
from utils import console

CONDITIONS = {
    "baseline": "reference_condition_grounded_answer",
    "closed_book": "rag_ablation_no_context_oracle_retrieval_comparison",
    "greedy": "decoding_setting_temperature_0_not_a_published_method",
    "self_refine_adapted": "local_inspired_baseline_NOT_an_upstream_reproduction",
    "cove_adapted": "local_implementation_of_published_method_no_official_code",
    "uqlm_best_response": "official_uqlm_implementation",
}
METHODS = tuple(name for name in CONDITIONS if name != "baseline")


class ReductionRunner:
    """Runs every configured reduction method for each generator."""

    def __init__(self, config: dict, runner):
        cfg = config.get("reduction", {})
        methods = cfg.get("methods")
        if methods is None:
            legacy = cfg.get("method")  # older configs name a single method
            methods = [legacy] if legacy else list(METHODS)
        unknown = [m for m in methods if m not in METHODS]
        if unknown:
            # "_adapted" is deliberate: a bare "self_refine" is refused so a
            # table can never cite this as plain, unqualified Self-Refine.
            raise ValueError(f"Unsupported reduction method(s) {unknown}; choose from {list(METHODS)}")
        if not getattr(runner, "families", None):
            raise ValueError("Reduction needs at least one enabled detector to score the answers")
        self.methods = list(methods)
        self.max_iterations = int(cfg.get("max_iterations", 3))
        self.runner = runner
        self.output_dir = Path(config.get("benchmark", {}).get("output_dir", "results/current"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._best = None
        if "uqlm_best_response" in self.methods:
            from reducers.uqlm_best_response import UQLMBestResponseReducer
            device = config.get("detectors", {}).get("uqlm", {}).get("device")
            uqlm = getattr(runner, "detectors", {}).get("uqlm")
            nli = uqlm.shared_nli() if uqlm is not None and hasattr(uqlm, "shared_nli") else None
            self._best = UQLMBestResponseReducer(device=device, nli=nli)

    def conditions(self) -> List[str]:
        return ["baseline", *self.methods]

    def _produce(self, model: BaseModel, sample: BenchmarkSample) -> Dict[str, dict]:
        """Every condition's answer for one question. A failure is recorded
        on that condition only; methods that start from the baseline are
        skipped (with the reason) when the baseline itself failed."""
        from reducers.cove import ChainOfVerificationReducer
        from reducers.self_refine import SelfRefineReducer

        q, c = sample.question, sample.context
        out: Dict[str, dict] = {}

        def run(name, fn):
            start = time.monotonic()
            try:
                answer, calls, details = fn()
                out[name] = {"answer": answer, "n_calls": calls, "details": details, "error": None,
                             "latency_seconds": time.monotonic() - start}
            except Exception as exc:
                detail = str(exc).strip() or repr(exc)
                out[name] = {"answer": None, "n_calls": None, "details": None,
                             "error": f"{type(exc).__name__}: {detail}",
                             "latency_seconds": time.monotonic() - start}
                logger.opt(exception=exc).debug("{} failed for {} ({}): {}", name, sample.sample_id, model.name, detail)

        run("baseline", lambda: (model.generate(grounded_prompt(q, c)), 1, {}))
        baseline = out["baseline"]["answer"]
        needs_baseline = {"self_refine_adapted", "cove_adapted", "uqlm_best_response"}

        for name in self.methods:
            if name in needs_baseline and baseline is None:
                out[name] = {"answer": None, "n_calls": 0, "details": None, "latency_seconds": 0.0,
                             "error": "skipped: baseline answer failed"}
                continue
            if name == "closed_book":
                run(name, lambda: (model.generate(closed_book_prompt(q)), 1, {}))
            elif name == "greedy":
                run(name, lambda: (model.generate(grounded_prompt(q, c), temperature=0.0), 1, {}))
            elif name == "self_refine_adapted":
                def self_refine():
                    result = SelfRefineReducer(model=model, max_iterations=self.max_iterations).reduce(
                        q, c, initial_answer=baseline)
                    return result.final_answer, 2 * len(result.feedback_history) - (
                        1 if result.stopped_reason == "no_issues_reported" else 0), {
                        "iterations": result.iterations, "stopped_reason": result.stopped_reason,
                        "feedback_history": result.feedback_history}
                run(name, self_refine)
            elif name == "cove_adapted":
                def cove():
                    result = ChainOfVerificationReducer(model).reduce(q, c, baseline)
                    return result.final_answer, result.n_calls, {
                        "verification_questions": result.verification_questions,
                        "verification_answers": result.verification_answers}
                run(name, cove)
            elif name == "uqlm_best_response":
                def best():
                    samples = self.runner.bank.get(model, q, c)
                    chosen = self._best.select(baseline, samples)
                    candidates = [baseline, *samples]
                    return chosen, 0, {"chosen_index": candidates.index(chosen) if chosen in candidates else None,
                                       "n_candidates": len(candidates)}
                run(name, best)
        return out

    def _score(self, case: dict, model, cond: str, item: dict, answers: Dict[str, dict]) -> dict:
        """Score one answer with every detector. An answer that
        uqlm_best_response picked FROM the samples would otherwise be scored
        against evidence that contains itself (a perfect match, biasing it
        towards low risk), so it is scored leave-one-out: against the other
        candidates (the baseline + the samples) minus every exact copy of the
        picked answer. Its evidence can therefore be smaller than the other
        conditions' (the number of copies removed)."""
        details = item.get("details") or {}
        index = details.get("chosen_index")
        # index 0: the baseline itself was picked, scored like the baseline.
        # index None (not found verbatim among the candidates) is still
        # scored without any exact copy of itself, like a picked sample.
        if cond != "uqlm_best_response" or index == 0:
            return self.runner.score_answer(case, model)
        samples = self.runner.bank.get(model, case["question"], case["context"])
        chosen = case["answer"]
        # UQLM prefers the most repeated answer, so the pick often has exact
        # copies among the samples; every copy is removed, not just one.
        others = [a for a in [answers["baseline"]["answer"], *samples] if a != chosen]
        if not others:
            raise ValueError("no evidence left after removing the picked answer's copies")
        with self.runner.bank.evidence(model, case["question"], case["context"], others):
            return self.runner.score_answer(case, model)

    def run(self, datasets: Dict[str, List[BenchmarkSample]], generators: List[BaseModel]) -> pd.DataFrame:
        samples = [sample for group in datasets.values() for sample in group]
        rows = []
        total = len(generators)
        for index, model in enumerate(generators, 1):
            label = f"[{index}/{total}] {model.name}"
            started = time.monotonic()
            failures = consecutive = 0
            first_error = None
            bar = console.progress(samples, desc=f"{label:<26}", total=len(samples), unit="question")
            for sample in bar:
                base = {"sample_id": sample.sample_id, "dataset": sample.dataset, "model": model.name,
                        "question": sample.question}
                if consecutive >= MAX_CONSECUTIVE_FAILURES:
                    for cond in self.conditions():
                        rows.append({**base, "condition": cond, "reproduction_status": CONDITIONS[cond],
                                     "error": f"skipped after {MAX_CONSECUTIVE_FAILURES} consecutive failures"})
                    failures += 1
                    continue
                answers = self._produce(model, sample)
                sample_failed = answers["baseline"]["error"] is not None
                for cond in self.conditions():
                    item = answers[cond]
                    row = {**base, "condition": cond, "reproduction_status": CONDITIONS[cond],
                           "answer": item["answer"], "n_calls": item["n_calls"],
                           "latency_seconds": item["latency_seconds"],
                           "details": json.dumps(item["details"], ensure_ascii=False) if item["details"] else None,
                           "error": item["error"]}
                    if item["answer"] is not None:
                        case = {"case_id": f"{sample.sample_id}:{cond}", "question": sample.question,
                                "context": sample.context, "answer": item["answer"]}
                        row.update(self._score(case, model, cond, item, answers))
                    elif item["error"] and not item["error"].startswith("skipped"):
                        first_error = first_error or f"{cond}: {item['error']}"
                    rows.append(row)
                if sample_failed:
                    failures += 1
                    consecutive += 1
                    first_error = first_error or f"baseline: {answers['baseline']['error']}"
                    bar.set_postfix_str(f"failed={failures}")
                else:
                    consecutive = 0
            bar.close()
            elapsed = console.duration(time.monotonic() - started)
            if failures == 0:
                console.line(f"✓ {model.name} · {len(samples)} questions × {len(self.conditions())} conditions in {elapsed}")
            else:
                logger.warning(f"{model.name}: {failures}/{len(samples)} questions failed in {elapsed}"
                               f" · first error: {(first_error or '')[:160]}")

        frame = pd.DataFrame(rows)
        output = self.output_dir / "reduction_comparison.csv"
        frame.to_csv(output, index=False)
        logger.debug("Saved reduction comparison to {}", output)
        return frame


def paired_deltas(frame: pd.DataFrame, score_columns: List[str]) -> pd.DataFrame:
    """For every non-baseline condition and score column: the paired
    difference (condition − baseline) on the same question and model.
    Negative = the method's answer looks less hallucinated."""
    if frame.empty or "condition" not in frame.columns:
        return pd.DataFrame()
    keys = [k for k in ("run", "sample_id", "model") if k in frame.columns]
    base = frame[frame["condition"] == "baseline"].set_index(keys)
    out = []
    for cond, group in frame[frame["condition"] != "baseline"].groupby("condition", sort=False):
        g = group.set_index(keys)
        for col in score_columns:
            if col not in g.columns or col not in base.columns:
                continue
            both = pd.concat([g[col].rename("method"), base[col].rename("baseline")], axis=1, join="inner").dropna()
            if both.empty:
                continue
            delta = both["method"] - both["baseline"]
            part = delta.rename("delta").reset_index()
            part["condition"], part["detector"] = cond, col.removesuffix("_score")
            out.append(part)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()
