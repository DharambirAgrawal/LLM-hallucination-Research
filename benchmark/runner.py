"""Run pinned upstream detectors on fixed, labeled responses.

Detector families and the score columns they produce (higher = more likely
hallucinated in every column):

  needs a generator (uses the shared samples, scored once per model):
    selfcheckgpt   selfcheckgpt_<ngram|bertscore|nli|prompt>
    uqlm           uqlm_<semantic_negentropy|noncontradiction|entailment|
                         cosine_sim|bert_score>  (exact_match opt-in)
  generator-independent (scored once per case in each run):
    uqlm_judge     uqlm_judge
    minicheck      minicheck
    summac         summac
    alignscore     alignscore
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Optional

import pandas as pd
from loguru import logger

from data.datasets import BenchmarkSample, DatasetLoader
from detectors import (
    AlignScoreDetector,
    MiniCheckDetector,
    SelfCheckGPTDetector,
    SummaCDetector,
)
from detectors.sampling import SampleBank
from models.base_model import BaseModel
from utils import console

# Stop a model/detector after this many failures in a row: the cause is
# almost always environmental (server down, model missing, package broken),
# and continuing would spend hours recording the same error.
MAX_CONSECUTIVE_FAILURES = 10

GENERATOR_FAMILIES = ("selfcheckgpt", "uqlm")
FAMILIES = ("selfcheckgpt", "uqlm", "uqlm_judge", "minicheck", "summac", "alignscore")


@dataclass
class Family:
    name: str
    columns: List[str]                  # output names (without "_score")
    thresholds: Dict[str, float]
    needs_generator: bool
    score: Callable[..., Dict[str, float]]   # (case, generator) -> {column: risk}


def _selfcheck_scores(result) -> Dict[str, object]:
    out: Dict[str, object] = {f"selfcheckgpt_{m}": v for m, v in result.scores.items()}
    if result.errors:
        out["__errors__"] = "; ".join(f"{m}: {e}" for m, e in result.errors.items())
    return out


def _apply(row: dict, fam: "Family", values: Dict[str, object]) -> Optional[str]:
    """Write a family's scores into `row`; columns a scorer could not
    produce stay empty and are named in the family's error field."""
    partial = values.pop("__errors__", None)
    for col in fam.columns:
        row[f"{col}_score"] = values.get(col)
    row[f"{fam.name}_error"] = f"partial: {partial}" if partial else None
    return partial


# families whose models take a `device` setting and can be moved to the CPU
CPU_CAPABLE = ("selfcheckgpt", "uqlm", "summac", "alignscore")


def _is_out_of_memory(exc: BaseException) -> bool:
    return "out of memory" in str(exc).lower() or type(exc).__name__ == "OutOfMemoryError"


def _free_gpu_cache() -> None:
    """Give memory torch has cached but no longer uses back to the GPU, so
    Ollama (a separate process) can load the next model."""
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


class BenchmarkRunner:
    """Detector-validation harness; it contains no detection algorithm."""

    def __init__(self, config: dict, generator: Optional[BaseModel] = None):
        self.config = config
        self.generator = generator
        self.output_dir = Path(config.get("benchmark", {}).get("output_dir", "results/current"))
        self.output_dir.mkdir(parents=True, exist_ok=True)
        sc = config.get("detectors", {}).get("selfcheckgpt", {})
        self.bank = SampleBank(int(sc.get("n_samples", 5)), float(sc.get("temperature", 1.0)))
        self.detectors: Dict[str, object] = {}
        self.families: Dict[str, Family] = {}
        self._judge = config.get("judge", {})
        self._host = config.get("ollama", {}).get("host", "http://localhost:11434")
        self._raw: Dict[str, Callable] = {}
        self._build(config.get("detectors", {}), self._judge, self._host)
        for name in list(self.families):
            self._guard(name)

    # ── construction ────────────────────────────────────────────────────
    def _build(self, cfgs: dict, judge: dict, host: str) -> None:
        cfg = cfgs.get("selfcheckgpt", {})
        if cfg.get("enabled", False):
            methods = cfg.get("methods") or [cfg.get("method", "ngram")]
            thresholds = {m: float(cfg.get("thresholds", {}).get(m, cfg.get("threshold", 0.5)))
                          for m in methods}
            detector = SelfCheckGPTDetector(
                model=self.generator, method=methods,
                n_samples=self.bank.n_samples, temperature=self.bank.temperature,
                threshold=float(cfg.get("threshold", 0.5)), thresholds=thresholds,
                device=cfg.get("device", "cpu"), judge_model=judge.get("model"),
                judge_host=host, bank=self.bank,
            )
            self.detectors["selfcheckgpt"] = detector
            self.families["selfcheckgpt"] = Family(
                "selfcheckgpt", [f"selfcheckgpt_{m}" for m in methods],
                {f"selfcheckgpt_{m}": t for m, t in thresholds.items()}, True,
                lambda case, g, d=detector: _selfcheck_scores(
                    d.detect(case["question"], case["context"], case["answer"], model=g)),
            )

        cfg = cfgs.get("uqlm", {})
        if cfg.get("enabled", False):
            from detectors.uqlm_detector import DEFAULT_SCORERS, UQLMConsistencyDetector
            scorers = cfg.get("scorers") or list(DEFAULT_SCORERS)
            detector = UQLMConsistencyDetector(self.bank, scorers, cfg.get("device"))
            self.detectors["uqlm"] = detector
            self.families["uqlm"] = Family(
                "uqlm", [f"uqlm_{s}" for s in scorers],
                {f"uqlm_{s}": float(cfg.get("threshold", 0.5)) for s in scorers}, True,
                lambda case, g, d=detector: {
                    f"uqlm_{k}": v for k, v in
                    d.detect(case["question"], case["context"], case["answer"], g).items()},
            )

        cfg = cfgs.get("uqlm_judge", {})
        if cfg.get("enabled", False):
            from detectors.uqlm_detector import UQLMJudgeDetector
            if not judge.get("model"):
                raise ValueError("uqlm_judge needs judge.model in the config")
            detector = UQLMJudgeDetector(judge["model"], host, cfg.get("template", "true_false_uncertain"))
            self.detectors["uqlm_judge"] = detector
            self.families["uqlm_judge"] = Family(
                "uqlm_judge", ["uqlm_judge"], {"uqlm_judge": float(cfg.get("threshold", 0.5))}, False,
                lambda case, g, d=detector: {
                    "uqlm_judge": d.detect(case["question"], case["context"], case["answer"])},
            )

        for name, factory in (
            ("summac", lambda c: SummaCDetector(threshold=c.get("threshold", 0.5),
                                                device=c.get("device", "cpu"),
                                                model_name=c.get("model_name", "vitc"),
                                                conv_weights=c.get("conv_weights",
                                                    "external_models/summac/summac_conv_vitc_sent_perc_e.bin"))),
            ("minicheck", lambda c: MiniCheckDetector(model_name=c.get("model_name", "flan-t5-large"),
                                                      cache_dir=c.get("cache_dir", "external_models/minicheck"),
                                                      threshold=c.get("threshold", 0.5))),
            ("alignscore", lambda c: AlignScoreDetector(checkpoint_path=c["checkpoint_path"],
                                                        threshold=c.get("threshold", 0.5),
                                                        device=c.get("device", "cpu"),
                                                        model_name=c.get("model_name", "roberta-base"))),
        ):
            cfg = cfgs.get(name, {})
            if cfg.get("enabled", False):
                detector = factory(cfg)
                self.detectors[name] = detector
                self.families[name] = Family(
                    name, [name], {name: float(cfg.get("threshold", 0.5))}, False,
                    lambda case, g, d=detector, n=name: {n: d.detect(case["context"], case["answer"]).score},
                )

        if not self.families:
            logger.warning("No detector is enabled in the configuration")

    # ── GPU memory: the detectors share the GPU with Ollama ─────────────
    def _guard(self, name: str) -> None:
        """Route family `name` through out-of-memory recovery."""
        self._raw[name] = self.families[name].score
        self.families[name].score = lambda case, g, n=name: self._call(n, case, g)

    def _call(self, name: str, case: dict, generator) -> Dict[str, object]:
        """Score one case. On a GPU out-of-memory error (Ollama may have
        loaded a large model into the memory the detector needed): free
        torch's cache and retry; if it happens again, reload this detector on
        the CPU for the rest of the run (same scores, slower) and retry."""
        def attempt() -> Dict[str, object]:
            values = self._raw[name](case, generator)
            # SelfCheckGPT isolates each scorer's error instead of raising
            if "out of memory" in str(values.get("__errors__") or "").lower():
                raise RuntimeError(f"CUDA out of memory: {values['__errors__']}")
            return values

        try:
            result = attempt()
        except Exception as exc:
            if not _is_out_of_memory(exc):
                raise
            _free_gpu_cache()
            try:
                result = attempt()
            except Exception as again:
                if not _is_out_of_memory(again):
                    raise
                self._move_to_cpu(name)
                result = attempt()
        _free_gpu_cache()
        return result

    def attach(self, generators: list) -> None:
        """Let each Ollama generator ask for GPU memory when its model does
        not fit (see OllamaModel._make_room)."""
        for g in generators:
            if hasattr(g, "on_memory_pressure"):
                g.on_memory_pressure = self.free_gpu

    def free_gpu(self) -> None:
        """Move every detector still on the GPU to the CPU for the rest of the
        run, so Ollama can load a model that did not fit."""
        detectors = self.config.get("detectors", {})
        for name in [n for n in CPU_CAPABLE if n in self.families]:
            if detectors.get(name, {}).get("device") != "cpu":
                self._move_to_cpu(name, reason="an Ollama model needed the GPU memory")
        _free_gpu_cache()

    def _move_to_cpu(self, name: str, reason: str = "GPU out of memory twice") -> None:
        cfg = dict(self.config.get("detectors", {}).get(name, {}))
        if name not in CPU_CAPABLE or cfg.get("device") == "cpu":
            raise RuntimeError(f"{name}: out of memory twice, even after freeing the GPU cache. "
                               "Free GPU memory (smaller Ollama model, or `device: cpu` for the "
                               "detectors in config.yaml)")
        logger.warning(f"{name}: {reason}; reloading it on the CPU for the rest of "
                       "this run (same scores, slower)")
        old = self.detectors.pop(name, None)
        cfg.update(enabled=True, device="cpu")
        self.config.setdefault("detectors", {})[name] = cfg
        self._build({name: cfg}, self._judge, self._host)
        del old
        import gc
        gc.collect()
        _free_gpu_cache()
        self._guard(name)

    # ── public helpers ──────────────────────────────────────────────────
    def thresholds(self) -> Dict[str, float]:
        """Decision threshold of every score column."""
        return {f"{col}_score": thr for fam in self.families.values() for col, thr in fam.thresholds.items()}

    def generator_columns(self) -> List[str]:
        """Score columns that depend on the generator (reported per model)."""
        return [f"{col}_score" for fam in self.families.values() if fam.needs_generator for col in fam.columns]

    def score_answer(self, case: dict, generator) -> Dict[str, object]:
        """Every enabled detector on one answer: {<col>_score, <family>_error}.
        Used by the reduction stage; a failing family does not stop the rest."""
        row: Dict[str, object] = {}
        for fam in self.families.values():
            try:
                _apply(row, fam, fam.score(case, generator))
            except Exception as exc:
                detail = str(exc).strip() or repr(exc)
                for col in fam.columns:
                    row[f"{col}_score"] = None
                row[f"{fam.name}_error"] = f"{type(exc).__name__}: {detail}"
                logger.opt(exception=exc).debug("{} failed on {}: {}", fam.name, case.get("case_id"), detail)
        return row

    # ── Stage A ─────────────────────────────────────────────────────────
    def validate(
        self,
        datasets: Dict[str, List[BenchmarkSample]],
        generators: Optional[List[BaseModel]] = None,
    ) -> pd.DataFrame:
        """Score fixed labeled answers and preserve every failure.

        Generator-independent detectors score each case once per invocation;
        no scores are carried between runs. The sampling-based
        detectors then run one model at a time over every case, so Ollama
        keeps a single model loaded; a `model` column tags those rows.
        """
        cases = []
        for samples in datasets.values():
            cases.extend(DatasetLoader.detection_cases(samples))

        gen_families = [f for f in self.families.values() if f.needs_generator]
        fixed_families = [f for f in self.families.values() if not f.needs_generator]
        active_generators = (
            generators if generators is not None
            else ([self.generator] if self.generator else [])
        )
        if gen_families and not any(active_generators):
            raise ValueError("A sampling-based detector is enabled but no generator model was provided")

        base_rows = [dict(case) for case in cases]
        for fam in fixed_families:
            self._score_all(base_rows, [fam], fam.name, None)

        if not gen_families:
            rows = base_rows
        else:
            rows = []
            total = len(active_generators)
            for index, generator in enumerate(active_generators, 1):
                model_name = getattr(generator, "name", None)
                model_rows = [{**row, "model": model_name} for row in base_rows]
                self._score_all(model_rows, gen_families, f"[{index}/{total}] {model_name}", generator)
                rows.extend(model_rows)
                # free Ollama's memory for the next model and the detectors
                if hasattr(generator, "release"):
                    generator.release()

        frame = pd.DataFrame(rows)
        output = self.output_dir / "detector_validation_raw.csv"
        frame.to_csv(output, index=False)
        logger.debug("Saved raw detector validation to {}", output)
        return frame

    def _score_all(self, rows: List[dict], families: List[Family], label: str, generator) -> None:
        """Fill every family's columns on every row, with one progress bar
        and a one-line outcome per family. Tracebacks go to run.log."""
        started = time.monotonic()
        failures = {f.name: 0 for f in families}
        consecutive = {f.name: 0 for f in families}
        first_error: Dict[str, str] = {}
        partials: Dict[str, int] = {}
        bar = console.progress(rows, desc=f"{label:<26}", total=len(rows))
        for row in bar:
            for fam in families:
                if consecutive[fam.name] >= MAX_CONSECUTIVE_FAILURES:
                    for col in fam.columns:
                        row[f"{col}_score"] = None
                    row[f"{fam.name}_error"] = f"skipped after {MAX_CONSECUTIVE_FAILURES} consecutive failures"
                    failures[fam.name] += 1
                    continue
                try:
                    partial = _apply(row, fam, fam.score(row, generator))
                    consecutive[fam.name] = 0
                    if partial:
                        partials[fam.name] = partials.get(fam.name, 0) + 1
                        first_error.setdefault(fam.name, row[f"{fam.name}_error"])
                except Exception as exc:
                    detail = str(exc).strip() or repr(exc)
                    for col in fam.columns:
                        row[f"{col}_score"] = None
                    row[f"{fam.name}_error"] = f"{type(exc).__name__}: {detail}"
                    logger.opt(exception=exc).debug("{} {} failed on {}: {}", label, fam.name, row["case_id"], detail)
                    failures[fam.name] += 1
                    consecutive[fam.name] += 1
                    first_error.setdefault(fam.name, row[f"{fam.name}_error"])
            failed = sum(failures.values())
            if failed:
                bar.set_postfix_str(f"failed={failed}")
        bar.close()

        elapsed = console.duration(time.monotonic() - started)
        names = ", ".join(f.name for f in families)
        for name, count in partials.items():
            logger.warning(f"{label.strip()} · {name}: {count}/{len(rows)} cases missing some scores "
                           f"· first: {first_error.get(name, '')[:160]}")
        if not any(failures.values()):
            if not partials:
                console.line(f"✓ {label.strip()} · {names} · {len(rows)} cases in {elapsed}")
            return
        for fam in families:
            if failures[fam.name]:
                aborted = consecutive[fam.name] >= MAX_CONSECUTIVE_FAILURES
                logger.warning(
                    f"{label.strip()} · {fam.name}: {failures[fam.name]}/{len(rows)} cases failed in {elapsed}"
                    + (" (stopped early)" if aborted else "")
                    + f" · first error: {first_error[fam.name][:160]}"
                )
