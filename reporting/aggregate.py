"""Load one or many run folders and compute every table the reports use.

A *run folder* is what one `main.py` repetition writes (run_01, run_02, …):
detector_validation_summary.csv, detector_validation_raw.csv, optionally
reduction_comparison.csv, run_manifest.json and config_used.yaml. A
scripts/run_full.py output holds one run folder per detector and run
(<detector>/run_01 …). Every run folder found under the input is used,
except anything inside a `combined/` folder (those are outputs).

Across runs, each metric is reported as mean ± std (sample std, n-1). Only
the dataset subset is shared between runs (same seed); SelfCheckGPT samples
and reduction answers are drawn fresh in every run, so the std measures how
stable a result is under the model's own sampling.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import yaml

from benchmark.detector_validation import DetectorValidator

RUN_NAME = re.compile(r"^run_\d+$")
METRICS = ["roc_auc", "average_precision", "accuracy", "precision", "recall", "f1"]
MODEL_INDEPENDENT = "n/a (model-independent detector)"


@dataclass
class RunData:
    folder: Path
    run: str
    summary: pd.DataFrame
    raw: Optional[pd.DataFrame]
    reduction: Optional[pd.DataFrame]
    reduction_scores: Optional[pd.DataFrame]   # scores from a detector in another venv
    manifest: Optional[dict]
    config: Optional[dict]
    group: str = ""          # detector group (core, minicheck, …) that produced it


def _read_csv(path: Path) -> Optional[pd.DataFrame]:
    return pd.read_csv(path) if path.is_file() else None


def discover(input_dir: Path) -> List[RunData]:
    runs = []
    for path in sorted(input_dir.rglob("detector_validation_summary.csv")):
        if "combined" in path.relative_to(input_dir).parts:
            continue
        folder = path.parent
        # Layouts: <out>/run_XX/<group>/ (scripts/run_full.py) or <out>/run_XX/ (main.py)
        if RUN_NAME.match(folder.parent.name):
            run, group = folder.parent.name, folder.name
        elif RUN_NAME.match(folder.name):
            run, group = folder.name, folder.parent.name
        else:
            run, group = "run_01", folder.name
        manifest_path = folder / "run_manifest.json"
        config_path = folder / "config_used.yaml"
        runs.append(RunData(
            folder=folder,
            run=run,
            summary=pd.read_csv(path).assign(run=run),
            raw=(lambda f: f.assign(run=run) if f is not None else None)(
                _read_csv(folder / "detector_validation_raw.csv")),
            reduction=(lambda f: f.assign(run=run) if f is not None else None)(
                _read_csv(folder / "reduction_comparison.csv")),
            reduction_scores=(lambda f: f.assign(run=run) if f is not None else None)(
                _read_csv(folder / "reduction_scores.csv")),
            manifest=json.loads(manifest_path.read_text()) if manifest_path.is_file() else None,
            config=yaml.safe_load(config_path.read_text()) if config_path.is_file() else None,
            group=group,
        ))
    if not runs:
        raise SystemExit(f"No run folder (detector_validation_summary.csv) found under {input_dir}")
    return runs


def concat(frames: List[Optional[pd.DataFrame]]) -> pd.DataFrame:
    frames = [f for f in frames if f is not None and not f.empty]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def mean_std(frame: pd.DataFrame, keys: List[str], values: List[str]) -> pd.DataFrame:
    """Mean and std across runs: one row per `keys`, columns
    `<value>_mean` / `<value>_std`, plus `n_runs`."""
    if frame.empty:
        return pd.DataFrame()
    values = [v for v in values if v in frame.columns]
    grouped = frame.groupby(keys, dropna=False, sort=False)
    out = grouped[values].agg(["mean", "std"])
    out.columns = [f"{col}_{stat}" for col, stat in out.columns]
    out = out.reset_index()
    out.insert(len(keys), "n_runs", grouped["run"].nunique().to_numpy())
    return out


# ── detector validation ─────────────────────────────────────────────────

def summary_by_run(runs: List[RunData]) -> pd.DataFrame:
    frame = concat([r.summary for r in runs])
    lead = ["run", "detector", "model", "n_cases", "n_failed", "threshold"]
    lead = [c for c in lead if c in frame.columns]
    return frame[lead + [c for c in frame.columns if c not in lead]]


def thresholds(summary: pd.DataFrame) -> Dict[str, float]:
    return summary.drop_duplicates("detector").set_index("detector")["threshold"].to_dict()


def per_dataset_by_run(runs: List[RunData], limits: Dict[str, float],
                       per_model: set) -> pd.DataFrame:
    """Detector metrics per dataset (task type), recomputed from each run's
    raw scores with the same sklearn metrics main.py uses. `per_model` names
    the detectors whose scores depend on the generator."""
    validator = DetectorValidator()
    rows = []
    for r in runs:
        raw = r.raw
        if raw is None or raw.empty:
            continue
        has_model = "model" in raw.columns and raw["model"].notna().any()
        score_cols = [c for c in raw.columns if c.endswith("_score")]
        for column in score_cols:
            detector = column.removesuffix("_score")
            if detector in per_model and has_model:
                groups = raw.groupby(["dataset", "model"], sort=False)
            else:
                frame = raw.drop_duplicates(subset="case_id") if has_model else raw
                groups = ((key, g) for key, g in frame.groupby("dataset", sort=False))
            for key, group in groups:
                dataset, model = (key if isinstance(key, tuple) else (key, MODEL_INDEPENDENT))
                usable = group[["label", column]].dropna()
                if usable.empty or usable["label"].nunique() < 2:
                    continue
                metrics = validator.evaluate(usable["label"], usable[column], limits.get(detector, 0.5))
                rows.append({"run": r.run, "dataset": dataset, "detector": detector,
                             "model": model, **metrics,
                             "n_failed": int(group[column].isna().sum())})
    return pd.DataFrame(rows)


def failures(runs: List[RunData]) -> pd.DataFrame:
    """Every failure message, counted per run / stage / detector / model."""
    columns = ["run", "stage", "detector", "model", "count", "error"]
    rows = []

    def count(frame: pd.DataFrame, error_col: str, stage: str, detector: str, run: str) -> None:
        errors = frame[frame[error_col].notna()]
        if errors.empty:
            return
        keyed = pd.DataFrame({
            "model": errors["model"].fillna("—") if "model" in errors else "—",
            "error": errors[error_col].astype(str).str.slice(0, 160),
        })
        for (model, message), group in keyed.groupby(["model", "error"]):
            rows.append({"run": run, "stage": stage, "detector": detector,
                         "model": model, "count": len(group), "error": message})

    for r in runs:
        if r.raw is not None:
            for column in [c for c in r.raw.columns if c.endswith("_error")]:
                count(r.raw, column, "validation", column.removesuffix("_error"), r.run)
        for frame, stage in ((r.reduction, "reduction answer"), (r.reduction_scores, "reduction scoring")):
            if frame is None:
                continue
            if "error" in frame.columns:
                keyed = frame.assign(error=frame["condition"].astype(str) + ": " + frame["error"].astype(str))
                count(keyed[frame["error"].notna()], "error", stage, "generation", r.run)
            for column in [c for c in frame.columns if c.endswith("_error") and c != "error"]:
                count(frame, column, stage, column.removesuffix("_error"), r.run)
    return pd.DataFrame(rows, columns=columns)


# ── reduction ───────────────────────────────────────────────────────────

def reduction_table(runs: List[RunData]) -> pd.DataFrame:
    """Every reduction answer of every run, with the scores computed in this
    environment plus those added by detectors in other environments
    (reduction_scores.csv), joined on run, question, model and condition.

    Score files are grouped by detector group (minicheck, summac, …) and
    each group is merged once. MiniCheck's and SummaC's
    files share the same keys, so pooling them before the merge would keep
    only one environment's columns."""
    frame = concat([r.reduction for r in runs])
    if frame.empty:
        return frame
    keys = ["run", "sample_id", "model", "condition"]
    by_env: Dict[str, list] = {}
    for r in runs:
        if r.reduction_scores is not None and not r.reduction_scores.empty:
            by_env.setdefault(r.group or str(r.folder.parent), []).append(r.reduction_scores)
    for env, frames in sorted(by_env.items()):
        extra = pd.concat(frames, ignore_index=True)
        new_cols = [c for c in extra.columns if c not in frame.columns and c != "error"]
        if new_cols:
            frame = frame.merge(extra[keys + new_cols].drop_duplicates(subset=keys), on=keys, how="left")
    return frame


def score_columns(frame: pd.DataFrame) -> List[str]:
    return [c for c in frame.columns if c.endswith("_score")]


def reduction_deltas(frame: pd.DataFrame) -> pd.DataFrame:
    """Paired change (method − baseline) per run, question, model, method
    and detector. Negative = the method's answer looks less hallucinated."""
    from benchmark.reduction_runner import paired_deltas

    if frame.empty:
        return pd.DataFrame()
    deltas = paired_deltas(frame, score_columns(frame))
    if deltas.empty:
        return deltas
    dataset = frame.drop_duplicates("sample_id").set_index("sample_id")["dataset"]
    deltas["dataset"] = deltas["sample_id"].map(dataset)
    return deltas


def reduction_by_run(deltas: pd.DataFrame, keys: List[str]) -> pd.DataFrame:
    """Per run and `keys` (always incl. condition + detector): mean change,
    share of questions better / worse than the baseline, and n."""
    if deltas.empty:
        return pd.DataFrame()
    grouped = deltas.groupby(["run", *keys], sort=False)["delta"]
    out = grouped.agg(n_pairs="count", mean_delta="mean").reset_index()   # (question, model) pairs
    out["better"] = grouped.apply(lambda s: float((s < 0).mean())).to_numpy()
    out["worse"] = grouped.apply(lambda s: float((s > 0).mean())).to_numpy()
    return out


def reduction_examples(frame: pd.DataFrame, deltas: pd.DataFrame, detector: str, n: int = 2) -> pd.DataFrame:
    """For each method: the questions where `detector` saw the largest
    improvement and the largest regression, with both answers."""
    if deltas.empty or detector not in set(deltas["detector"]):
        return pd.DataFrame()
    d = deltas[deltas["detector"] == detector]
    answers = frame.set_index(["run", "sample_id", "model", "condition"])["answer"]
    cut = lambda s: (str(s)[:200] + "…") if isinstance(s, str) and len(s) > 200 else s  # noqa: E731
    rows = []
    for cond, group in d.groupby("condition", sort=False):
        for kind, pick in (("improved most", group.nsmallest(n, "delta")),
                           ("got worse most", group.nlargest(n, "delta"))):
            for r in pick.itertuples(index=False):
                rows.append({"method": cond, "kind": kind, "model": r.model, "sample_id": r.sample_id,
                             f"Δ {detector}": r.delta,
                             "baseline answer": cut(answers.get((r.run, r.sample_id, r.model, "baseline"))),
                             "method answer": cut(answers.get((r.run, r.sample_id, r.model, cond)))})
    return pd.DataFrame(rows)


def per_unit(deltas: pd.DataFrame) -> pd.DataFrame:
    """One paired change per (question, model): repeated runs of the same
    question and model are averaged first, so the unit of analysis is the
    question × model pair, not the run (runs of one pair are not
    independent)."""
    if deltas.empty:
        return deltas
    return (deltas.groupby(["condition", "detector", "sample_id", "model", "dataset"], sort=False)["delta"]
            .mean().reset_index())


MIN_PAIRS_FOR_CI = 10   # below this a bootstrap interval is not meaningful


def bootstrap_ci(values, n_resamples: int = 2000, seed: int = 0, level: float = 0.95):
    """Percentile bootstrap confidence interval of the mean (seeded). With
    fewer than MIN_PAIRS_FOR_CI values no interval is given (NaN): resampling
    a handful of values produces intervals that look precise but are not."""
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if len(values) < MIN_PAIRS_FOR_CI:
        return (np.nan, np.nan)
    rng = np.random.default_rng(seed)
    means = rng.choice(values, size=(n_resamples, len(values)), replace=True).mean(axis=1)
    tail = (1 - level) / 2
    return float(np.quantile(means, tail)), float(np.quantile(means, 1 - tail))


def method_effects(units: pd.DataFrame, detectors: List[str], by: Optional[str] = None) -> pd.DataFrame:
    """Per method and detector (and optionally per model or dataset): mean
    paired change with a 95% bootstrap CI, and the share of question × model
    pairs that got better / worse."""
    keys = ["condition", "detector"] + ([by] if by else [])
    rows = []
    for key, group in units[units["detector"].isin(detectors)].groupby(keys, sort=False):
        d = group["delta"].to_numpy(dtype=float)
        low, high = bootstrap_ci(d)
        rows.append({**dict(zip(keys, key)), "n_pairs": len(d), "mean_change": float(np.mean(d)),
                     "ci_low": low, "ci_high": high,
                     "better": float((d < 0).mean()), "worse": float((d > 0).mean())})
    return pd.DataFrame(rows)


def constant_detectors(raw: pd.DataFrame) -> List[str]:
    """Detectors that gave every answer the same score: no signal at all."""
    if raw is None or raw.empty:
        return []
    cols = [c for c in raw.columns if c.endswith("_score")]
    return [c.removesuffix("_score") for c in cols if raw[c].dropna().nunique() == 1]


def is_number(value) -> bool:
    return isinstance(value, (int, float, np.floating)) and not pd.isna(value)
