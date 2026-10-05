"""Preflight-based scheduling estimates, independent of experimental scores.

The planning range is a heuristic, not a statistical confidence interval.
Nothing in this module supplies scores or changes the measured run protocol.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timezone
from pathlib import Path

NOTES = [
    "Based on one median-length question; not a guaranteed minimum or a confidence interval.",
    "Long answers/contexts, judge calls, model switching, CPU fallback, retries and reducer early stopping can change runtime substantially.",
    "Setup/preflight already elapsed is shown separately. Report export, file I/O and worker process startup are not timed by this estimate.",
]


def reduction_answers_per_question(config: dict) -> int:
    """Answer rows the core sends to fixed detectors, including the baseline."""
    reduction = config.get("reduction", {})
    methods = reduction.get("methods")
    if methods is None:
        # Match ReductionRunner's legacy/default method selection without
        # importing neural libraries into the multi-environment controller.
        methods = ([reduction["method"]] if reduction.get("method") else
                   ["closed_book", "greedy", "self_refine_adapted", "cove_adapted", "uqlm_best_response"])
    selected = config.get("selected_models")
    models = [m for m in config.get("models", []) if not selected or m["name"] in selected]
    return len(models) * (1 + len(methods))


def build_estimate(stages: dict, runs: int, *, calibration: dict | None = None,
                   elapsed_seconds: float = 0.0, group: str = "standalone",
                   reload_fixed_detectors: bool = False) -> dict:
    """Scale measured stage workloads, counting cold detector startup once per worker/run."""
    calibration = calibration or {}
    stages = dict(stages)
    if reload_fixed_detectors:
        stages["detector_loading"] = calibration.get("fixed_detector_startup_seconds", 0.0)
    else:
        # Standalone runs retain these detector objects after preflight.
        stages.pop("sampling_detector_loading", None)
    if runs < 1 or not stages or any(not math.isfinite(v) or v < 0 for v in stages.values()):
        raise ValueError("Runtime estimate needs positive runs and finite nonnegative stage timings")
    per_run = sum(stages.values())
    if per_run <= 0:
        raise ValueError("No usable preflight timing measurements")
    remaining = runs * per_run
    return {
        "schema_version": 1, "source": "preflight_timings",
        "created_at": datetime.now(timezone.utc).isoformat(), "group": group,
        "runs": runs, "stages_seconds_per_run": stages,
        "seconds_per_run": per_run, "remaining_compute_seconds": remaining,
        "planning_range_seconds": [remaining * 0.75, remaining * 2.0],
        "elapsed_setup_preflight_seconds": elapsed_seconds,
        "calibration": calibration, "notes": list(NOTES),
    }


def combine_estimates(groups: dict[str, dict], runs: int, *, elapsed_seconds: float = 0.0) -> dict:
    """Detector groups run sequentially, so add their times, never average them."""
    if not groups:
        raise ValueError("No detector group timings available")
    stages = {}
    for name, estimate in groups.items():
        if estimate.get("schema_version") != 1 or estimate.get("source") != "preflight_timings":
            raise ValueError(f"Invalid timing data for {name}")
        for stage, seconds in estimate["stages_seconds_per_run"].items():
            stages[f"{name}: {stage}"] = seconds
    result = build_estimate(stages, runs, elapsed_seconds=elapsed_seconds, group="all_selected_groups")
    result["groups"] = groups
    return result


def save_estimate(path: Path, estimate: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(estimate, indent=2, allow_nan=False), encoding="utf-8")


def print_estimate(estimate: dict, path: Path) -> None:
    from utils import console
    console.section("Runtime estimate · from this machine's preflight")
    groups = estimate.get("groups")
    if groups:
        for name, group in groups.items():
            console.kv(name, f"~{console.duration(group['seconds_per_run'])} per run")
        counts = [g.get("calibration", {}) for g in groups.values()]
        questions = sorted({g["questions"] for g in counts if "questions" in g})
        if questions:
            console.kv("loaded questions", "/".join(map(str, questions)) + " per run")
    else:
        for stage, seconds in estimate["stages_seconds_per_run"].items():
            if seconds:
                console.kv(stage.replace("_", " "), f"~{console.duration(seconds)} per run")
        if "questions" in estimate["calibration"]:
            console.kv("loaded questions", f"{estimate['calibration']['questions']} per run")
    console.kv("one complete run", f"~{console.duration(estimate['seconds_per_run'])}")
    console.kv("remaining compute", f"{estimate['runs']} run(s): ~{console.duration(estimate['remaining_compute_seconds'])}")
    low, high = estimate["planning_range_seconds"]
    console.kv("planning range", f"~{console.duration(low)} to ~{console.duration(high)} for all runs")
    console.kv("already elapsed", console.duration(estimate['elapsed_setup_preflight_seconds']))
    console.line("Approximate, not a minimum; one probe cannot predict every answer length or retry.")
    console.line("Allow additional time for reports, file writing and worker startup.")
    console.kv("estimate saved", path)
