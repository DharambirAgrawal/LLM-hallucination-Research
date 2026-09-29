#!/usr/bin/env python3
"""Turn a run folder into combined CSVs, PNG charts, and a REPORT.md.

Works on both layouts, so a smoke run and a full run produce the same files:

    single main.py run (main.py calls this itself at the end):
        <input>/detector_validation_summary.csv, detector_validation_raw.csv,
        reduction_comparison.csv (optional), run_manifest.json

    scripts/run_full.py run (one sub-folder per detector):
        <input>/<detector_name>/detector_validation_summary.csv   (one or more)
        <input>/<detector_name>/detector_validation_raw.csv       (one or more)
        <input>/selfcheckgpt/reduction_comparison.csv              (optional)

Produces, per detector: an overall (all datasets blended) comparison, a
per-dataset breakdown (does it do better on QA than summarization, etc.), and
— if the reduction stage ran — baseline-vs-refined scores and win rate
(fraction of samples that actually improved) by model and by dataset.

This aggregates and recomputes metrics with the same sklearn calls main.py
itself uses (`benchmark.detector_validation.DetectorValidator`); it does not
invent a new metric definition. Latency is intentionally not surfaced here —
this report is about output quality, not runtime — though `latency_seconds`
stays available in the raw `combined_reduction.csv` if it's ever needed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from benchmark.detector_validation import DetectorValidator  # noqa: E402

METRICS = ["roc_auc", "average_precision", "accuracy", "precision", "recall", "f1"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, help="A full-run output folder")
    return parser.parse_args()


def find(input_dir: Path, filename: str) -> list[Path]:
    """The run's own file when `input_dir` is a single main.py run,
    otherwise one per detector sub-folder (scripts/run_full.py layout)."""
    if (input_dir / filename).is_file():
        return [input_dir / filename]
    return sorted(input_dir.glob(f"*/{filename}"))


def load_summaries(input_dir: Path) -> pd.DataFrame:
    frames = []
    for summary_path in find(input_dir, "detector_validation_summary.csv"):
        frame = pd.read_csv(summary_path)
        frame["run_folder"] = summary_path.parent.name
        frames.append(frame)
    if not frames:
        raise SystemExit(f"No detector_validation_summary.csv found in {input_dir} or its sub-folders")
    return pd.concat(frames, ignore_index=True)


def load_reduction(input_dir: Path) -> pd.DataFrame | None:
    matches = find(input_dir, "reduction_comparison.csv")
    if not matches:
        return None
    return pd.concat((pd.read_csv(path) for path in matches), ignore_index=True)


def load_raws(input_dir: Path) -> pd.DataFrame | None:
    frames = []
    for raw_path in find(input_dir, "detector_validation_raw.csv"):
        frame = pd.read_csv(raw_path)
        frame["run_folder"] = raw_path.parent.name
        frames.append(frame)
    if not frames:
        return None
    return pd.concat(frames, ignore_index=True)


def threshold_map(summary: pd.DataFrame) -> dict[str, float]:
    return summary.drop_duplicates("detector").set_index("detector")["threshold"].to_dict()


def per_dataset_breakdown(raw: pd.DataFrame, thresholds: dict[str, float]) -> pd.DataFrame:
    """Metrics per (dataset, detector[, model]) instead of blended across
    every enabled dataset — `detector_validation_summary.csv` reports one
    number per detector over *all* loaded datasets combined, which hides
    whether a detector does better on QA than on summarization, say. This
    recomputes the same sklearn metrics main.py uses, just grouped finer,
    straight from each run's own `detector_validation_raw.csv`."""
    validator = DetectorValidator()
    has_model = "model" in raw.columns and raw["model"].notna().any()
    rows = []

    if "selfcheckgpt_score" in raw.columns and has_model:
        for (dataset_name, model_name), group in raw.groupby(["dataset", "model"]):
            usable = group[["label", "selfcheckgpt_score"]].dropna()
            if usable.empty or usable["label"].nunique() < 2:
                continue
            metrics = validator.evaluate(
                usable["label"], usable["selfcheckgpt_score"],
                thresholds.get("selfcheckgpt", 0.5),
            )
            metrics.update(detector="selfcheckgpt", dataset=dataset_name, model=model_name)
            rows.append(metrics)

    other_columns = [
        c for c in raw.columns if c.endswith("_score") and c != "selfcheckgpt_score"
    ]
    if other_columns:
        # Model-independent detector scores are duplicated once per model row
        # in the raw file (same case, scored once, attached to every
        # generator's rows) — de-duplicate by case before recomputing, same
        # as main.py does for the blended summary.
        dedup = raw.drop_duplicates(subset="case_id") if has_model else raw
        for dataset_name, group in dedup.groupby("dataset"):
            for column in other_columns:
                usable = group[["label", column]].dropna()
                if usable.empty or usable["label"].nunique() < 2:
                    continue
                detector = column.removesuffix("_score")
                metrics = validator.evaluate(
                    usable["label"], usable[column], thresholds.get(detector, 0.5),
                )
                metrics.update(
                    detector=detector, dataset=dataset_name,
                    model="n/a (model-independent detector)",
                )
                rows.append(metrics)

    return pd.DataFrame(rows)


def reduction_summary(reduction: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    """Grouped reduction stats: mean baseline/refined score, mean delta, and
    win_rate (fraction of samples where the refined answer scored lower —
    i.e. less hallucinated — than the baseline). This is what both the
    reduction charts and the reduction tables in REPORT.md are built from,
    so the bars and the numbers next to them always agree. Deliberately does
    not compute or surface latency — quality of the reduction, not its
    runtime, is what this report is for."""
    ok = reduction[reduction["error"].isna()] if "error" in reduction.columns else reduction
    if ok.empty:
        return pd.DataFrame()
    grouped = ok.groupby(group_cols)
    out = grouped.agg(
        n_samples=("score_delta", "count"),
        baseline_score=("baseline_score", "mean"),
        refined_score=("refined_score", "mean"),
        score_delta=("score_delta", "mean"),
    ).reset_index()
    out["win_rate"] = grouped["score_delta"].apply(lambda s: (s < 0).mean()).to_numpy()
    return out


def per_detector_means(summary: pd.DataFrame) -> pd.DataFrame:
    """One row per detector: mean over models for detectors scored per-model
    (SelfCheckGPT), pass-through for model-independent detectors."""
    return summary.groupby("detector", as_index=False)[METRICS].mean(numeric_only=True)


def chart_detector_comparison(summary: pd.DataFrame, charts_dir: Path) -> str:
    means = per_detector_means(summary).set_index("detector")
    metrics = [m for m in ["roc_auc", "precision", "recall", "f1"] if m in means.columns]
    ax = means[metrics].plot(kind="bar", figsize=(8, 5), rot=0)
    ax.set_ylabel("score")
    ax.set_ylim(0, 1)
    ax.set_title("Detector comparison (mean over models where applicable)")
    ax.legend(loc="lower right")
    plt.tight_layout()
    out = charts_dir / "detector_comparison.png"
    plt.savefig(out, dpi=150)
    plt.close()
    return out.name


def chart_selfcheckgpt_per_model(summary: pd.DataFrame, charts_dir: Path) -> str | None:
    rows = summary[
        (summary["detector"] == "selfcheckgpt")
        & summary["model"].notna()
        & (summary["model"] != "n/a (model-independent detector)")
    ]
    if rows.empty:
        return None
    plot_df = rows.set_index("model")[["roc_auc", "f1"]]
    ax = plot_df.plot(kind="bar", figsize=(8, 5), rot=30)
    ax.set_ylabel("score")
    ax.set_ylim(0, 1)
    ax.set_title("SelfCheckGPT: AUROC / F1 per generator model")
    plt.tight_layout()
    out = charts_dir / "selfcheckgpt_per_model.png"
    plt.savefig(out, dpi=150)
    plt.close()
    return out.name


def chart_per_dataset(breakdown: pd.DataFrame, charts_dir: Path) -> str | None:
    if breakdown.empty:
        return None
    pivot = (
        breakdown.groupby(["dataset", "detector"], as_index=False)["roc_auc"]
        .mean(numeric_only=True)
        .pivot(index="dataset", columns="detector", values="roc_auc")
    )
    ax = pivot.plot(kind="bar", figsize=(9, 5), rot=20)
    ax.set_ylabel("AUROC")
    ax.set_ylim(0, 1)
    ax.set_title("AUROC per dataset (task type) x detector")
    plt.tight_layout()
    out = charts_dir / "per_dataset_breakdown.png"
    plt.savefig(out, dpi=150)
    plt.close()
    return out.name


def chart_reduction_two_panel(summary_by: pd.DataFrame, group_col: str, title: str, filename: str, charts_dir: Path) -> str | None:
    """Left panel: mean baseline vs. refined score. Right panel: win rate
    (fraction of samples the refined answer actually improved on)."""
    if summary_by.empty:
        return None
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

    plot_df = summary_by.set_index(group_col)[["baseline_score", "refined_score"]]
    plot_df.plot(kind="bar", ax=ax1, rot=25)
    top = plot_df.max().max()
    if pd.notna(top) and top > 0:
        ax1.set_ylim(0, top * 1.3)  # headroom so the legend never covers a bar
    ax1.legend(loc="upper right", ncol=2)
    ax1.set_ylabel("SelfCheckGPT hallucination score (lower = better)")
    ax1.set_title(f"{title}: baseline vs. refined")

    win_df = summary_by.set_index(group_col)["win_rate"]
    win_df.plot(kind="bar", ax=ax2, rot=25, color="#2ca02c")
    ax2.set_ylabel("win rate (fraction improved)")
    ax2.set_ylim(0, 1)
    ax2.set_title(f"{title}: reduction win rate")

    plt.tight_layout()
    out = charts_dir / filename
    plt.savefig(out, dpi=150)
    plt.close()
    return out.name


def to_markdown_table(frame: pd.DataFrame) -> str:
    def fmt(value: object) -> str:
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    header = "| " + " | ".join(frame.columns) + " |"
    divider = "|" + "|".join(["---"] * len(frame.columns)) + "|"
    rows = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join([header, divider, *rows])


def run_details(input_dir: Path) -> list[str]:
    """Markdown lines describing what produced each run: command, commit,
    duration, dataset checksums and model digests (from run_manifest.json)."""
    lines = []
    for path in find(input_dir, "run_manifest.json"):
        m = json.loads(path.read_text(encoding="utf-8"))
        where = "" if path.parent == input_dir else f" — `{path.parent.name}`"
        git = m.get("git", {})
        lines += [
            f"### Run details{where}",
            "",
            f"- command: `{m.get('command')}`",
            f"- started {m.get('started_at')}, took {m.get('duration_seconds')} s "
            f"(stages: {m.get('stage_seconds')})",
            f"- repository commit `{git.get('commit')}`"
            + (" **with uncommitted changes**" if git.get("uncommitted_changes") else ""),
            f"- seed {m.get('seed')}, Python {m.get('python')}, "
            + ", ".join(f"{k} {v}" for k, v in m.get("packages", {}).items()),
            "",
        ]
        if m.get("datasets"):
            lines += ["| dataset | samples | sha256 |", "|---|---|---|"]
            lines += [f"| {d['name']} | {d['n_samples']} | `{(d.get('sha256') or '—')[:16]}` |"
                      for d in m["datasets"]]
            lines.append("")
        if m.get("models"):
            lines += ["| model | tag | size | quant | digest |", "|---|---|---|---|---|"]
            lines += [f"| {g['name']} | {g.get('model')} | {g.get('parameter_size') or '—'} | "
                      f"{g.get('quantization') or '—'} | `{(g.get('digest') or '—')[:19]}` |"
                      for g in m["models"]]
            lines.append("")
    return lines


def write_report(
    input_dir: Path,
    summary: pd.DataFrame,
    breakdown: pd.DataFrame | None,
    reduction_by_model: pd.DataFrame,
    reduction_by_dataset: pd.DataFrame,
    reduction_by_dataset_model: pd.DataFrame,
    chart_files: list[str],
) -> Path:
    lines = [
        f"# Run report — {input_dir.name}",
        "",
        "Built by `scripts/generate_report.py` from the run's "
        "`detector_validation_summary.csv`/`detector_validation_raw.csv` (and "
        "`reduction_comparison.csv` for SelfCheckGPT). A `scripts/run_full.py` "
        "run combines one `main.py` run per detector, each in its own venv. "
        "See `docs/HOW_TO_RUN.md`.",
        "",
        "Not a reportable research result on its own — no held-out split, "
        "confidence intervals, or human review yet (see `docs/REPRODUCIBILITY.md`). "
        "This confirms the pipeline and shows the current numbers.",
        "",
        "## Detector validation summary (all enabled datasets blended)",
        "",
        to_markdown_table(summary.drop(columns=["run_folder"], errors="ignore")),
        "",
    ]
    if breakdown is not None and not breakdown.empty:
        lines += [
            "## Detector validation, broken out per dataset (task type)",
            "",
            "Same metrics as above, computed separately for each dataset "
            "instead of blended — this is where you see whether a detector "
            "does better on QA than on dialogue or summarization.",
            "",
            to_markdown_table(breakdown[
                ["dataset", "detector", "model", "n_cases", "roc_auc", "precision", "recall", "f1"]
            ]),
            "",
        ]
    for chart in chart_files:
        title = chart.replace(".png", "").replace("_", " ")
        lines += [f"## {title}", "", f"![{title}](charts/{chart})", ""]

    if not reduction_by_dataset_model.empty:
        lines += [
            "## Reduction: baseline vs. refined, per model x dataset",
            "",
            "`win_rate` is the fraction of samples where the refined answer "
            "scored lower (less hallucinated) than the baseline — a clearer "
            "improvement signal than the mean score alone when only a few "
            "samples flip. Row-level data (every sample, every iteration) is "
            "in `combined_reduction.csv`, not repeated here.",
            "",
            to_markdown_table(reduction_by_dataset_model),
            "",
        ]
        lines += [
            "### Reduction, by model (all datasets combined)",
            "",
            to_markdown_table(reduction_by_model),
            "",
            "### Reduction, by dataset (all models combined)",
            "",
            to_markdown_table(reduction_by_dataset),
            "",
        ]
    lines += ["## How this run was produced", "", *run_details(input_dir)]
    report_path = input_dir / "REPORT.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")
    return report_path


def generate(input_dir: Path) -> dict[str, Path]:
    """Write every combined CSV, chart, and REPORT.md into `input_dir`;
    return the produced files keyed by a short label."""
    charts_dir = input_dir / "charts"
    charts_dir.mkdir(parents=True, exist_ok=True)
    produced: dict[str, Path] = {}

    summary = load_summaries(input_dir)
    summary.to_csv(input_dir / "combined_summary.csv", index=False)
    produced["combined summary"] = input_dir / "combined_summary.csv"

    reduction = load_reduction(input_dir)
    reduction_by_model = pd.DataFrame()
    reduction_by_dataset = pd.DataFrame()
    reduction_by_dataset_model = pd.DataFrame()
    if reduction is not None:
        reduction.to_csv(input_dir / "combined_reduction.csv", index=False)
        produced["reduction rows"] = input_dir / "combined_reduction.csv"
        reduction_by_model = reduction_summary(reduction, ["model"])
        reduction_by_dataset = reduction_summary(reduction, ["dataset"])
        reduction_by_dataset_model = reduction_summary(reduction, ["dataset", "model"])
        if not reduction_by_dataset_model.empty:
            reduction_by_dataset_model.to_csv(
                input_dir / "combined_reduction_summary.csv", index=False
            )
            produced["reduction summary"] = input_dir / "combined_reduction_summary.csv"

    raw = load_raws(input_dir)
    breakdown = None
    if raw is not None:
        breakdown = per_dataset_breakdown(raw, threshold_map(summary))
        if not breakdown.empty:
            breakdown.to_csv(input_dir / "combined_per_dataset_breakdown.csv", index=False)
            produced["per-dataset"] = input_dir / "combined_per_dataset_breakdown.csv"

    chart_files = [chart_detector_comparison(summary, charts_dir)]
    per_model_chart = chart_selfcheckgpt_per_model(summary, charts_dir)
    if per_model_chart:
        chart_files.append(per_model_chart)
    if breakdown is not None and not breakdown.empty:
        per_dataset_chart = chart_per_dataset(breakdown, charts_dir)
        if per_dataset_chart:
            chart_files.append(per_dataset_chart)
    reduction_chart = chart_reduction_two_panel(
        reduction_by_model, "model", "Reduction by model",
        "reduction_by_model.png", charts_dir,
    )
    if reduction_chart:
        chart_files.append(reduction_chart)
    reduction_dataset_chart = chart_reduction_two_panel(
        reduction_by_dataset, "dataset", "Reduction by dataset",
        "reduction_by_dataset.png", charts_dir,
    )
    if reduction_dataset_chart:
        chart_files.append(reduction_dataset_chart)
    produced["charts"] = charts_dir

    produced["report"] = write_report(
        input_dir, summary, breakdown,
        reduction_by_model, reduction_by_dataset, reduction_by_dataset_model,
        chart_files,
    )
    return produced


def main() -> None:
    args = parse_args()
    for label, path in generate(Path(args.input)).items():
        print(f"  {label:<18} {path}")


if __name__ == "__main__":
    main()
