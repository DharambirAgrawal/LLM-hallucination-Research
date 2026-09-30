"""Build every table, chart and report for a run folder or a set of runs.

    generate(run_dir)                         → report of that one run
    generate(parent, out_dir=parent/combined) → every run under `parent`
                                                combined (mean ± std)

Both write the same layout into the output folder:

    REPORT.md, report.html, report.docx   same content, three formats
    takeaways.md                          the key findings on their own
    charts/*.png                          every figure in the report
    tables/*.csv                          every table in the report
    (combined only) raw_all_runs.csv, reduction_all_runs.csv
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from loguru import logger  # noqa: E402

from reporting import aggregate as agg  # noqa: E402
from reporting.document import Report  # noqa: E402

STATUS = ("Engineering results, not reportable research evidence yet: thresholds are not "
          "calibrated on a held-out split, there are no confidence intervals beyond "
          "run-to-run std, and no human review (see docs/REPRODUCIBILITY.md).")
DETECTOR_SOURCES = {
    "selfcheckgpt": "SelfCheckGPT, official package (Manakul et al., 2023)",
    "uqlm": "UQLM, official package (Bouchard et al., 2025)",
    "uqlm_judge": "UQLM LLM-as-a-judge, official package",
    "minicheck": "MiniCheck, official package (Tang et al., 2024)",
    "summac": "SummaC, official package (Laban et al., 2022)",
    "alignscore": "AlignScore, official package (Zha et al., 2023)",
}
METHOD_NOTES = {
    "closed_book": "no context in the prompt; the gap to the baseline is what RAG (the context) adds",
    "greedy": "temperature 0 decoding; a decoding setting, not a published method",
    "self_refine_adapted": "Self-Refine loop (Madaan et al., 2023); local inspired adaptation",
    "cove_adapted": "Chain-of-Verification, factored (Dhuliawala et al., 2023); local implementation, no official code exists",
    "uqlm_best_response": "UQLM semantic-entropy best-response selection; official implementation",
}


def pm(mean, std) -> str:
    if not agg.is_number(mean):
        return "—"
    return f"{mean:.3f}" if not agg.is_number(std) else f"{mean:.3f} ± {std:.3f}"


def short_model(model: str) -> str:
    return "—" if model == agg.MODEL_INDEPENDENT else str(model)


def family_of(detector: str) -> str:
    for fam in ("uqlm_judge", "selfcheckgpt", "uqlm", "minicheck", "summac", "alignscore"):
        if detector == fam or detector.startswith(fam + "_"):
            return fam
    return detector


# ── charts ──────────────────────────────────────────────────────────────

def _save(fig, path: Path) -> Path:
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def heatmap(table: pd.DataFrame, title: str, path: Path, cmap: str, vmin: float, vmax: float,
            fmt: str = "{:.2f}", center: Optional[float] = None) -> Optional[Path]:
    """Annotated heatmap of a numeric table (rows × columns)."""
    if table is None or table.empty:
        return None
    data = table.to_numpy(dtype=float)
    fig, ax = plt.subplots(figsize=(max(6, 0.9 * table.shape[1] + 3), max(3, 0.42 * table.shape[0] + 1.6)))
    if center is not None:
        from matplotlib.colors import TwoSlopeNorm
        span = max(abs(np.nanmin(data - center)) if np.isfinite(data).any() else 1,
                   abs(np.nanmax(data - center)) if np.isfinite(data).any() else 1, 1e-6)
        norm = TwoSlopeNorm(vcenter=center, vmin=center - span, vmax=center + span)
        image = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")
    else:
        image = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(table.shape[1]), [str(c) for c in table.columns], rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(table.shape[0]), [str(i) for i in table.index], fontsize=8)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isfinite(data[i, j]):
                ax.text(j, i, fmt.format(data[i, j]), ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, shrink=0.8)
    ax.set_title(title, fontsize=10)
    return _save(fig, path)


def consistency_chart(by_run: pd.DataFrame, path: Path) -> Optional[Path]:
    """AUROC of every detector in every run (mean over models for the
    generator-dependent ones): flat lines = stable across runs."""
    if by_run.empty or by_run["run"].nunique() < 2:
        return None
    per = by_run.groupby(["detector", "run"])["roc_auc"].mean().unstack("run")
    fig, ax = plt.subplots(figsize=(9, 5))
    for detector, row in per.iterrows():
        ax.plot(list(row.index), row.values, marker="o", label=detector)
    ax.set_ylim(0, 1)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=1)
    ax.set_title("AUROC in each run (mean over models; dashed line = chance)")
    ax.set_ylabel("AUROC")
    ax.legend(fontsize=7, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.grid(alpha=0.3)
    return _save(fig, path)


# ── report content ──────────────────────────────────────────────────────

def run_plan(runs: List[agg.RunData], summary: pd.DataFrame, reduction: pd.DataFrame) -> List[str]:
    manifests = [r.manifest for r in runs if r.manifest]
    configs = [r.config for r in runs if r.config]
    items = [f"Runs: {len({r.run for r in runs})} ({', '.join(sorted({r.run for r in runs}))}); "
             "the same questions in every run, fresh model sampling in each."]
    if manifests:
        datasets = {d["name"]: d for m in manifests for d in m.get("datasets", [])}
        items.append("Datasets (questions per run): " + ", ".join(
            f"{n} {d['n_samples']}" for n, d in datasets.items())
            + f" → {sum(d['n_samples'] for d in datasets.values())} questions")
        models = {g["name"]: g for m in manifests for g in m.get("models", [])}
        if models:
            items.append("Generator models: " + ", ".join(
                f"{g['name']} ({g.get('model')}, {g.get('parameter_size') or '?'}, "
                f"{g.get('quantization') or '?'})" for g in models.values()))
    families = set(family_of(d) for d in summary["detector"].unique())
    items.append("Detectors: " + "; ".join(DETECTOR_SOURCES.get(f, f) for f in sorted(families)))
    if configs:
        cfg = configs[0]
        det = cfg.get("detectors", {})
        if "selfcheckgpt" in families:
            sc = det.get("selfcheckgpt", {})
            items.append(f"SelfCheckGPT scorers: {', '.join(sc.get('methods') or [sc.get('method', 'ngram')])}; "
                         f"{sc.get('n_samples')} samples per question at temperature {sc.get('temperature')}")
        if "uqlm" in families:
            items.append(f"UQLM scorers: {', '.join(det.get('uqlm', {}).get('scorers') or [])} (use_best=False)")
        if families & {"uqlm_judge"} or "selfcheckgpt_prompt" in set(summary["detector"]):
            items.append(f"Judge model (UQLM judge, SelfCheckGPT prompt): {cfg.get('judge', {}).get('model')}")
        for name in ("minicheck", "summac", "alignscore"):
            if name in families:
                items.append(f"{name}: model {det.get(name, {}).get('model_name')}")
        if not reduction.empty:
            red = cfg.get("reduction", {})
            items.append("Reduction methods vs. baseline: " + "; ".join(
                f"{m} ({METHOD_NOTES.get(m, '')})" for m in red.get("methods", [])))
            items.append(f"self_refine_adapted: up to {red.get('max_iterations')} rounds")
        else:
            items.append("Reduction: not run in this folder")
        items.append(f"Seed: {cfg.get('benchmark', {}).get('seed')}")
    if manifests:
        m = manifests[0]
        git = m.get("git", {})
        items.append(f"Code: commit {str(git.get('commit'))[:12]}"
                     + (" (with uncommitted changes)" if git.get("uncommitted_changes") else ""))
        items.append(f"Environment: Python {m.get('python')}, " + ", ".join(
            f"{k} {v}" for k, v in m.get("packages", {}).items()))
        seconds = sum(x.get("duration_seconds") or 0 for x in manifests)
        items.append(f"Run time: {seconds / 60:.1f} min over {len(manifests)} run folder(s)")
    return items


def takeaways(summary_ms: pd.DataFrame, dataset_ms: pd.DataFrame, red_ms: pd.DataFrame,
              fails: pd.DataFrame, n_runs: int) -> List[str]:
    out = []
    if not summary_ms.empty:
        ranked = summary_ms.sort_values("roc_auc_mean", ascending=False)
        best = ranked.iloc[0]
        who = best["detector"] + ("" if best["model"] == agg.MODEL_INDEPENDENT else f" with {best['model']}")
        out.append(f"Best separation of factual vs. hallucinated answers: {who}, AUROC "
                   f"{pm(best['roc_auc_mean'], best['roc_auc_std'])}.")
        per_detector = summary_ms.groupby("detector")["roc_auc_mean"].mean().sort_values(ascending=False)
        out.append("AUROC by detector (mean over models where it depends on the model): " + ", ".join(
            f"{d} {v:.3f}" for d, v in per_detector.items()) + ".")
        near_chance = [d for d, v in per_detector.items() if v < 0.6]
        if near_chance:
            out.append(f"Close to chance (AUROC < 0.6): {', '.join(near_chance)}.")
        if n_runs > 1 and summary_ms["roc_auc_std"].notna().any():
            worst = summary_ms.sort_values("roc_auc_std", ascending=False).iloc[0]
            out.append(f"Run-to-run stability: largest AUROC std {worst['roc_auc_std']:.3f} "
                       f"({worst['detector']} · {short_model(worst['model'])}) over {n_runs} runs.")
    if not dataset_ms.empty:
        by_ds = dataset_ms.groupby("dataset")["roc_auc_mean"].mean().sort_values()
        if len(by_ds) > 1:
            out.append(f"Hardest dataset for detection (mean AUROC over detectors): {by_ds.index[0]} "
                       f"({by_ds.iloc[0]:.3f}); easiest: {by_ds.index[-1]} ({by_ds.iloc[-1]:.3f}).")
    if not red_ms.empty:
        for cond, group in red_ms.groupby("condition", sort=False):
            lower = int((group["mean_delta_mean"] < 0).sum())
            higher = int((group["mean_delta_mean"] > 0).sum())
            out.append(f"Reduction · {cond}: lower hallucination risk than the baseline answer by "
                       f"{lower} of {len(group)} detectors, higher by {higher} "
                       f"(mean paired change per detector in the Stage B table).")
    total_failed = int(fails["count"].sum()) if not fails.empty else 0
    out.append("No failures." if total_failed == 0 else
               f"{total_failed} failure(s) — see the Failures section; failed items are excluded "
               "from the metrics, never scored as zero.")
    return out


def display_ms(frame: pd.DataFrame, keys: List[str], cols: Dict[str, str]) -> pd.DataFrame:
    """Readable table: `mean ± std` strings for each metric."""
    if frame.empty:
        return frame
    out = frame[[k for k in keys if k in frame.columns]].copy()
    if "model" in out.columns:
        out["model"] = out["model"].map(short_model)
    for label, col in cols.items():
        if f"{col}_mean" in frame.columns:
            std = frame[f"{col}_std"] if f"{col}_std" in frame.columns else [None] * len(frame)
            out[label] = [pm(m, s) for m, s in zip(frame[f"{col}_mean"], std)]
    return out


def generate(input_dir: Path, out_dir: Optional[Path] = None, title: Optional[str] = None) -> Dict[str, Path]:
    input_dir = Path(input_dir)
    out_dir = Path(out_dir or input_dir)
    charts = out_dir / "charts"
    tables = out_dir / "tables"
    charts.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    combined = out_dir != input_dir

    runs = agg.discover(input_dir)
    n_runs = len({r.run for r in runs})
    by_run = agg.summary_by_run(runs)
    metric_cols = [m for m in agg.METRICS if m in by_run.columns]
    summary_ms = agg.mean_std(by_run, ["detector", "model"], metric_cols + ["n_cases", "n_failed"])
    per_model = set(by_run.loc[by_run["model"] != agg.MODEL_INDEPENDENT, "detector"])
    dataset_by_run = agg.per_dataset_by_run(runs, agg.thresholds(by_run), per_model)
    dataset_ms = agg.mean_std(dataset_by_run, ["dataset", "detector", "model"], metric_cols + ["n_cases", "n_failed"]) \
        if not dataset_by_run.empty else pd.DataFrame()

    reduction = agg.reduction_table(runs)
    deltas = agg.reduction_deltas(reduction)
    red = {}
    for name, keys in (("overall", ["condition", "detector"]),
                       ("by_model", ["condition", "detector", "model"]),
                       ("by_dataset", ["condition", "detector", "dataset"])):
        per_run = agg.reduction_by_run(deltas, keys)
        red[name] = (per_run, agg.mean_std(per_run, keys, ["n_questions", "mean_delta", "better", "worse"])
                     if not per_run.empty else pd.DataFrame())
    costs_run = agg.condition_costs(reduction)
    costs = agg.mean_std(costs_run, ["model", "condition"],
                         ["n_answers", "calls_per_answer", "seconds_per_answer", "failed"]) \
        if not costs_run.empty else pd.DataFrame()
    example_detector = next((d for d in ("uqlm_judge", "minicheck", "selfcheckgpt_nli")
                             if not deltas.empty and d in set(deltas["detector"])),
                            deltas["detector"].iloc[0] if not deltas.empty else None)
    examples = agg.reduction_examples(reduction, deltas, example_detector) if example_detector else pd.DataFrame()
    fails = agg.failures(runs)

    # tables: combined folders get per-run and mean ± std; a run folder one each
    pairs = {"summary": (by_run, summary_ms), "per_dataset": (dataset_by_run, dataset_ms),
             "reduction_overall": red["overall"], "reduction_by_model": red["by_model"],
             "reduction_by_dataset": red["by_dataset"], "reduction_costs": (costs_run, costs)}
    for name, (per_run, ms) in pairs.items():
        if combined:
            for suffix, frame in (("by_run", per_run), ("mean_std", ms)):
                if frame is not None and not frame.empty:
                    frame.to_csv(tables / f"{name}_{suffix}.csv", index=False)
        elif per_run is not None and not per_run.empty and name != "summary":
            per_run.drop(columns=["run"], errors="ignore").to_csv(tables / f"{name}.csv", index=False)
    for name, frame in {"reduction_examples": examples, "failures": fails}.items():
        if frame is not None and not frame.empty:
            frame.to_csv(tables / f"{name}.csv", index=False)
    if combined:
        agg.concat([r.raw for r in runs]).to_csv(out_dir / "raw_all_runs.csv", index=False)
        if not reduction.empty:
            reduction.to_csv(out_dir / "reduction_all_runs.csv", index=False)

    # charts
    auroc = summary_ms.assign(model=summary_ms["model"].map(short_model)) \
        .pivot_table(index="detector", columns="model", values="roc_auc_mean", sort=False)
    fig_auroc = heatmap(auroc, "AUROC per detector and generator model (— = does not use a model)",
                        charts / "detector_auroc.png", "viridis", 0.4, 1.0)
    auprc = summary_ms.assign(model=summary_ms["model"].map(short_model)) \
        .pivot_table(index="detector", columns="model", values="average_precision_mean", sort=False)
    fig_auprc = heatmap(auprc, "AUPRC per detector and generator model", charts / "detector_auprc.png",
                        "viridis", 0.0, 1.0)
    fig_dataset = None
    if not dataset_ms.empty:
        ds = dataset_ms.pivot_table(index="detector", columns="dataset", values="roc_auc_mean",
                                    aggfunc="mean", sort=False)
        fig_dataset = heatmap(ds, "AUROC per detector and dataset (mean over models where it depends on one)",
                              charts / "per_dataset_auroc.png", "viridis", 0.4, 1.0)
    fig_consistency = consistency_chart(by_run, charts / "run_consistency.png")
    fig_red = fig_better = fig_red_model = fig_cost = None
    if not red["overall"][1].empty:
        overall = red["overall"][1]
        fig_red = heatmap(overall.pivot_table(index="condition", columns="detector", values="mean_delta_mean", sort=False),
                          "Mean paired change in hallucination risk vs. baseline answer (green = lower risk)",
                          charts / "reduction_delta.png", "RdYlGn_r", 0, 0, "{:+.2f}", center=0.0)
        fig_better = heatmap(overall.pivot_table(index="condition", columns="detector", values="better_mean", sort=False),
                             "Share of questions where the method's answer scored lower risk than the baseline",
                             charts / "reduction_better_share.png", "Greens", 0.0, 1.0, "{:.0%}")
        by_model = red["by_model"][1]
        judge_col = example_detector
        if judge_col:
            fig_red_model = heatmap(
                by_model[by_model["detector"] == judge_col].pivot_table(
                    index="condition", columns="model", values="mean_delta_mean", sort=False),
                f"Mean change vs. baseline per model ({judge_col})", charts / "reduction_delta_by_model.png",
                "RdYlGn_r", 0, 0, "{:+.2f}", center=0.0)
    if not costs.empty:
        fig_cost = heatmap(costs.pivot_table(index="condition", columns="model", values="seconds_per_answer_mean", sort=False),
                           "Seconds per answer (generation only)", charts / "reduction_seconds.png",
                           "Blues", 0, float(costs["seconds_per_answer_mean"].max() or 1), "{:.1f}")

    # report
    name = title or (f"{input_dir.name} · combined over {n_runs} run(s)" if combined else input_dir.name)
    report = Report(f"Hallucination benchmark — {name}",
                    f"Generated {datetime.now():%Y-%m-%d %H:%M} by reporting/build.py from the run's own CSV files.")
    report.p(STATUS)
    report.h1("Run plan")
    report.bullets(run_plan(runs, by_run, reduction))
    red_ms = red["overall"][1]
    points = takeaways(summary_ms, dataset_ms, red_ms, fails, n_runs)
    report.h1("Key takeaways")
    report.bullets(points)

    report.h1("Stage A · Detector validation")
    report.p("Each detector scores fixed answers whose label is known (0 = faithful, 1 = hallucinated): "
             "HaluEval's correct/hallucinated pairs, RAGTruth's human-annotated LLM answers and HaluBench's "
             "PASS/FAIL answers. Every score is oriented the same way (higher = more likely hallucinated; "
             "UQLM confidences are reported as 1 − confidence). AUROC/AUPRC are threshold-free; accuracy, "
             "precision, recall and F1 use the configured, not yet calibrated threshold. Sampling-based "
             "detectors (SelfCheckGPT, UQLM) are reported per generator model. In this stage the checked "
             "answers come from the datasets, not from our models, so the sampling-based detectors measure "
             "how well a given answer agrees with our models' own answers. Generator-independent detectors "
             "(UQLM judge, MiniCheck, SummaC, AlignScore) score each answer once and later runs reuse it, so "
             "their run-to-run std is 0 by construction.")
    report.figure(fig_auroc, "AUROC per detector and generator model (0.5 = chance).")
    report.table(display_ms(summary_ms, ["detector", "model", "n_runs"],
                            {"cases": "n_cases", "failed": "n_failed", "AUROC": "roc_auc",
                             "AUPRC": "average_precision", "accuracy": "accuracy",
                             "precision": "precision", "recall": "recall", "F1": "f1"}),
                 f"Mean ± std over {n_runs} runs:" if n_runs > 1 else "Results:")
    report.figure(fig_auprc, "AUPRC per detector and generator model.")
    if not dataset_ms.empty:
        report.h2("By dataset (task type)")
        report.figure(fig_dataset, "AUROC per detector and dataset.")
        report.table(display_ms(dataset_ms, ["dataset", "detector", "model"],
                                {"cases": "n_cases", "failed": "n_failed", "AUROC": "roc_auc",
                                 "AUPRC": "average_precision", "F1": "f1"}),
                     "Failed cases are excluded; a high failure count on one dataset (e.g. SelfCheckGPT "
                     "BERTScore on very short answers) means that detector's numbers there cover fewer answers.")

    if not red_ms.empty:
        report.h1("Stage B · Reduction methods vs. baseline")
        report.p("For every question, each model first gives its grounded baseline answer (context in the "
                 "prompt). Each reduction method then produces its own answer to the same question, and "
                 "every detector scores every answer against the same samples. The numbers are paired "
                 "differences (method − baseline) on the same question and model: negative = the method's "
                 "answer looks less hallucinated to that detector. 'better'/'worse' = share of questions "
                 "where the risk went down/up.")
        report.bullets([f"{m}: {note}" for m, note in METHOD_NOTES.items()
                        if m in set(red_ms["condition"])])
        if "uqlm_best_response" in set(red_ms["condition"]):
            report.p("uqlm_best_response: UQLM returns the most repeated answer of the most probable meaning "
                     "cluster, otherwise the longest one (length bias). The picked answer is scored leave-one-out "
                     "(against the baseline and the other samples), never against itself.")
        report.figure(fig_red, "Mean paired change in risk, per method and detector.")
        report.figure(fig_better, "Share of questions improved, per method and detector.")
        report.table(display_ms(red_ms, ["condition", "detector", "n_runs"],
                                {"questions": "n_questions", "Δ risk": "mean_delta",
                                 "better": "better", "worse": "worse"}), "Per method and detector:")
        report.h2("By model")
        report.figure(fig_red_model, f"Mean change vs. baseline per model, as seen by {example_detector}.")
        report.table(display_ms(red["by_model"][1], ["condition", "detector", "model"],
                                {"Δ risk": "mean_delta", "better": "better", "worse": "worse"}))
        report.h2("By dataset")
        report.table(display_ms(red["by_dataset"][1], ["condition", "detector", "dataset"],
                                {"Δ risk": "mean_delta", "better": "better", "worse": "worse"}))
        report.h2("Cost")
        report.figure(fig_cost, "Generation seconds per answer, per method and model.")
        report.table(display_ms(costs, ["model", "condition"],
                                {"answers": "n_answers", "calls/answer": "calls_per_answer",
                                 "sec/answer": "seconds_per_answer", "failed": "failed"}))
        if not examples.empty:
            report.h2("Examples")
            report.table(examples, f"Per method: the largest improvements and regressions according to "
                                   f"{example_detector} (answers truncated):")

    if n_runs > 1:
        report.h1("Run by run")
        report.figure(fig_consistency, "AUROC in every run: flat lines mean the result is stable.")
        report.table(by_run[[c for c in ["run", "detector", "model", "n_cases", "n_failed", "roc_auc",
                                         "average_precision", "f1"] if c in by_run.columns]]
                     .assign(model=lambda f: f["model"].map(short_model)))
        if not red["overall"][0].empty:
            report.table(red["overall"][0], "Reduction per run:")

    report.h1("Failures")
    if fails.empty:
        report.p("Nothing failed in any run.")
    else:
        report.table(fails, "Failed items are excluded from the metrics (never scored as 0). Tracebacks: run.log.")

    report.h1("Files")
    report.bullets((["detector_validation_summary.csv", "detector_validation_raw.csv"]
                    + (["reduction_comparison.csv"] if not reduction.empty else [])
                    + ["selfcheckgpt_samples.jsonl", "run_manifest.json", "config_used.yaml"]
                    if not combined else [])
                   + [f"{p.relative_to(out_dir).as_posix()}" for p in sorted(tables.glob("*.csv"))]
                   + (["raw_all_runs.csv", "reduction_all_runs.csv"] if combined else [])
                   + ([f"{r.folder.relative_to(input_dir).as_posix()}/" for r in runs] if combined else []))

    produced = {"report (md)": report.to_markdown(out_dir / "REPORT.md"),
                "report (html)": report.to_html(out_dir / "report.html")}
    try:
        produced["report (docx)"] = report.to_docx(out_dir / "report.docx")
    except ImportError:
        logger.warning("python-docx is not installed; report.docx skipped (pip install python-docx)")
    (out_dir / "takeaways.md").write_text(
        f"# Takeaways — {name}\n\n{STATUS}\n\n" + "\n".join(f"- {p}" for p in points) + "\n",
        encoding="utf-8")
    produced.update({"takeaways": out_dir / "takeaways.md", "charts": charts, "tables": tables})
    return produced
