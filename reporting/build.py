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

STATUS = ("Not yet reportable research evidence: detector thresholds are not calibrated on a "
          "held-out split, the screening bar (AUROC ≥ 0.65) is a fixed choice, small runs may not "
          "support confidence intervals, and no answers were reviewed by humans "
          "(see docs/REPRODUCIBILITY.md).")
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
    fig.tight_layout(pad=1.2)
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    return path


def heatmap(table: pd.DataFrame, title: str, path: Path, cmap: str, vmin: float, vmax: float,
            fmt: str = "{:.2f}", center: Optional[float] = None,
            scale_columns: bool = False, stars: Optional[pd.DataFrame] = None) -> Optional[Path]:
    """Annotated heatmap of a numeric table (rows × columns). With
    `scale_columns`, colours are scaled within each column (for detectors
    whose score ranges differ); the printed numbers are always the real
    values."""
    if table is None or table.empty:
        return None
    values = table.to_numpy(dtype=float)
    data = values
    if scale_columns:
        span = np.nanmax(np.abs(values), axis=0, initial=0.0)
        data = values / np.where(span > 0, span, 1.0)
    fig, ax = plt.subplots(figsize=(max(6, 0.95 * table.shape[1] + 3),
                                    max(3, 0.5 * table.shape[0] + 1.6)))
    if center is not None:
        from matplotlib.colors import TwoSlopeNorm
        span = max(abs(np.nanmin(data - center)) if np.isfinite(data).any() else 1,
                   abs(np.nanmax(data - center)) if np.isfinite(data).any() else 1, 1e-6)
        norm = TwoSlopeNorm(vcenter=center, vmin=center - span, vmax=center + span)
        image = ax.imshow(data, cmap=cmap, norm=norm, aspect="auto")
    else:
        image = ax.imshow(data, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
    ax.set_xticks(range(table.shape[1]), [str(c) for c in table.columns], rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(table.shape[0]), [str(i) for i in table.index], fontsize=9)
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            if np.isfinite(values[i, j]):
                mark = "✱" if stars is not None and bool(stars.iloc[i, j]) else ""
                ax.text(j, i, fmt.format(values[i, j]) + mark, ha="center", va="center", fontsize=8)
    bar = fig.colorbar(image, ax=ax, shrink=0.8)
    if scale_columns:
        bar.set_label("relative to each column's largest change", fontsize=7)
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


# ── names and explanations used in the report ──────────────────────────

VALIDATED_AUROC = 0.65   # a detector must reach this AUROC in Stage A to judge Stage B

PRETTY = {
    "selfcheckgpt_ngram": "SelfCheckGPT n-gram", "selfcheckgpt_bertscore": "SelfCheckGPT BERTScore",
    "selfcheckgpt_nli": "SelfCheckGPT NLI", "selfcheckgpt_prompt": "SelfCheckGPT LLM-prompt",
    "uqlm_semantic_negentropy": "UQLM semantic entropy", "uqlm_noncontradiction": "UQLM non-contradiction",
    "uqlm_entailment": "UQLM entailment", "uqlm_cosine_sim": "UQLM cosine similarity",
    "uqlm_exact_match": "UQLM exact match", "uqlm_bert_score": "UQLM BERTScore",
    "uqlm_judge": "UQLM LLM judge", "minicheck": "MiniCheck", "summac": "SummaC", "alignscore": "AlignScore",
}
METHOD_PRETTY = {
    "baseline": "Baseline (with context)", "closed_book": "Closed-book (no context)",
    "greedy": "Greedy decoding", "self_refine_adapted": "Self-Refine (adapted)",
    "cove_adapted": "Chain-of-Verification", "uqlm_best_response": "UQLM best answer",
}
TABLE_NOTES = {
    "detector_ranking.csv": "Stage A: one row per detector (AUROC, AUPRC, F1, answers scored, failures)",
    "baseline_vs_methods.csv": "Stage B: matched baseline and method risk from the primary detector",
    "reduction_validated.csv": "Stage B: change vs. baseline per method × validated detector, with 95% CI",
    "summary_by_run.csv": "Stage A metrics of every detector × model in every run",
    "summary_mean_std.csv": "Stage A metrics, mean ± std over runs",
    "per_dataset_by_run.csv": "Stage A metrics per dataset, every run",
    "per_dataset_mean_std.csv": "Stage A metrics per dataset, mean ± std over runs",
    "per_dataset.csv": "Stage A metrics per dataset",
    "reduction_overall_by_run.csv": "Stage B per method × detector (all 13 detectors), every run",
    "reduction_overall_mean_std.csv": "Stage B per method × detector (all detectors), mean ± std",
    "reduction_overall.csv": "Stage B per method × detector (all detectors)",
    "reduction_by_model_by_run.csv": "Stage B per method × detector × model, every run",
    "reduction_by_model_mean_std.csv": "Stage B per method × detector × model, mean ± std",
    "reduction_by_model.csv": "Stage B per method × detector × model",
    "reduction_by_dataset_by_run.csv": "Stage B per method × detector × dataset, every run",
    "reduction_by_dataset_mean_std.csv": "Stage B per method × detector × dataset, mean ± std",
    "reduction_by_dataset.csv": "Stage B per method × detector × dataset",
    "reduction_validated_by_model.csv": "Stage B per method × model, judged by the best validated detector, with 95% CI",
    "reduction_validated_by_dataset.csv": "Stage B per method × dataset, judged by the best validated detector, with 95% CI",
    "reduction_examples.csv": "the answers behind the largest improvements / regressions",
    "failures.csv": "every failure message, counted per run, stage, detector and model",
}


def pretty(detector: str) -> str:
    return PRETTY.get(detector, detector)


def compares_with(detector: str) -> str:
    return "the model's own samples" if family_of(detector) in ("selfcheckgpt", "uqlm") else "the context"


def cut(text, n: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= n else text[: n - 1] + "…"


def ci_text(mean, low, high) -> str:
    if not agg.is_number(low):
        return f"{mean:+.3f}"
    return f"{mean:+.3f} [95% CI {low:+.3f}, {high:+.3f}]"


def pct(value) -> str:
    return f"{value:.0%}" if agg.is_number(value) else "—"


def verdict(low, high) -> str:
    if not agg.is_number(low):
        return "descriptive only"
    if high < 0:
        return "lower risk"
    if low > 0:
        return "higher risk"
    return "no clear change"


# ── charts ──────────────────────────────────────────────────────────────

def method_diagram(path: Path, exploratory: bool = False) -> Path:
    """The experiment in one picture: validate the detectors, then use the
    ones that pass to judge the reduction methods."""
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

    fig, ax = plt.subplots(figsize=(11, 4.6))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 4.6)
    ax.axis("off")

    def box(x, y, w, h, title, body, colour):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05,rounding_size=0.12",
                                    facecolor=colour, edgecolor="#57606a", linewidth=1))
        ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="top", fontsize=10, weight="bold")
        ax.text(x + w / 2, y + h - 0.68, body, ha="center", va="top", fontsize=8, linespacing=1.35)

    def arrow(x1, y1, x2, y2, label=""):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14,
                                     color="#57606a", linewidth=1.2))
        if label:
            ax.text((x1 + x2) / 2, (y1 + y2) / 2 + 0.12, label, ha="center", fontsize=7, color="#57606a")

    ax.text(0.1, 4.45, "Stage A · can the detectors be trusted?", fontsize=10, color="#0969da", weight="bold")
    box(0.1, 2.55, 3.0, 1.65, "Labeled data", "questions with context and\nanswers already known to be\nfaithful or hallucinated\n(HaluEval · RAGTruth · HaluBench)", "#f6f8fa")
    box(4.0, 2.55, 3.0, 1.65, "Every detector scores\nthe known answers", "SelfCheckGPT · UQLM\n(compare with model samples)\nMiniCheck · SummaC · judge\n(compare with the context)", "#ddf4ff")
    box(7.9, 2.55, 3.0, 1.65,
        "Selected detector views" if exploratory else "Screened detectors",
        "Too few questions to validate;\nshow highest observed AUROC\nonly as a descriptive check" if exploratory else
        f"AUROC ≥ {VALIDATED_AUROC}:\nused to judge method answers\nwith stated limitations", "#dafbe1")
    arrow(3.1, 3.37, 4.0, 3.37)
    arrow(7.0, 3.37, 7.9, 3.37, "AUROC")

    ax.text(0.1, 2.2, "Stage B · do the methods reduce hallucination?", fontsize=10, color="#0969da", weight="bold")
    box(0.1, 0.2, 3.0, 1.75, "Each model answers", "baseline (with context), then\nclosed-book · greedy · Self-Refine\nChain-of-Verification\nUQLM best answer", "#f6f8fa")
    box(4.0, 0.2, 3.0, 1.75,
        "Selected detectors\nscore every answer" if exploratory else "Screened detectors\nscore every answer",
        "same questions, same models,\nsame samples: a paired\ncomparison per question", "#ddf4ff")
    box(7.9, 0.2, 3.0, 1.75, "Change vs. baseline",
        "mean change in risk;\nsmall runs are descriptive\n(no uncertainty interval)" if exploratory else
        "mean change in risk with a\n95% confidence interval,\nshare of answers better / worse", "#fff8c5")
    arrow(3.1, 1.07, 4.0, 1.07)
    arrow(7.0, 1.07, 7.9, 1.07)
    arrow(9.4, 2.55, 5.5, 1.95)
    ax.text(8.3, 2.12, "descriptive view" if exploratory else "used as judges", fontsize=7, color="#57606a")
    return _save(fig, path)


def ranking_chart(ranking: pd.DataFrame, path: Path, exploratory: bool = False) -> Optional[Path]:
    if ranking.empty or ranking["AUROC"].isna().all():
        return None
    data = ranking.dropna(subset=["AUROC"]).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(3, 0.38 * len(data) + 1.4)))
    colours = ["#0969da" if c == "the model's own samples" else "#1a7f37" for c in data["compares with"]]
    xerr = None
    if data["AUROC low"].notna().any() and (data["AUROC high"] - data["AUROC low"]).fillna(0).gt(0).any():
        xerr = [(data["AUROC"] - data["AUROC low"]).fillna(0).to_numpy(),
                (data["AUROC high"] - data["AUROC"]).fillna(0).to_numpy()]
    ax.barh(data["detector"], data["AUROC"], color=colours, xerr=xerr, capsize=3)
    ax.axvline(0.5, color="#cf222e", linestyle="--", linewidth=1)
    ax.axvline(VALIDATED_AUROC, color="#1a7f37", linestyle=":", linewidth=1.2)
    ax.text(0.5, len(data) - 0.4, " chance", color="#cf222e", fontsize=8, va="bottom")
    ax.text(VALIDATED_AUROC, len(data) - 0.4, f" screening bar ≥ {VALIDATED_AUROC}",
            color="#1a7f37", fontsize=8, va="bottom")
    for y, v in enumerate(data["AUROC"]):
        ax.text(min(v, 0.97) + 0.01, y, f"{v:.2f}", va="center", fontsize=8)
    ax.set_xlim(0, 1.05)
    ax.set_xlabel("AUROC (1 = always ranks the hallucinated answer higher, 0.5 = coin flip)")
    ax.set_title("Stage A · Observed separation on labeled answers"
                 + (" (small sample)" if exploratory else ""), fontsize=10)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color="#0969da", label="compares with the model's own samples"),
                       Patch(color="#1a7f37", label="compares with the context")],
              loc="lower right", fontsize=8)
    return _save(fig, path)


def effects_chart(effects: pd.DataFrame, validated: List[str], auroc: Dict[str, float], path: Path) -> Optional[Path]:
    if effects.empty:
        return None
    panels = [d for d in validated if d in set(effects["detector"])][:2]
    if not panels:
        return None
    methods = [m for m in METHOD_PRETTY if m in set(effects["condition"])]
    fig, axes = plt.subplots(len(panels), 1, figsize=(9, max(3.4, len(panels) * 2.9)), squeeze=False)
    for ax, det in zip(axes[:, 0], panels):
        e = (effects[effects["detector"] == det].set_index("condition")
             .reindex(methods).dropna(subset=["mean_change"]).iloc[::-1])
        colours = ["#1a7f37" if v < 0 else "#cf222e" for v in e["mean_change"].fillna(0)]
        err = [(e["mean_change"] - e["ci_low"]).fillna(0).clip(lower=0).to_numpy(),
               (e["ci_high"] - e["mean_change"]).fillna(0).clip(lower=0).to_numpy()]
        ax.barh([METHOD_PRETTY[m] for m in e.index], e["mean_change"], color=colours, xerr=err, capsize=3)
        ax.axvline(0, color="#24292f", linewidth=1)
        ax.set_title(f"{pretty(det)} (AUROC {auroc.get(det, float('nan')):.2f})", fontsize=9)
        ax.set_xlabel("Mean risk change: left is lower risk; right is higher risk", fontsize=8)
        ax.tick_params(labelsize=8)
    fig.suptitle("Stage B · Paired change vs. the baseline answer (whiskers when CI is available)", fontsize=10)
    return _save(fig, path)


def paired_risk_table(reduction: pd.DataFrame, detector: str) -> pd.DataFrame:
    """Absolute baseline and method risk on the same answered pairs."""
    score = f"{detector}_score"
    keys = ["run", "sample_id", "model"]
    if reduction.empty or not set([*keys, "condition", score]).issubset(reduction.columns):
        return pd.DataFrame()
    base = reduction.loc[reduction["condition"] == "baseline", keys + [score]].rename(
        columns={score: "baseline_risk"})
    methods = reduction.loc[reduction["condition"] != "baseline", keys + ["condition", score]].rename(
        columns={score: "method_risk"})
    paired = methods.merge(base, on=keys, how="inner", validate="many_to_one").dropna(
        subset=["baseline_risk", "method_risk"])
    if paired.empty:
        return pd.DataFrame()
    return (paired.groupby("condition", sort=False)
            .agg(baseline_risk=("baseline_risk", "mean"), method_risk=("method_risk", "mean"),
                 pairs=("method_risk", "count"), questions=("sample_id", "nunique"))
            .reset_index())


def absolute_risk_chart(paired: pd.DataFrame, detector: str, path: Path) -> Optional[Path]:
    """The direct baseline-versus-method view shown in the older report."""
    if paired.empty:
        return None
    data = paired.set_index("condition").reindex(
        [m for m in METHOD_PRETTY if m in set(paired["condition"])]).iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(3.4, 0.65 * len(data) + 1.4)))
    y = np.arange(len(data))
    baseline_bars = ax.barh(y - 0.19, data["baseline_risk"], height=0.36,
                            color="#8c959f", label="Baseline")
    method_bars = ax.barh(y + 0.19, data["method_risk"], height=0.36,
                          color="#0969da", label="With method")
    ax.bar_label(baseline_bars, fmt="%.2f", padding=2, fontsize=8)
    ax.bar_label(method_bars, fmt="%.2f", padding=2, fontsize=8)
    ax.set_yticks(y, [METHOD_PRETTY.get(m, m) for m in data.index])
    ax.set_xlim(0, max(data[["baseline_risk", "method_risk"]].max().max() * 1.18, 0.1))
    ax.set_xlabel(f"Mean {pretty(detector)} risk on matched answers (lower = less hallucinated)")
    ax.set_title("Baseline vs. each method · same question and generator model", fontsize=11)
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="x", alpha=0.2)
    return _save(fig, path)


def model_effect_chart(frame: pd.DataFrame, model: str, path: Path) -> Optional[Path]:
    data = frame[frame["model"] == model]
    if data.empty:
        return None
    data = (data.set_index("condition").reindex(
        [m for m in METHOD_PRETTY if m in set(data["condition"])]).dropna(subset=["mean_change"]).iloc[::-1])
    if data.empty:
        return None
    fig, ax = plt.subplots(figsize=(9, max(3.2, 0.55 * len(data) + 1.3)))
    values = data["mean_change"].fillna(0)
    bars = ax.barh([METHOD_PRETTY.get(m, m) for m in data.index], values,
                   color=["#1a7f37" if v < 0 else "#cf222e" for v in values])
    ax.bar_label(bars, fmt="%+.2f", padding=3, fontsize=8)
    ax.axvline(0, color="#24292f", linewidth=1)
    span = max(values.abs().max(), 0.05) * 1.3
    ax.set_xlim(-span, span)
    ax.set_xlabel("Mean risk change vs. baseline (left = lower risk; right = higher risk)")
    ax.set_title(f"{model} · judged by {pretty(data['detector'].iloc[0])}", fontsize=11)
    ax.grid(axis="x", alpha=0.2)
    return _save(fig, path)


# ── the report ──────────────────────────────────────────────────────────

def detector_ranking(summary_ms: pd.DataFrame) -> pd.DataFrame:
    """One row per detector, best first (mean over models where it depends
    on the model; low/high = the range over models)."""
    if summary_ms.empty:
        return pd.DataFrame()
    rows = []
    for det, g in summary_ms.groupby("detector"):
        rows.append({
            "detector": pretty(det), "key": det, "compares with": compares_with(det),
            "AUROC": g["roc_auc_mean"].mean(), "AUROC low": g["roc_auc_mean"].min(),
            "AUROC high": g["roc_auc_mean"].max(), "run-to-run std": g["roc_auc_std"].mean(),
            "AUPRC": g["average_precision_mean"].mean(), "F1": g["f1_mean"].mean(),
            "answers scored": g["n_cases_mean"].sum(), "failed": g["n_failed_mean"].sum(),
        })
    out = pd.DataFrame(rows).sort_values("AUROC", ascending=False, na_position="last").reset_index(drop=True)
    out.insert(0, "rank", range(1, len(out) + 1))
    return out


def generate(input_dir: Path, out_dir: Optional[Path] = None, title: Optional[str] = None) -> Dict[str, Path]:
    input_dir = Path(input_dir)
    out_dir = Path(out_dir or input_dir)
    charts = out_dir / "charts"
    tables = out_dir / "tables"
    charts.mkdir(parents=True, exist_ok=True)
    tables.mkdir(parents=True, exist_ok=True)
    combined = out_dir != input_dir

    # ── data ────────────────────────────────────────────────────────────
    runs = agg.discover(input_dir)
    n_runs = len({r.run for r in runs})
    by_run = agg.summary_by_run(runs)
    for metric in agg.METRICS:           # absent when every detector failed
        if metric not in by_run.columns:
            by_run[metric] = np.nan
    metric_cols = list(agg.METRICS)
    summary_ms = agg.mean_std(by_run, ["detector", "model"], metric_cols + ["n_cases", "n_failed"])
    per_model = set(by_run.loc[by_run["model"] != agg.MODEL_INDEPENDENT, "detector"])
    dataset_by_run = agg.per_dataset_by_run(runs, agg.thresholds(by_run), per_model)
    dataset_ms = agg.mean_std(dataset_by_run, ["dataset", "detector", "model"], metric_cols + ["n_cases", "n_failed"]) \
        if not dataset_by_run.empty else pd.DataFrame()
    raw_all = agg.concat([r.raw for r in runs])
    no_signal = agg.constant_detectors(raw_all)      # same score for every answer
    ranking = detector_ranking(summary_ms)
    auroc = dict(zip(ranking.get("key", []), ranking.get("AUROC", [])))
    n_questions = raw_all["sample_id"].nunique() if "sample_id" in raw_all else 0
    evidence_limited = n_questions < agg.MIN_QUESTIONS_FOR_CI
    validated = [k for k, v in auroc.items()
                 if agg.is_number(v) and v >= VALIDATED_AUROC and k not in no_signal]
    fallback = evidence_limited or not validated
    if fallback:  # no trustworthy gate: use three for an explicitly descriptive check
        validated = [k for k in ranking.dropna(subset=["AUROC"])["key"] if k not in no_signal][:3] \
            if not ranking.empty else []

    reduction = agg.reduction_table(runs)
    deltas = agg.reduction_deltas(reduction)
    units = agg.per_unit(deltas)
    effects = agg.method_effects(units, validated) if not units.empty else pd.DataFrame()
    red = {}
    for name, keys in (("overall", ["condition", "detector"]),
                       ("by_model", ["condition", "detector", "model"]),
                       ("by_dataset", ["condition", "detector", "dataset"])):
        per_run = agg.reduction_by_run(deltas, keys)
        red[name] = (per_run, agg.mean_std(per_run, keys, ["n_pairs", "mean_delta", "better", "worse"])
                     if not per_run.empty else pd.DataFrame())
    primary = validated[0] if validated else None
    by_model = agg.method_effects(units, [primary], by="model") if primary and not units.empty else pd.DataFrame()
    by_dataset = agg.method_effects(units, [primary], by="dataset") if primary and not units.empty else pd.DataFrame()
    examples = agg.reduction_examples(reduction, deltas, primary, n=1) if primary else pd.DataFrame()
    fails = agg.failures(runs)

    # ── tables (CSV) ────────────────────────────────────────────────────
    pairs = {"summary": (by_run, summary_ms), "per_dataset": (dataset_by_run, dataset_ms),
             "reduction_overall": red["overall"], "reduction_by_model": red["by_model"],
             "reduction_by_dataset": red["by_dataset"]}
    for name, (per_run, ms) in pairs.items():
        if combined:
            for suffix, frame in (("by_run", per_run), ("mean_std", ms)):
                if frame is not None and not frame.empty:
                    frame.to_csv(tables / f"{name}_{suffix}.csv", index=False)
        elif per_run is not None and not per_run.empty and name != "summary":
            per_run.drop(columns=["run"], errors="ignore").to_csv(tables / f"{name}.csv", index=False)
    singles = {"detector_ranking": ranking.drop(columns=["key"], errors="ignore"),
               "reduction_validated": effects, "reduction_validated_by_model": by_model,
               "reduction_validated_by_dataset": by_dataset, "reduction_examples": examples, "failures": fails}
    for name, frame in singles.items():
        if frame is not None and not frame.empty:
            frame.to_csv(tables / f"{name}.csv", index=False)
    if combined:
        agg.concat([r.raw for r in runs]).to_csv(out_dir / "raw_all_runs.csv", index=False)
        if not reduction.empty:
            reduction.to_csv(out_dir / "reduction_all_runs.csv", index=False)

    # ── charts ──────────────────────────────────────────────────────────
    for old in charts.glob("*.png"):
        old.unlink()
    fig_method = method_diagram(charts / "how_the_experiment_works.png", exploratory=evidence_limited)
    fig_rank = ranking_chart(ranking, charts / "detector_ranking.png", exploratory=evidence_limited)
    dataset_figures = []
    if not dataset_ms.empty:
        order = [k for k in ranking["key"] if k in set(dataset_ms["detector"])][:7]
        ds = (dataset_ms.groupby(["detector", "dataset"])["roc_auc_mean"].mean().unstack("dataset")
              .reindex(order).rename(index=pretty))
        for index, start in enumerate(range(0, len(ds.columns), 5), 1):
            figure = heatmap(ds.iloc[:, start:start + 5],
                             "Stage A · AUROC by dataset (0.5 = chance; higher = better)",
                             charts / f"detector_by_dataset_{index:02d}.png", "viridis", 0.3, 1.0)
            if figure:
                dataset_figures.append(figure)
    fig_effects = effects_chart(effects, validated, auroc, charts / "reduction_effects.png")
    paired = paired_risk_table(reduction, primary) if primary else pd.DataFrame()
    fig_absolute = absolute_risk_chart(paired, primary, charts / "baseline_vs_methods.png") if primary else None
    model_figures = {}
    if not by_model.empty:
        for index, model in enumerate(sorted(by_model["model"].dropna().unique())):
            figure = model_effect_chart(
                by_model, model, charts / f"reduction_model_{index + 1:02d}.png")
            if figure:
                model_figures[model] = figure
    if not paired.empty:
        paired.to_csv(tables / "baseline_vs_methods.csv", index=False)

    def method_grid(frame, col, title, path):
        if frame.empty:
            return None
        order = [m for m in METHOD_PRETTY if m in set(frame["condition"])]
        values = frame.pivot_table(index="condition", columns=col, values="mean_change", sort=False).reindex(order)
        clear = frame.assign(sig=(frame["ci_high"] < 0) | (frame["ci_low"] > 0)).pivot_table(
            index="condition", columns=col, values="sig", aggfunc="max", sort=False).reindex(order)
        return heatmap(values.rename(index=METHOD_PRETTY), title, path, "RdYlGn_r", 0, 0, "{:+.2f}",
                       center=0.0, stars=clear.reindex_like(values).fillna(False).astype(bool))
    fig_b_model = method_grid(by_model, "model", f"Stage B by generator model ({pretty(primary)}; ✱ = 95% CI excludes 0)",
                              charts / "reduction_by_model.png") if primary else None
    dataset_effect_figures = []
    if primary and not by_dataset.empty and by_dataset["dataset"].nunique() > 1:
        all_datasets = sorted(by_dataset["dataset"].dropna().unique())
        for index, start in enumerate(range(0, len(all_datasets), 5), 1):
            subset = by_dataset[by_dataset["dataset"].isin(all_datasets[start:start + 5])]
            figure = method_grid(subset, "dataset",
                                 f"Stage B by dataset ({pretty(primary)}; ✱ = 95% CI excludes 0)",
                                 charts / f"reduction_by_dataset_{index:02d}.png")
            if figure:
                dataset_effect_figures.append(figure)
    fig_a_model = None
    gen_rows = summary_ms[summary_ms["model"] != agg.MODEL_INDEPENDENT]
    if gen_rows["model"].nunique() > 1:
        order = [k for k in ranking["key"] if k in set(gen_rows["detector"])]
        grid = gen_rows.pivot_table(index="detector", columns="model", values="roc_auc_mean").reindex(order)
        fig_a_model = heatmap(grid.rename(index=pretty), "Stage A · AUROC of the sampling-based detectors "
                              "with each generator model's samples", charts / "detector_by_model.png",
                              "viridis", 0.3, 1.0)
    all_detector_figures = []
    if not red["overall"][1].empty:
        allt = red["overall"][1].pivot_table(index="condition", columns="detector",
                                              values="mean_delta_mean", sort=False)
        allt = allt.reindex(columns=[k for k in ranking["key"] if k in allt.columns]).rename(
            index=METHOD_PRETTY, columns=pretty)
        for index, start in enumerate(range(0, len(allt.columns), 5), 1):
            figure = heatmap(allt.iloc[:, start:start + 5],
                             "Appendix · paired risk change (colour scaled within each detector)",
                             charts / f"reduction_all_detectors_{index:02d}.png", "RdYlGn_r", 0, 0,
                             "{:+.2f}", center=0.0, scale_columns=True)
            if figure:
                all_detector_figures.append(figure)
    fig_consistency = consistency_chart(by_run.assign(detector=by_run["detector"].map(pretty)),
                                        charts / "run_consistency.png")

    # ── numbers used in the text ────────────────────────────────────────
    manifests = [r.manifest for r in runs if r.manifest]
    models = sorted({g["name"] for m in manifests for g in m.get("models", [])})
    datasets = {d["name"]: d["n_samples"] for m in manifests for d in m.get("datasets", [])}
    if not models:
        source = reduction if not reduction.empty and "model" in reduction else by_run
        models = sorted(str(m) for m in source.get("model", pd.Series(dtype=str)).dropna().unique()
                        if m != agg.MODEL_INDEPENDENT)
    if not datasets and {"dataset", "sample_id"}.issubset(raw_all.columns):
        datasets = raw_all.groupby("dataset")["sample_id"].nunique().to_dict()
    n_answers = raw_all["case_id"].nunique() if not raw_all.empty else 0
    n_halluc = int(raw_all.drop_duplicates("case_id")["label"].sum()) if not raw_all.empty else 0
    near_chance = [pretty(k) for k, v in auroc.items()
                   if agg.is_number(v) and abs(v - 0.5) < 0.1 and k not in no_signal]
    judge_model = next((r.config.get("judge", {}).get("model") for r in runs if r.config), None)
    weak_llm = [pretty(k) for k in ("uqlm_judge", "selfcheckgpt_prompt")
                if agg.is_number(auroc.get(k)) and auroc[k] < 0.6]
    inverted = [pretty(k) for k, v in auroc.items() if agg.is_number(v) and v <= 0.4]

    summary_points = []
    scope = (f"{len(models)} generator model{'s' if len(models) != 1 else ''} "
             f"({', '.join(models) or '—'}) answered {sum(datasets.values())} questions from "
             f"{len(datasets)} dataset{'s' if len(datasets) != 1 else ''}; the detectors were checked on "
             f"{n_answers} answers with known labels ({n_halluc} hallucinated); "
             f"{n_runs} independent run{'s' if n_runs != 1 else ''}.")
    if evidence_limited:
        summary_points.append(
            f"Engineering smoke data: only {n_questions} distinct "
            f"question{'s' if n_questions != 1 else ''}. AUROC rankings and method "
            "changes below are descriptive; this run cannot validate detectors or establish that a method works.")
    if not ranking.empty and ranking["AUROC"].notna().any():
        top = ranking.iloc[0]
        summary_points.append(
            f"Highest observed detector AUROC: {top['detector']} ({top['AUROC']:.2f}; higher AUROC is better "
            "at separating the labeled answers).")
        if not evidence_limited:
            summary_points.append(
                (f"Passed the AUROC ≥ {VALIDATED_AUROC} screening bar: "
                 f"{', '.join(pretty(v) for v in validated)}. " if not fallback else
                 f"No detector reached AUROC {VALIDATED_AUROC}; Stage B uses the three best "
                 f"({', '.join(pretty(v) for v in validated)}) as an exploratory view. ")
                + (f"Close to chance: {', '.join(near_chance)}." if near_chance else ""))
        if inverted:
            summary_points.append(f"Pointing the wrong way (AUROC ≤ 0.4): {', '.join(inverted)}.")
        if no_signal:
            summary_points.append(f"No signal at all (the same score for every answer): "
                                  f"{', '.join(pretty(k) for k in no_signal)}.")
    if not effects.empty and primary:
        candidates = effects[(effects["detector"] == primary) & (effects["condition"] != "closed_book")]
        if not candidates.empty:
            best = candidates.sort_values("mean_change").iloc[0]
            if best["mean_change"] < 0:
                summary_points.append(
                    f"Largest observed risk decrease among the reduction methods: "
                    f"{METHOD_PRETTY.get(best['condition'], best['condition'])} "
                    f"({best['mean_change']:+.3f} vs. baseline according to {pretty(primary)}; "
                    + ("descriptive only, too few questions)." if not agg.is_number(best["ci_low"]) else
                       f"95% CI {best['ci_low']:+.3f} to {best['ci_high']:+.3f})."))
            else:
                summary_points.append(
                    f"No tested reduction method lowered mean {pretty(primary)} risk on these pairs; "
                    "the full paired results appear in Stage B.")
        for cond in [m for m in METHOD_PRETTY if m in set(effects["condition"])]:
            e = effects[(effects["condition"] == cond) & (effects["detector"] == primary)]
            if e.empty:
                continue
            e = e.iloc[0]
            others = effects[(effects["condition"] == cond) & (effects["detector"] != primary)]
            if not agg.is_number(e["ci_low"]):
                summary_points.append(
                    f"{METHOD_PRETTY[cond]}: observed change {e['mean_change']:+.3f} in {pretty(primary)} "
                    f"risk; lower on {e['better']:.0%} of question × model pairs. Too few distinct "
                    "questions for an uncertainty interval or a research conclusion.")
            else:
                agree = [verdict(r.ci_low, r.ci_high) for r in others.itertuples()
                         if agg.is_number(r.ci_low)]
                same = sum(v == verdict(e["ci_low"], e["ci_high"]) for v in agree)
                summary_points.append(
                    f"{METHOD_PRETTY[cond]}: {verdict(e['ci_low'], e['ci_high'])} according to {pretty(primary)} "
                    f"({ci_text(e['mean_change'], e['ci_low'], e['ci_high'])}; better on {e['better']:.0%} of "
                    f"pairs, worse on {e['worse']:.0%})"
                    + (f"; {same} of {len(agree)} other detector(s) agree." if agree else "."))
    total_failed = int(fails["count"].sum()) if not fails.empty else 0
    summary_points.append("Nothing failed." if total_failed == 0 else
                          f"{total_failed} scoring failure(s), excluded from the numbers (see Reliability).")

    # ── report ──────────────────────────────────────────────────────────
    name = title or (f"{input_dir.name} · combined over {n_runs} run(s)" if combined else input_dir.name)
    report = Report(f"Hallucination benchmark — {name}",
                    f"Generated {datetime.now():%Y-%m-%d %H:%M} from this run's own data files.")

    report.h1("Summary")
    report.p(scope)
    report.note("Reading the numbers.",
                "Stage A AUROC: higher is better for the detector (0.5 = chance, 1.0 = perfect). "
                "Stage B risk: lower is better for the answer. Change = method minus baseline, so a "
                "negative change means lower measured hallucination risk. "
                "Each detector has its own score scale; compare a method with its baseline using the same detector.")
    report.bullets(summary_points)
    report.note("Research status.",
                (f"This run has only {n_questions} distinct questions and is an engineering check, not a "
                 "research conclusion. " if evidence_limited else "") + STATUS)

    report.h1("How the experiment works")
    report.p("The study asks two questions in order. First (Stage A): which hallucination detectors can be "
             "trusted? Every detector scores answers whose label is already known, and we measure how well it "
             "separates hallucinated from faithful ones. Second (Stage B): does a reduction method make a model "
             "hallucinate less? Each model answers the same questions with and without each method. "
             + ("This small run shows selected detector scores descriptively; it cannot validate them."
                if evidence_limited else "Detectors that passed the Stage A screen judge the answers."))
    report.figure(fig_method, "The two stages of the experiment.")
    report.h2("Data")
    report.bullets([f"{d}: {n} questions" for d, n in datasets.items()] or ["(no manifest found)"])
    report.h2("Detectors (official code for every one)")
    sample_based = [pretty(k) for k in auroc if compares_with(k) == "the model's own samples"]
    context_based = [pretty(k) for k in auroc if compares_with(k) == "the context"]
    report.bullets([
        "Compare an answer with the model's own samples (a model that knows the answer says the same thing "
        f"again; a hallucinated detail changes between samples): {', '.join(sample_based) or '—'}.",
        "Compare an answer with the context (trained fact-checking models, or a separate LLM grading the "
        f"answer): {', '.join(context_based) or '—'}."
        + (f" LLM judge model: {judge_model}." if judge_model and any("judge" in c or "prompt" in c
                                                                       for c in auroc) else ""),
        "Every score is a risk: higher means more likely hallucinated.",
    ])
    report.h2("Reduction methods")
    if reduction.empty:
        report.p("No reduction method answers were recorded in this result folder.")
    else:
        report.bullets([f"{METHOD_PRETTY[m]}: {note}" for m, note in METHOD_NOTES.items()
                        if m in set(reduction["condition"])])

    report.h1("Stage A · Which detectors separate the labeled answers?")
    report.figure(fig_rank, "Detector AUROC on labeled answers. Higher is better for a detector.")
    report.note("How to read this chart.",
                "Each bar is one detector. AUROC is the chance that the detector gives a hallucinated answer a "
                "higher risk than a faithful one: 1.0 is perfect, 0.5 (red dashed line) is a coin flip. "
                f"The green dotted line (≥ {VALIDATED_AUROC}) is the planned screening bar. "
                "Error bars, when shown, span generator models; they are not confidence intervals. "
                + ("With so few questions, crossing the line does not validate a detector."
                   if evidence_limited else "Scores past the line are used for Stage B."))
    if not ranking.empty and ranking["AUROC"].notna().any():
        top = ranking.iloc[0]
        report.p(f"What it shows: {top['detector']} has the highest observed AUROC ({top['AUROC']:.2f}). "
                 + (f"Only {n_questions} questions were tested, so this ordering is unstable."
                    if evidence_limited else
                    f"{len(validated) if not fallback else 0} of {len(ranking)} detector scores reach the "
                    "screening bar."))
    shown = ranking.drop(columns=["key", "AUROC low", "AUROC high"], errors="ignore").copy()
    shown = shown.drop(columns=["compares with", "run-to-run std"], errors="ignore")
    for col in ("answers scored", "failed"):
        if col in shown:
            shown[col] = shown[col].round().astype("Int64")
    if n_runs < 2:
        shown = shown.drop(columns=["run-to-run std"], errors="ignore")
    report.table(shown, "All detectors (AUPRC: like AUROC but focused on finding the hallucinated answers; "
                        "F1 uses an uncalibrated threshold):")
    if fig_a_model:
        report.h2("By generator model")
        report.figure(fig_a_model, "AUROC of each sampling-based detector with each model's samples.")
        report.note("How to read this chart.",
                    "SelfCheckGPT and UQLM judge an answer by comparing it with samples from a generator model, so "
                    "their accuracy can depend on which model produced the samples. Each column is one model.")
        per_model_auc = gen_rows.groupby("model")["roc_auc_mean"].mean().sort_values()
        report.p(f"What it shows: averaged over these detectors, samples from {per_model_auc.index[-1]} make "
                 f"detection easiest (AUROC {per_model_auc.iloc[-1]:.2f}) and samples from "
                 f"{per_model_auc.index[0]} hardest ({per_model_auc.iloc[0]:.2f}).")
    if dataset_figures:
        report.h2("By dataset")
        for figure in dataset_figures:
            report.figure(figure, "Observed AUROC of leading detector scores by dataset; higher is better.")
        report.note("How to read this chart.",
                    "Rows are detectors (best first), columns are datasets; brighter = better separation. A "
                    "detector that is bright on some datasets and dark on others only works for some kinds of "
                    "questions.")
        by_ds = (dataset_ms.groupby(["dataset", "detector"])["roc_auc_mean"].mean()
                 .groupby("dataset").mean().sort_values())
        if len(by_ds) > 1:
            report.p(f"What it shows: averaged over detectors, detection is hardest on {by_ds.index[0]} "
                     f"(AUROC {by_ds.iloc[0]:.2f}) and easiest on {by_ds.index[-1]} ({by_ds.iloc[-1]:.2f}).")

    if not effects.empty:
        report.h1("Stage B · Do the reduction methods reduce hallucination?")
        report.p("Each method is paired with the baseline answer from the same question, generator model, "
                 "and run. " +
                 (f"Only {n_questions} distinct questions were used, so the three highest observed detectors "
                  "show descriptive comparisons; they are not validated judges. " if evidence_limited else
                  "No detector passed the screening bar, so the three highest observed detectors show "
                  "exploratory comparisons. " if fallback else
                  "Only detectors passing Stage A are used for the main comparison. ")
                 + "Repeated runs of a question and model are averaged first; uncertainty resamples "
                 "distinct questions, keeping their model results together.")
        if fig_absolute:
            report.h2("Baseline versus each method")
            report.figure(fig_absolute, f"Mean risk from {pretty(primary)} on matched baseline and method answers.")
            report.note("How to read these bars.",
                        "Each grey/blue pair uses the same questions, models, and runs. Lower blue than grey "
                        "means the method produced answers this detector judged less hallucinated. Baseline "
                        "bars can differ slightly when a method has missing scores; only complete pairs count. "
                        "Absolute scores from different detectors must not be compared with each other.")
            direct = paired.assign(method=paired["condition"].map(METHOD_PRETTY),
                                   change=paired["method_risk"] - paired["baseline_risk"])
            report.table(direct[["method", "baseline_risk", "method_risk", "change", "pairs", "questions"]]
                         .rename(columns={"baseline_risk": "baseline risk", "method_risk": "method risk"}),
                         f"Matched comparison according to {pretty(primary)} (negative change = lower risk):")
        report.h2("Paired change across detectors")
        report.figure(fig_effects, "Mean change vs. baseline for the first two selected detector scores.")
        report.note("How to read this chart.",
                    "Each bar is the average change in risk: left of zero (green) means lower measured risk, "
                    "right of zero (red) means higher. Whiskers show a 95% confidence interval only when "
                    f"at least {agg.MIN_PAIRS_FOR_CI} question × model pairs and "
                    f"{agg.MIN_QUESTIONS_FOR_CI} distinct questions are available. Without whiskers, "
                    "direction is descriptive, not evidence of a reliable effect.")
        tbl = effects.assign(method=effects["condition"].map(METHOD_PRETTY),
                             detector=effects["detector"].map(pretty),
                             change=[ci_text(m, l, h) for m, l, h in zip(effects["mean_change"], effects["ci_low"], effects["ci_high"])],
                             result=[verdict(l, h) for l, h in zip(effects["ci_low"], effects["ci_high"])])
        tbl["better"], tbl["worse"] = tbl["better"].map(pct), tbl["worse"].map(pct)
        primary_tbl = tbl[tbl["detector"] == pretty(primary)]
        report.table(primary_tbl[["method", "change", "result", "better", "worse", "n_pairs", "n_questions"]]
                     .rename(columns={"n_pairs": "pairs", "n_questions": "questions"}),
                     f"Paired change according to {pretty(primary)}; better/worse is the share of "
                     "question × model pairs:")
        if tbl["detector"].nunique() > 1:
            report.p("The other selected detectors can disagree; their complete per-method results, "
                     "pair counts, and intervals are in tables/reduction_validated.csv. "
                     "Agreement across detectors strengthens a finding; disagreement needs investigation.")
        if "closed_book" in set(effects["condition"]):
            e = effects[(effects["condition"] == "closed_book") & (effects["detector"] == primary)]
            if not e.empty:
                report.p("Closed-book is a check of the setup, not a candidate method: removing context is "
                         f"expected to increase risk. Here it changed by "
                         f"{ci_text(*e[['mean_change', 'ci_low', 'ci_high']].iloc[0])} according to {pretty(primary)}.")
        if model_figures:
            report.h2("Results for each generator model")
            report.p("Each chart compares the reduction methods for one generator model. "
                     "Bars left of zero mean lower measured risk than that model's baseline; right means higher. "
                     "These are descriptive when no interval is available.")
            for model, figure in model_figures.items():
                report.h3(str(model))
                report.figure(figure, f"Paired risk change for {model}, judged by {pretty(primary)}.")
        if dataset_effect_figures:
            frame, col, label = by_dataset, "dataset", "dataset"
            report.h2("By dataset")
            for figure in dataset_effect_figures:
                report.figure(figure, f"Change in risk per method and dataset, judged by {pretty(primary)}.")
            report.note("How to read this chart.",
                        f"Each cell is a method's average change in risk vs. the baseline for one {label} (green = "
                        "lower risk). ✱ marks cells whose 95% confidence interval does not include zero; cells "
                        f"without ✱ are not distinguishable from the baseline. Cells with fewer than "
                        f"{agg.MIN_PAIRS_FOR_CI} pairs or {agg.MIN_QUESTIONS_FOR_CI} distinct questions "
                        "get no interval (too little data).")
            clear = frame[(frame["ci_high"] < 0) | (frame["ci_low"] > 0)]
            better = clear[clear["mean_change"] < 0]
            worse = clear[clear["mean_change"] > 0]
            parts = []
            if not better.empty:
                parts.append("clearly lower risk: " + "; ".join(
                    f"{METHOD_PRETTY.get(r.condition, r.condition)} on {getattr(r, col)}" for r in better.itertuples()))
            if not worse.empty:
                parts.append("clearly higher risk: " + "; ".join(
                    f"{METHOD_PRETTY.get(r.condition, r.condition)} on {getattr(r, col)}" for r in worse.itertuples()))
            if frame["ci_low"].isna().all():
                report.p(f"What it shows: these {label} comparisons have fewer than "
                         f"{agg.MIN_PAIRS_FOR_CI} pairs or {agg.MIN_QUESTIONS_FOR_CI} distinct questions, "
                         "so they are descriptive only.")
            else:
                report.p("What it shows: " + (". ".join(parts) + "." if parts else
                                              f"no method is clearly different from the baseline on any single {label}."))
        if not examples.empty:
            report.h2("Examples")
            report.p(f"For each method, the question where {pretty(primary)} saw the largest improvement:")
            change_col = f"Δ {primary}"
            for _, r in examples[examples["kind"] == "improved most"].iterrows():
                method = METHOD_PRETTY.get(r["method"], r["method"])
                source = raw_all if "question" in raw_all.columns else reduction
                question = source.loc[source["sample_id"] == r["sample_id"], "question"].dropna() \
                    if "question" in source.columns else pd.Series(dtype=str)
                report.bullets([
                    f"{method} · {r['model']} · {r['sample_id']} · change in risk {r[change_col]:+.2f}",
                    f"Question: {cut(question.iloc[0], 200) if not question.empty else '—'}",
                    f"Baseline answer: {cut(r['baseline answer'], 240)}",
                    f"Answer with this method: {cut(r['method answer'], 240)}",
                ])
    else:
        report.h1("Stage B · Reduction methods")
        report.p("No reduction answers were recorded for this run."
                 if reduction.empty else
                 "No matched method-versus-baseline detector scores were available for a valid comparison. "
                 "Check the reduction and scoring failures in the Reliability section and the raw CSV files.")

    report.h1("Reliability")
    if n_runs > 1 and not ranking.empty and ranking["run-to-run std"].notna().any():
        worst = ranking.sort_values("run-to-run std", ascending=False).iloc[0]
        report.p(f"Across {n_runs} runs (same questions, fresh generator samples), the largest AUROC variation was "
                 f"±{worst['run-to-run std']:.3f} ({worst['detector']}). Scores from MiniCheck, SummaC, "
                 "AlignScore and the judge are reused across runs because these checks are treated as "
                 "generator-independent; their reported run-to-run variation is therefore zero by design.")
    else:
        report.p("Run-to-run AUROC variation could not be estimated from the available scores."
                 if n_runs > 1 else
                 "Only one run: run-to-run variation is not measured here (see the combined report).")
    if weak_llm:
        report.p(f"The LLM-based detectors ({', '.join(weak_llm)}) were close to chance on these labels. "
                 f"The judge model was {judge_model}; its quality, the prompts, and the dataset may all "
                 "affect this result. Human review or a separate judge comparison is needed before "
                 "attributing the cause.")
    if no_signal:
        report.p(f"{', '.join(pretty(k) for k in no_signal)} gave the same score to every answer, so it carries "
                 "no information on this data and is not used to judge anything.")
    if fails.empty:
        report.p("No scoring failures.")
    else:
        f = (fails.groupby(["stage", "detector"]).agg(failures=("count", "sum"), reason=("error", "first"))
             .reset_index())
        f["reason"] = f["reason"].map(lambda t: cut(t, 110))
        report.table(f, "Failures are left out of every number above, never scored as 0. Known causes: "
                        "SelfCheckGPT BERTScore cannot score very short answers (upstream limitation); an LLM judge "
                        "sometimes replies in a form its parser cannot read.")

    report.h1("Appendix")
    report.h2("What exactly was run")
    report.bullets(run_plan(runs, by_run, reduction))
    if fig_b_model:
        report.h2("Method changes across generator models")
        report.figure(fig_b_model, f"Mean paired risk change by method and model, judged by {pretty(primary)}.")
    if all_detector_figures:
        report.h2("Stage B as seen by every detector")
        for figure in all_detector_figures:
            report.figure(figure, "Mean change vs. baseline for every detector, including those that did not "
                                  "pass Stage A. Values are real scores; colours are scaled within each detector.")
    if fig_consistency:
        report.h2("AUROC in every run")
        report.figure(fig_consistency, "AUROC of each detector in each run: flat lines = stable.")
    report.h2("Data files")
    report.p("Every number in this report comes from these files; the full detail (every answer, every score) "
             "is in them rather than in this document.")
    data_notes = {
        "detector_validation_raw.csv": "every labeled answer with this group's detector scores",
        "detector_validation_summary.csv": "this group's Stage A metrics",
        "reduction_comparison.csv": "every method's answer with every core detector's score",
        "reduction_scores.csv": "this group's scores of the reduction answers",
        "selfcheckgpt_samples.jsonl": "the samples the sampling-based detectors compared against",
        "run_manifest.json": "what exactly ran (versions, commit, checksums, model digests)",
    }
    if combined:
        files = ["raw_all_runs.csv: every labeled answer of every run with every score",
                 "reduction_all_runs.csv: every method's answer of every run with every score",
                 "run_XX/: each run's own report and data (see its report's Appendix)"]
    else:
        files = [f"{(r.folder / name).relative_to(input_dir).as_posix()}: {note}"
                 for r in runs for name, note in data_notes.items() if (r.folder / name).is_file()]
    files += [f"tables/{p.name}: {TABLE_NOTES.get(p.name, '')}" for p in sorted(tables.glob("*.csv"))]
    report.bullets(files)

    produced = {"report (md)": report.to_markdown(out_dir / "REPORT.md"),
                "report (html)": report.to_html(out_dir / "report.html")}
    try:
        produced["report (docx)"] = report.to_docx(out_dir / "report.docx")
    except ImportError:
        logger.warning("python-docx is not installed; report.docx skipped (pip install python-docx)")
    (out_dir / "takeaways.md").write_text(
        f"# Takeaways — {name}\n\n{scope}\n\n" + "\n".join(f"- {p}" for p in summary_points)
        + f"\n\n{STATUS}\n", encoding="utf-8")
    produced.update({"takeaways": out_dir / "takeaways.md", "charts": charts, "tables": tables})
    return produced
