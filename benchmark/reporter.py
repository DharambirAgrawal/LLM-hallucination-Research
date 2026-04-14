"""
Reporter
========
Generates:
  - Rich console summary table
  - Matplotlib / Seaborn charts (bar plots, heatmaps, ROC curves)
  - Standalone HTML report
  - JSON summary
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd
from loguru import logger
from rich.console import Console
from rich.table import Table
from rich import box


console = Console()

METHOD_ORDER = [
    "token_similarity",
    "semantic_similarity",
    "llm_based",
    "bert_stochastic",
    "ensemble",
]

METHOD_LABELS = {
    "token_similarity":    "Token Similarity",
    "semantic_similarity": "Semantic Similarity",
    "llm_based":           "LLM Prompt-Based",
    "bert_stochastic":     "BERT Stochastic",
    "ensemble":            "Ensemble",
}


class Reporter:
    def __init__(self, output_dir: str = "results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    # ── console ───────────────────────────────────────────────

    def print_summary(self, summary_df: pd.DataFrame):
        """Print a rich table to the console."""
        console.rule("[bold cyan]Hallucination Benchmark Results[/bold cyan]")

        for model_name, mdf in summary_df.groupby("model"):
            table = Table(
                title=f"Model: [bold green]{model_name}[/bold green]",
                box=box.ROUNDED,
                show_header=True,
                header_style="bold magenta",
            )
            table.add_column("Method",    style="cyan",    width=24)
            table.add_column("Dataset",   style="white",   width=18)
            table.add_column("Accuracy",  justify="right", style="yellow")
            table.add_column("Precision", justify="right", style="green")
            table.add_column("Recall",    justify="right", style="blue")
            table.add_column("F1",        justify="right", style="bold white")
            table.add_column("AUC-ROC",   justify="right", style="magenta")
            table.add_column("Samples",   justify="right", style="dim")

            for _, row in mdf.sort_values(["method", "dataset"]).iterrows():
                method_label = METHOD_LABELS.get(row["method"], row["method"])
                table.add_row(
                    method_label,
                    str(row.get("dataset", "-")),
                    f'{row.get("accuracy",  "-"):.4f}' if pd.notna(row.get("accuracy"))  else "-",
                    f'{row.get("precision", "-"):.4f}' if pd.notna(row.get("precision")) else "-",
                    f'{row.get("recall",    "-"):.4f}' if pd.notna(row.get("recall"))    else "-",
                    f'{row.get("f1",        "-"):.4f}' if pd.notna(row.get("f1"))        else "-",
                    f'{row.get("auc_roc",   "-"):.4f}' if pd.notna(row.get("auc_roc"))   else "-",
                    str(int(row.get("n_samples", 0))),
                )

            console.print(table)
            console.print()

    def print_comparison_table(self, comparison_df: pd.DataFrame):
        """Print the method comparison table (averaged across all models/datasets)."""
        console.rule("[bold yellow]Method Comparison (Averaged)[/bold yellow]")
        table = Table(box=box.DOUBLE_EDGE, header_style="bold cyan")
        table.add_column("Method",    style="cyan",  width=26)
        table.add_column("Accuracy",  justify="right", style="yellow")
        table.add_column("Precision", justify="right", style="green")
        table.add_column("Recall",    justify="right", style="blue")
        table.add_column("F1",        justify="right", style="bold white")
        table.add_column("AUC-ROC",   justify="right", style="magenta")

        for _, row in comparison_df.iterrows():
            label = METHOD_LABELS.get(row["method"], row["method"])
            table.add_row(
                label,
                f'{row.get("accuracy",  0):.4f}',
                f'{row.get("precision", 0):.4f}',
                f'{row.get("recall",    0):.4f}',
                f'{row.get("f1",        0):.4f}',
                f'{row.get("auc_roc",   0):.4f}',
            )
        console.print(table)

    # ── charts ────────────────────────────────────────────────

    def save_charts(self, summary_df: pd.DataFrame, results_df: pd.DataFrame):
        """Save all charts to the output directory."""
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import seaborn as sns
            sns.set_theme(style="darkgrid", palette="muted")
        except ImportError:
            logger.warning("matplotlib/seaborn not installed — skipping charts")
            return

        self._chart_method_comparison(summary_df, plt, sns)
        self._chart_model_heatmap(summary_df, plt, sns)
        self._chart_score_distributions(results_df, plt, sns)
        self._chart_precision_recall(summary_df, plt, sns)

        logger.info(f"Charts saved to {self.output_dir}/")

    def _chart_method_comparison(self, df: pd.DataFrame, plt, sns):
        """Grouped bar chart: Accuracy / Precision / Recall / F1 per method."""
        avg = (
            df.groupby("method")[["accuracy", "precision", "recall", "f1"]]
            .mean()
            .reset_index()
        )
        avg["method_label"] = avg["method"].map(METHOD_LABELS)
        avg_melt = avg.melt(
            id_vars="method_label",
            value_vars=["accuracy", "precision", "recall", "f1"],
            var_name="metric",
            value_name="score",
        )

        fig, ax = plt.subplots(figsize=(12, 6))
        sns.barplot(
            data=avg_melt, x="method_label", y="score", hue="metric", ax=ax
        )
        ax.set_title("Hallucination Detection: Method Comparison", fontsize=14, fontweight="bold")
        ax.set_xlabel("Detection Method")
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.05)
        ax.legend(title="Metric", loc="upper right")
        plt.xticks(rotation=20, ha="right")
        plt.tight_layout()
        plt.savefig(self.output_dir / "method_comparison.png", dpi=150)
        plt.close()

    def _chart_model_heatmap(self, df: pd.DataFrame, plt, sns):
        """Heatmap: F1 score for each (model, method)."""
        pivot = df.pivot_table(index="model", columns="method", values="f1", aggfunc="mean")
        # Re-order columns
        ordered = [m for m in METHOD_ORDER if m in pivot.columns]
        pivot   = pivot[ordered]
        pivot.columns = [METHOD_LABELS.get(c, c) for c in pivot.columns]

        fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns) * 2), max(4, len(pivot) * 1.5)))
        sns.heatmap(
            pivot, annot=True, fmt=".3f", cmap="RdYlGn",
            vmin=0, vmax=1, linewidths=0.5, ax=ax,
        )
        ax.set_title("F1 Score Heatmap: Model × Detection Method", fontsize=13, fontweight="bold")
        ax.set_xlabel("Detection Method")
        ax.set_ylabel("Model")
        plt.tight_layout()
        plt.savefig(self.output_dir / "model_method_heatmap.png", dpi=150)
        plt.close()

    def _chart_score_distributions(self, df: pd.DataFrame, plt, sns):
        """Violin / KDE plots of hallucination scores split by ground truth."""
        score_cols = {
            "token_score":    "Token Similarity",
            "semantic_score": "Semantic Similarity",
            "llm_score":      "LLM-Based",
            "bert_score":     "BERT Stochastic",
            "ensemble_score": "Ensemble",
        }
        available = [c for c in score_cols if c in df.columns]
        if not available:
            return

        n = len(available)
        fig, axes = plt.subplots(1, n, figsize=(n * 4, 5), sharey=True)
        if n == 1:
            axes = [axes]

        for ax, col in zip(axes, available):
            sub = df[[col, "gt_hallucinated"]].dropna()
            sub["label"] = sub["gt_hallucinated"].map({True: "Hallucinated", False: "Factual"})
            sns.violinplot(data=sub, x="label", y=col, ax=ax,
                           palette={"Factual": "#2ecc71", "Hallucinated": "#e74c3c"})
            ax.set_title(score_cols[col], fontsize=11)
            ax.set_xlabel("")
            ax.set_ylabel("Hallucination Score" if ax == axes[0] else "")
            ax.set_ylim(-0.05, 1.05)

        fig.suptitle("Score Distributions: Factual vs Hallucinated", fontsize=13, fontweight="bold")
        plt.tight_layout()
        plt.savefig(self.output_dir / "score_distributions.png", dpi=150)
        plt.close()

    def _chart_precision_recall(self, df: pd.DataFrame, plt, sns):
        """Precision-Recall scatter for all (model, method) pairs."""
        sub = df[["model", "method", "precision", "recall", "f1"]].dropna()
        if sub.empty:
            return

        fig, ax = plt.subplots(figsize=(9, 7))
        models  = sub["model"].unique()
        markers = ["o", "s", "^", "D", "P", "X", "*"]

        for i, method in enumerate(METHOD_ORDER):
            mdata = sub[sub["method"] == method]
            if mdata.empty:
                continue
            label = METHOD_LABELS.get(method, method)
            for j, model in enumerate(models):
                pt = mdata[mdata["model"] == model]
                if pt.empty:
                    continue
                ax.scatter(
                    pt["recall"].values, pt["precision"].values,
                    label=f"{label} / {model}",
                    marker=markers[j % len(markers)],
                    s=120, zorder=3,
                )
                for _, row in pt.iterrows():
                    ax.annotate(
                        f"{row['f1']:.2f}",
                        (row["recall"], row["precision"]),
                        textcoords="offset points", xytext=(5, 5),
                        fontsize=7, color="gray",
                    )

        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Recall",    fontsize=12)
        ax.set_ylabel("Precision", fontsize=12)
        ax.set_title("Precision vs Recall (F1 annotations)", fontsize=13, fontweight="bold")
        ax.legend(fontsize=7, loc="lower left", ncol=2)
        ax.grid(True, alpha=0.4)
        plt.tight_layout()
        plt.savefig(self.output_dir / "precision_recall.png", dpi=150)
        plt.close()

    # ── HTML report ───────────────────────────────────────────

    def save_html_report(self, summary_df: pd.DataFrame, results_df: pd.DataFrame):
        """Generate a self-contained HTML report."""
        import base64, io
        charts = {}
        for fname in ["method_comparison.png", "model_method_heatmap.png",
                      "score_distributions.png", "precision_recall.png"]:
            path = self.output_dir / fname
            if path.exists():
                with open(path, "rb") as f:
                    charts[fname] = base64.b64encode(f.read()).decode()

        def img_tag(name):
            if name in charts:
                return f'<img src="data:image/png;base64,{charts[name]}" style="max-width:100%;margin:12px 0;">'
            return ""

        table_html = summary_df.to_html(index=False, classes="df-table", border=0, float_format=lambda x: f"{x:.4f}")
        results_sample = results_df.head(50).to_html(index=False, classes="df-table", border=0)

        html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Hallucination Benchmark Report</title>
<style>
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
          background: #0f1117; color: #e0e0e0; max-width: 1400px; margin: 0 auto; padding: 24px; }}
  h1   {{ color: #64b5f6; border-bottom: 2px solid #1e88e5; padding-bottom: 8px; }}
  h2   {{ color: #81c784; margin-top: 36px; }}
  h3   {{ color: #ffb74d; }}
  .df-table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
  .df-table th {{ background: #1e3a5f; color: #90caf9; padding: 8px 12px; text-align: left; }}
  .df-table td {{ padding: 6px 12px; border-bottom: 1px solid #2a2a3e; }}
  .df-table tr:hover {{ background: #1a2030; }}
  .card {{ background: #1a1d2e; border: 1px solid #2a3050; border-radius: 10px; padding: 20px; margin: 16px 0; }}
  .badge {{ display:inline-block; padding:3px 10px; border-radius:12px; font-size:12px; margin:2px; }}
  .green  {{ background:#1b5e20; color:#a5d6a7; }}
  .blue   {{ background:#0d47a1; color:#90caf9; }}
  .orange {{ background:#e65100; color:#ffcc80; }}
  .caption {{ color:#aaa; font-size:12px; margin-top:6px; }}
  img {{ border-radius:8px; }}
</style>
</head>
<body>
<h1>🔬 Hallucination Detection Benchmark Report</h1>
<div class="card">
  <h3>Overview</h3>
  <p>Implements the 4 detection methods from the AWS ML blog:</p>
  <span class="badge orange">Token Similarity</span>
  <span class="badge blue">Semantic Similarity</span>
  <span class="badge green">LLM Prompt-Based</span>
  <span class="badge orange">BERT Stochastic</span>
  <span class="badge blue">Ensemble</span>
</div>

<h2>📊 Method Comparison Charts</h2>
<div class="card">{img_tag("method_comparison.png")}<p class="caption">Bar chart: Accuracy / Precision / Recall / F1 per detection method</p></div>
<div class="card">{img_tag("model_method_heatmap.png")}<p class="caption">Heatmap: F1 score for each (model × method) pair</p></div>
<div class="card">{img_tag("score_distributions.png")}<p class="caption">Score distributions split by ground-truth label</p></div>
<div class="card">{img_tag("precision_recall.png")}<p class="caption">Precision-Recall scatter (F1 annotations)</p></div>

<h2>📋 Full Metrics Table</h2>
<div class="card">{table_html}</div>

<h2>🗂 Sample Results (first 50 rows)</h2>
<div class="card">{results_sample}</div>

<p style="color:#555; font-size:12px; margin-top:40px;">
  Generated by hallucination_benchmark — based on methods from the AWS ML blog<br>
  "Detect hallucinations for RAG-based systems" (2025)
</p>
</body>
</html>"""

        path = self.output_dir / "report.html"
        with open(path, "w") as f:
            f.write(html)
        logger.info(f"HTML report saved to {path}")

    # ── JSON summary ─────────────────────────────────────────

    def save_json_summary(self, summary_df: pd.DataFrame):
        path = self.output_dir / "summary.json"
        summary_df.to_json(path, orient="records", indent=2)
        logger.info(f"JSON summary saved to {path}")
