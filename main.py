#!/usr/bin/env python3
"""
LLM Hallucination Reduction Experiment — Ollama Edition
========================================================
 
Test whether different reduction methods (RAG, Constrained Decoding,
Self-Verification) actually reduce hallucinations in LLM outputs.
 
Workflow:
  1. For each question in the dataset, model generates a baseline answer
  2. Detectors score that answer against the correct answer
  3. For each reducer, model generates a new answer using that strategy
  4. Detectors score the new answer
  5. Compare score changes to see which reducer helps most
 
Run any LLM through Ollama. No API keys. No GPU driver setup.
Just install Ollama, pull models, and run the experiment.
 
QUICK START
-----------
  # 1. Install Ollama
  curl -fsSL https://ollama.com/install.sh | sh
 
    # 2. Pull at least one model tag (must match config.yaml -> models[].model)
    # Tip: run `python main.py --list-available` to see the exact tags configured.
    ollama pull <model-tag>
 
  # 3. Run the experiment
  python main.py
 
COMMANDS
--------
  python main.py                                # run with config.yaml defaults
    python main.py --models llama3:latest         # run only specific models (names from config.yaml)
  python main.py --pull deepseek-r1:7b          # pull models then run
  python main.py --list-models                  # show all installed Ollama models
  python main.py --list-available               # show all models in config.yaml
  python main.py --datasets synthetic           # single dataset
  python main.py --quick                        # 20 synthetic samples, fast
  python main.py --no-bert                      # skip BERT stochastic (faster)
    python main.py --docx                         # also generate report.docx (tables + embedded charts)
  python main.py --dry-run                      # check setup only
  python main.py --host http://192.168.1.10:11434  # remote Ollama server
"""
 
import argparse
import copy
import sys
from datetime import datetime
from pathlib import Path
 
import pandas as pd
import yaml
from loguru import logger
from rich.console import Console
 
 
# ─────────────────────────────────────────────────────────────
# Argument parsing
# ─────────────────────────────────────────────────────────────
 
def parse_args():
    p = argparse.ArgumentParser(
        description="Hallucination Detection Benchmark (Ollama Edition)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--config",   default="config.yaml",
                   help="YAML config file (default: config.yaml)")
    p.add_argument("--models",   nargs="+", default=None,
                   help="Run only these model display-names from config")
    p.add_argument("--datasets", nargs="+", default=None,
                   help="Run only these dataset names")
    p.add_argument("--output",   default=None,
                   help="Override output directory")
    p.add_argument("--host",     default=None,
                   help="Ollama server URL (default: http://localhost:11434)")

    # Run controls
    p.add_argument("--runs", type=int, default=None,
                   help="Number of repeated runs (default: prompt or 1)")
    p.add_argument("--samples", type=int, default=None,
                   help="Max samples per dataset (default: prompt or config.yaml)")
    p.add_argument("--no-prompt", action="store_true",
                   help="Do not prompt; use config.yaml / CLI defaults")
 
    # Utility commands
    p.add_argument("--list-models",     action="store_true",
                   help="List all installed Ollama models and exit")
    p.add_argument("--list-available",  action="store_true",
                   help="List all models defined in config.yaml and exit")
    p.add_argument("--pull",            nargs="+", metavar="TAG",
                   help="Pull model tags via Ollama then run benchmark (e.g. llama3.2:3b)")
 
    # Speed flags
    p.add_argument("--quick",    action="store_true",
                   help="Quick mode: 20 synthetic samples only")
    p.add_argument("--no-llm",   action="store_true",
                   help="Disable LLM-based detector")
    p.add_argument("--no-bert",  action="store_true",
                   help="Disable BERT stochastic detector (much faster)")
    p.add_argument("--no-semantic", action="store_true",
                   help="Disable semantic similarity detector")
    p.add_argument("--dry-run",  action="store_true",
                   help="Check setup and model availability, then exit")
    p.add_argument("--docx", action="store_true",
                   help="Also generate report.docx with tables + embedded charts (requires python-docx)")
    return p.parse_args()
 
 
# ─────────────────────────────────────────────────────────────
# Logging
# ─────────────────────────────────────────────────────────────
 
def setup_logging(config: dict):
    log_cfg = config.get("logging", {})
    level   = log_cfg.get("level", "INFO")
    logger.remove()
    logger.add(
        sys.stderr, level=level,
        format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
        colorize=True,
    )
    if log_cfg.get("save_logs"):
        lf = Path(log_cfg.get("log_file", "results/benchmark.log"))
        lf.parent.mkdir(parents=True, exist_ok=True)
        logger.add(lf, level="DEBUG",
                   format="{time:YYYY-MM-DD HH:mm:ss} | {level:<8} | {message}")
 
 
# ─────────────────────────────────────────────────────────────
# Utility: list models
# ─────────────────────────────────────────────────────────────
 
def cmd_list_installed(host: str):
    """Print all Ollama-installed models."""
    from models.model_factory import ModelFactory
    ModelFactory.show_available(host)
 
 
def cmd_list_available(config: dict):
    """Print all models defined in config.yaml."""
    from rich.console import Console
    from rich.table import Table
    from rich import box
 
    c = Console()
    t = Table(
        title="Models defined in config.yaml",
        box=box.ROUNDED, header_style="bold cyan",
    )
    t.add_column("Name",   style="green")
    t.add_column("Tag",    style="yellow")
    t.add_column("Family", style="magenta")
    t.add_column("Auto-pull", justify="center")
 
    for m in config.get("models", []):
        t.add_row(
            m["name"],
            m["model"],
            m.get("family", "?"),
            "✓" if m.get("auto_pull") else "—",
        )
    c.print(t)
    c.print("\n[dim]To add a model: edit config.yaml → models section.[/dim]")
    c.print("[dim]Find model tags at: https://ollama.com/library[/dim]")
 
 
# ─────────────────────────────────────────────────────────────
# Utility: pull models
# ─────────────────────────────────────────────────────────────
 
def cmd_pull(tags: list[str], host: str):
    """Pull one or more model tags via Ollama."""
    import ollama as _ollama
    from rich.console import Console
    c = Console()
    client = _ollama.Client(host=host)
 
    for tag in tags:
        c.print(f"\n[cyan]Pulling {tag}...[/cyan]")
        try:
            for progress in client.pull(tag, stream=True):
                status    = progress.get("status", "")
                completed = progress.get("completed", 0)
                total     = progress.get("total", 0)
                if total:
                    pct = completed / total * 100
                    print(f"\r  {status}: {pct:.1f}%  ", end="", flush=True)
                elif status:
                    print(f"\r  {status}         ", end="", flush=True)
            print()
            c.print(f"[green]  ✓ {tag} ready[/green]")
        except Exception as e:
            c.print(f"[red]  ✗ Failed to pull {tag}: {e}[/red]")
 
 
# ─────────────────────────────────────────────────────────────
# Ollama connectivity check
# ─────────────────────────────────────────────────────────────
 
def check_ollama(host: str) -> bool:
    """Returns True if Ollama is reachable."""
    try:
        import ollama as _ollama
        client = _ollama.Client(host=host)
        client.list()
        logger.info(f"  ✓ Ollama is running at {host}")
        return True
    except Exception as e:
        logger.error(
            f"  ✗ Cannot reach Ollama at {host}\n"
            f"    Error: {e}\n\n"
            f"    ➜  Install:  curl -fsSL https://ollama.com/install.sh | sh\n"
            f"    ➜  Start:    ollama serve\n"
            f"    ➜  Windows/Mac: download from https://ollama.com/download"
        )
        return False
 
 
# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────
 
def main():
    args = parse_args()
 
    # Load config
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        print(f"ERROR: Config file not found: {cfg_path}")
        sys.exit(1)
 
    with open(cfg_path) as f:
        config = yaml.safe_load(f)

    # Ensure nested config sections exist
    config.setdefault("benchmark", {})
    config.setdefault("logging", {})
 
    # Apply CLI overrides
    if args.host:
        config.setdefault("ollama", {})["host"] = args.host
    if args.output:
        config["benchmark"]["output_dir"] = args.output
    if args.models:
        config["selected_models"] = args.models
    if args.no_llm:
        config.setdefault("detectors", {}).setdefault("llm_based", {})["enabled"] = False
    if args.no_bert:
        config.setdefault("detectors", {}).setdefault("bert_stochastic", {})["enabled"] = False
    if args.no_semantic:
        config.setdefault("detectors", {}).setdefault("semantic_similarity", {})["enabled"] = False
    if args.quick:
        config["datasets"] = [{"name": "synthetic", "source": "synthetic", "max_samples": 20}]
        config.setdefault("detectors", {}).setdefault("bert_stochastic", {})["n_samples"] = 3
    if args.datasets:
        config["datasets"] = [
            d for d in config.get("datasets", []) if d["name"] in args.datasets
        ]

    # Establish output root early (for logs + per-run folders)
    out_root = Path(config.get("benchmark", {}).get("output_dir", "results"))
    out_root.mkdir(parents=True, exist_ok=True)

    # If log_file is not explicitly set (or still points to default results/),
    # keep logs inside the chosen output directory.
    if config.get("logging", {}).get("save_logs", True):
        config["logging"].setdefault("log_file", str(out_root / "benchmark.log"))
 
    setup_logging(config)
    host = config.get("ollama", {}).get("host", "http://localhost:11434")
 
    # ── Utility commands ──────────────────────────────────────
    if args.list_models:
        cmd_list_installed(host)
        return
 
    if args.list_available:
        cmd_list_available(config)
        return
 
    # ── Banner ────────────────────────────────────────────────
    logger.info("=" * 65)
    logger.info("  🔬 Hallucination Detection Benchmark — Ollama Edition")
    logger.info("=" * 65)
 
    # ── Ollama check ──────────────────────────────────────────
    logger.info("\n[1/6] Checking Ollama connection...")
    if not check_ollama(host):
        sys.exit(1)
 
    # ── Pull if requested ─────────────────────────────────────
    if args.pull:
        logger.info("\n[2/6] Pulling models...")
        cmd_pull(args.pull, host)
        # After pull, mark those as auto_pull=False (already pulled)
    else:
        logger.info("\n[2/6] (skipping pull — use --pull <tag> to pull models)")
 
    if args.dry_run:
        logger.info("\n[dry-run] Checking model availability...")
        from models.model_factory import ModelFactory
        models = ModelFactory.build_all(config)
        logger.info(f"\n  Models that would run: {[m.name for m in models]}")
        logger.info("Dry run complete — exiting.")
        return
 
    # Deferred imports (after logging is set up)
    from data.datasets import DatasetLoader
    from models.model_factory import ModelFactory
    from benchmark.runner import BenchmarkRunner
    from benchmark.evaluator import Evaluator
    from benchmark.reporter import Reporter
 
    # ── Interactive prompts ───────────────────────────────────────
    console = Console()
    console.rule("[bold cyan]Experiment Setup[/bold cyan]")

    # Determine max samples per dataset
    n_samples = args.samples
    if n_samples is None and not args.no_prompt:
        default_samples = None
        for ds in config.get("datasets", []):
            if "max_samples" in ds:
                default_samples = ds.get("max_samples")
                break
        default_samples = int(default_samples or 50)
        try:
            n_samples = int(
                console.input(
                    "[bold yellow]How many samples per dataset? [/bold yellow]"
                    f"[dim](press Enter for default {default_samples})[/dim]: "
                ).strip() or str(default_samples)
            )
        except ValueError:
            n_samples = default_samples

    if n_samples is not None:
        for ds in config.get("datasets", []):
            ds["max_samples"] = int(n_samples)
        console.print(f"  ✓ Samples per dataset: [green]{int(n_samples)}[/green]")
    else:
        console.print("  ✓ Samples per dataset: [green](from config.yaml)[/green]")

    # Determine number of runs
    n_runs = args.runs
    if n_runs is None:
        if args.no_prompt:
            n_runs = 1
        else:
            try:
                n_runs = int(
                    console.input(
                        "[bold yellow]How many runs? [/bold yellow]"
                        "[dim](multiple runs check consistency, press Enter for 1)[/dim]: "
                    ).strip() or "1"
                )
            except ValueError:
                n_runs = 1

    n_runs = max(int(n_runs), 1)
    console.print(f"  ✓ Number of runs: [green]{n_runs}[/green]\n")

    # ── Load datasets ─────────────────────────────────────────
    logger.info("\n[3/6] Loading datasets...")
    loader   = DatasetLoader(config, seed=config.get("benchmark", {}).get("seed", 42))
    datasets = loader.load_all()
 
    if not datasets:
        logger.error("No datasets loaded! Check config.yaml")
        sys.exit(1)
 
    total = sum(len(v) for v in datasets.values())
    logger.info(f"  Total: {total} samples across {len(datasets)} datasets")
 
    # ── Build models ──────────────────────────────────────────
    logger.info("\n[4/6] Registering models...")
    models = ModelFactory.build_all(config)
 
    if not models:
        logger.error(
            "No models available!\n"
            "  • Run: python main.py --list-models    (see what's installed)\n"
            "  • Run: ollama pull llama3.2:3b          (install a model)\n"
            "  • Run: python main.py --pull llama3.2:3b mistral:7b"
        )
        sys.exit(1)
 
    # Print what we're about to run
    from rich.table import Table
    from rich import box
    c = Console()
    t = Table(title="Benchmark Plan", box=box.SIMPLE, header_style="bold cyan")
    t.add_column("Model",   style="green")
    t.add_column("Tag",     style="yellow")
    t.add_column("Family",  style="magenta")
    for m in models:
        t.add_row(m.name, m.config["model"], m.config.get("family", "?"))
    c.print(t)
 
    det_cfg = config.get("detectors", {})
    enabled_det = [k for k, v in det_cfg.items() if v.get("enabled", True)]
    logger.info(f"  Detectors: {', '.join(enabled_det)}")
 
    red_cfg = config.get("reducers", {})
    enabled_red = [k for k, v in red_cfg.items() if v.get("enabled", True)]
    logger.info(f"  Reducers:  baseline, {', '.join(enabled_red)}")
    logger.info(f"  Datasets:  {', '.join(datasets.keys())}")

    # ── Run benchmark (per-run folders) ─────────────────────────
    logger.info(f"\n[5/6] Running experiment ({n_runs} run(s))...")

    all_run_dfs: list[pd.DataFrame] = []
    base_out = Path(config.get("benchmark", {}).get("output_dir", "results"))
    base_out.mkdir(parents=True, exist_ok=True)

    for run_idx in range(n_runs):
        run_name = f"run_{run_idx + 1:02d}"
        run_out = base_out / run_name
        run_out.mkdir(parents=True, exist_ok=True)

        run_cfg = copy.deepcopy(config)
        run_cfg.setdefault("benchmark", {})["output_dir"] = str(run_out)
        run_cfg.setdefault("logging", {})
        if run_cfg.get("logging", {}).get("save_logs", True):
            run_cfg["logging"]["log_file"] = str(run_out / "benchmark.log")

        # Reconfigure logging so each run captures a dedicated log file
        setup_logging(run_cfg)
        logger.info(f"\n  === {run_name} of {n_runs} ===")

        # Persist exact run config for reproducibility
        with open(run_out / "config_used.yaml", "w") as f:
            yaml.safe_dump(run_cfg, f, sort_keys=False)

        # Execute run
        runner = BenchmarkRunner(run_cfg)
        run_df = runner.run(models, datasets)
        run_df["run"] = run_idx + 1
        all_run_dfs.append(run_df)

        # Evaluate & report for this run
        evaluator = Evaluator()
        summary_df = evaluator.evaluate(run_df)
        reduction_df = evaluator.reducer_comparison(run_df)

        reporter = Reporter(output_dir=str(run_out))
        reporter.save_charts(summary_df, reduction_df)
        reporter.save_json_summary(summary_df, reduction_df)
        reporter.save_html_report(summary_df, reduction_df, run_df)
        if args.docx:
            reporter.save_docx_report(summary_df, reduction_df, run_df)

        # Save tabular summaries for easy analysis
        summary_df.to_csv(run_out / "summary.csv", index=False)
        reduction_df.to_csv(run_out / "reductions.csv", index=False)

    # ── Combined / average outputs ───────────────────────────
    logger.info("\n[6/6] Generating combined (average) report...")
    combined_out = base_out / "combined"
    combined_out.mkdir(parents=True, exist_ok=True)

    # Switch logging to combined folder
    combined_cfg = copy.deepcopy(config)
    combined_cfg.setdefault("logging", {})
    if combined_cfg.get("logging", {}).get("save_logs", True):
        combined_cfg["logging"]["log_file"] = str(combined_out / "benchmark.log")
    setup_logging(combined_cfg)

    combined_all = pd.concat(all_run_dfs, ignore_index=True) if all_run_dfs else pd.DataFrame()
    if len(combined_all) == 0:
        logger.error("No results produced; cannot generate combined report.")
        sys.exit(1)

    combined_all.to_csv(combined_out / "all_results_all_runs.csv", index=False)

    # Average numeric columns per (model, dataset, sample, reducer)
    group_cols = ["model", "dataset", "sample_id", "reducer"]
    numeric_cols = [
        c for c in combined_all.columns
        if c.endswith("_score") or c == "latency_s"
    ]

    agg: dict[str, str] = {c: "mean" for c in numeric_cols}
    for c in ["question", "right_answer"]:
        if c in combined_all.columns:
            agg[c] = "first"

    combined_mean = (
        combined_all.groupby(group_cols, dropna=False)
                    .agg(agg)
                    .round(4)
                    .reset_index()
    )
    combined_mean.to_csv(combined_out / "all_results_mean.csv", index=False)

    # Consistency: std dev across runs per (model, dataset, sample, reducer)
    if n_runs > 1 and numeric_cols:
        combined_std = (
            combined_all.groupby(group_cols, dropna=False)[numeric_cols]
                        .std()
                        .round(4)
                        .reset_index()
        )
        combined_std.to_csv(combined_out / "consistency_std.csv", index=False)

    # Evaluate & report on averaged-per-sample results
    evaluator = Evaluator()
    summary_df = evaluator.evaluate(combined_mean)
    reduction_df = evaluator.reducer_comparison(combined_mean)

    reporter = Reporter(output_dir=str(combined_out))
    reporter.save_charts(summary_df, reduction_df)
    reporter.save_json_summary(summary_df, reduction_df)
    reporter.save_html_report(summary_df, reduction_df, combined_mean)
    if args.docx:
        reporter.save_docx_report(summary_df, reduction_df, combined_mean)

    summary_df.to_csv(combined_out / "summary.csv", index=False)
    reduction_df.to_csv(combined_out / "reductions.csv", index=False)

    with open(combined_out / "config_used.yaml", "w") as f:
        yaml.safe_dump(config, f, sort_keys=False)

    # Simple takeaways derived from the combined summary/reductions
    def _to_md_table(df: pd.DataFrame) -> str:
        headers = list(df.columns)

        def _fmt(v) -> str:
            if pd.isna(v):
                return ""
            if isinstance(v, float):
                return f"{v:.4f}"
            return str(v)

        lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join(["---"] * len(headers)) + " |",
        ]
        for _, row in df.iterrows():
            lines.append("| " + " | ".join(_fmt(row[h]) for h in headers) + " |")
        return "\n".join(lines)

    takeaways_lines = [
        "# Takeaways (Combined Average)",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Runs: {n_runs}",
        "",
        "This file is generated from `combined/summary.csv` and `combined/reductions.csv`.",
        "All numbers are computed from the benchmark outputs (no fabricated results).",
        "",
    ]

    score_cols = [c for c in ["token_score", "semantic_score", "bert_score", "llm_score"] if c in summary_df.columns]
    if score_cols:
        overall = (
            summary_df.groupby("reducer")[score_cols + (["mean_latency_s"] if "mean_latency_s" in summary_df.columns else [])]
                     .mean()
                     .round(4)
                     .reset_index()
        )
        takeaways_lines.append("## Overall mean scores (lower = better)")
        takeaways_lines.append("")
        takeaways_lines.append(_to_md_table(overall))
        takeaways_lines.append("")

    reduction_cols = [c for c in reduction_df.columns if c.endswith("_reduction")]
    if len(reduction_df) > 0 and reduction_cols:
        overall_red = (
            reduction_df.groupby("reducer")[reduction_cols]
                        .mean()
                        .round(4)
                        .reset_index()
        )
        takeaways_lines.append("## Overall mean reduction vs baseline (higher = better)")
        takeaways_lines.append("")
        takeaways_lines.append(_to_md_table(overall_red))
        takeaways_lines.append("")

    # Per-model best reducer (by mean score across available detectors)
    if score_cols and "model" in summary_df.columns:
        per_model = (
            summary_df.groupby(["model", "reducer"])[
                score_cols + (["mean_latency_s"] if "mean_latency_s" in summary_df.columns else [])
            ]
            .mean()
            .round(4)
            .reset_index()
        )
        per_model["mean_score"] = per_model[score_cols].mean(axis=1).round(4)

        # Baseline rows for reference
        baseline_ref = per_model[per_model["reducer"] == "baseline"][
            ["model", "mean_score"] + (["mean_latency_s"] if "mean_latency_s" in per_model.columns else [])
        ].rename(columns={
            "mean_score": "baseline_mean_score",
            "mean_latency_s": "baseline_mean_latency_s",
        })

        # Best (lowest) mean score per model
        best_rows = (
            per_model.sort_values(["model", "mean_score"], ascending=[True, True])
                     .groupby("model")
                     .head(1)
                     .copy()
        )
        best_rows = best_rows.merge(baseline_ref, on="model", how="left")
        best_rows["score_delta_vs_baseline"] = (
            best_rows["baseline_mean_score"] - best_rows["mean_score"]
        ).round(4)
        if "mean_latency_s" in best_rows.columns and "baseline_mean_latency_s" in best_rows.columns:
            best_rows["latency_delta_s"] = (
                best_rows["mean_latency_s"] - best_rows["baseline_mean_latency_s"]
            ).round(4)

        display_cols = [
            "model",
            "reducer",
            "mean_score",
            "baseline_mean_score",
            "score_delta_vs_baseline",
        ]
        if "mean_latency_s" in best_rows.columns:
            display_cols += ["mean_latency_s", "baseline_mean_latency_s", "latency_delta_s"]

        takeaways_lines.append("## Best reducer per model (by mean score across detectors)")
        takeaways_lines.append("")
        takeaways_lines.append(
            "Mean score is the simple average of the available detector scores in this run folder."
        )
        takeaways_lines.append("")
        takeaways_lines.append(_to_md_table(best_rows[display_cols]))
        takeaways_lines.append("")

    # Per-model best reducer (by mean reduction vs baseline)
    if len(reduction_df) > 0 and reduction_cols and "model" in reduction_df.columns:
        red = reduction_df.copy()
        red["mean_reduction"] = red[reduction_cols].mean(axis=1).round(4)
        best_red = (
            red.sort_values(["model", "mean_reduction"], ascending=[True, False])
               .groupby("model")
               .head(1)
               .reset_index(drop=True)
        )
        takeaways_lines.append("## Best reducer per model (by mean reduction vs baseline)")
        takeaways_lines.append("")
        takeaways_lines.append(_to_md_table(best_red[["model", "reducer", "mean_reduction"]]))
        takeaways_lines.append("")

    with open(combined_out / "takeaways.md", "w") as f:
        f.write("\n".join(takeaways_lines))

    # Final console message
    c.print(f"\n[bold green]✓ Experiment complete![/bold green]")
    c.print(f"  📁 Output root:    [cyan]{base_out}[/cyan]")
    c.print(f"  📁 Per-run folders:[cyan] {base_out}/run_XX/[/cyan]")
    c.print(f"  📁 Combined avg:   [cyan]{combined_out}[/cyan]")
    c.print(f"  📄 Combined report:[cyan]{combined_out / 'report.html'}[/cyan]")
    if n_runs > 1:
        c.print(f"  📈 Consistency:    [cyan]{combined_out / 'consistency_std.csv'}[/cyan]")


if __name__ == "__main__":
    main()