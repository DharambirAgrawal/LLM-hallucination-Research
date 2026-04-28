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
 
  # 2. Pull any models you want to test
  ollama pull deepseek-r1:7b
  ollama pull qwen2.5:7b
 
  # 3. Run the experiment
  python main.py
 
COMMANDS
--------
  python main.py                                # run with config.yaml defaults
  python main.py --models deepseek-r1-7b        # specific models
  python main.py --pull deepseek-r1:7b          # pull models then run
  python main.py --list-models                  # show all installed Ollama models
  python main.py --list-available               # show all models in config.yaml
  python main.py --datasets synthetic           # single dataset
  python main.py --quick                        # 20 synthetic samples, fast
  python main.py --no-bert                      # skip BERT stochastic (faster)
  python main.py --dry-run                      # check setup only
  python main.py --host http://192.168.1.10:11434  # remote Ollama server
"""
 
import argparse
import sys
from pathlib import Path
 
import yaml
from loguru import logger
 
 
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
 
    # Apply CLI overrides
    if args.host:
        config.setdefault("ollama", {})["host"] = args.host
    if args.output:
        config.setdefault("benchmark", {})["output_dir"] = args.output
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

    # Ask how many samples to run
    try:
        n_samples = int(console.input(
            "[bold yellow]How many questions from HaluEval QA? [/bold yellow]"
            "[dim](press Enter for default 50)[/dim]: "
        ).strip() or "50")
    except ValueError:
        n_samples = 50
    console.print(f"  ✓ Questions: [green]{n_samples}[/green]")

    # Ask how many times to run (for consistency check)
    try:
        n_runs = int(console.input(
            "[bold yellow]How many runs? [/bold yellow]"
            "[dim](multiple runs check consistency, press Enter for 1)[/dim]: "
        ).strip() or "1")
    except ValueError:
        n_runs = 1
    console.print(f"  ✓ Number of runs: [green]{n_runs}[/green]\n")

    # Override config with user input
    for ds in config.get("datasets", []):
        ds["max_samples"] = n_samples

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
    from rich.console import Console
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
 
    # ── Run benchmark (multiple runs for consistency) ─────────────
    logger.info(f"\n[5/6] Running experiment ({n_runs} run(s))...")

    all_run_results = []
    for run_idx in range(n_runs):
        if n_runs > 1:
            logger.info(f"\n  === RUN {run_idx + 1} of {n_runs} ===")
        runner     = BenchmarkRunner(config)
        results_df = runner.run(models, datasets)
        results_df["run"] = run_idx + 1
        all_run_results.append(results_df)

    # Combine all runs
    combined_df = pd.concat(all_run_results, ignore_index=True)

    # If multiple runs, save consistency stats (std deviation across runs)
    # Low std dev = consistent results across runs = trustworthy
    if n_runs > 1:
        score_cols = ["token_score", "semantic_score", "bert_score", "llm_score"]
        score_cols = [c for c in score_cols if c in combined_df.columns]
        consistency = (
            combined_df.groupby(["model", "sample_id", "reducer"])[score_cols]
            .std()
            .round(4)
            .reset_index()
        )
        consistency_path = Path(out_dir) / "consistency_std.csv"
        consistency.to_csv(consistency_path, index=False)
        logger.info(f"  Consistency (std dev) saved to: {consistency_path}")
        logger.info(f"  Low std dev = consistent results across {n_runs} runs")

    # Average scores across all runs for final evaluation
    score_cols_present = [
        c for c in ["token_score", "semantic_score", "bert_score", "llm_score"]
        if c in combined_df.columns
    ]
    results_df = (
        combined_df.groupby(["model", "dataset", "sample_id", "reducer"])
                   [score_cols_present]
                   .mean()
                   .round(4)
                   .reset_index()
    )

    # ── Evaluate & report ─────────────────────────────────────
    logger.info("\n[6/6] Evaluating and generating reports...")
    evaluator    = Evaluator()
    summary_df   = evaluator.evaluate(results_df)
    reduction_df = evaluator.reducer_comparison(results_df)

    out_dir  = config.get("benchmark", {}).get("output_dir", "results")
    reporter = Reporter(output_dir=out_dir)

    # Console output — per model tables
    reporter.print_summary(summary_df)
    reporter.print_reduction_table(reduction_df)

    # Charts — one set per model (updated reporter handles this)
    reporter.save_charts(summary_df, reduction_df)

    # Save files
    reporter.save_json_summary(summary_df, reduction_df)
    reporter.save_html_report(summary_df, reduction_df, results_df)

    # Save raw results CSV
    out = Path(out_dir)
    results_df.to_csv(out / "all_results.csv", index=False)

    c.print(f"\n[bold green]✓ Experiment complete![/bold green]")
    c.print(f"  📁 Raw CSV:        [cyan]{out / 'all_results.csv'}[/cyan]")
    c.print(f"  📄 HTML Report:    [cyan]{out / 'report.html'}[/cyan]")
    c.print(f"  📊 Charts:         [cyan]{out}/<model_name>_*.png[/cyan]")
    c.print(f"  📋 JSON Summary:   [cyan]{out / 'summary.json'}[/cyan]")
    if n_runs > 1:
        c.print(f"  📈 Consistency:    [cyan]{out / 'consistency_std.csv'}[/cyan]")


if __name__ == "__main__":
    main()