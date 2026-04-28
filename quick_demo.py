#!/usr/bin/env python3
"""
quick_demo.py
=============
Smoke-test that verifies all 4 detectors work using ONLY synthetic data
and the Token + Semantic detectors (no Ollama needed).

Run this first to confirm your install is working:
    python quick_demo.py

To also test the LLM-based and BERT stochastic detectors you need
Ollama running with at least one model installed:
    ollama serve
    ollama pull llama3.2:3b
    python quick_demo.py --with-ollama llama3.2:3b
"""

import argparse
import sys
import time
import warnings
warnings.filterwarnings("ignore")

from rich.console import Console
from rich.table import Table
from rich import box

c = Console()

# ─────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--with-ollama", metavar="MODEL_TAG", default=None,
                   help="Also test LLM + BERT detectors using this Ollama model (e.g. llama3.2:3b)")
    p.add_argument("--host", default="http://localhost:11434")
    return p.parse_args()

# ─────────────────────────────────────────────────────────────
def metrics(y_true, y_pred):
    tp = sum(t and p for t, p in zip(y_true, y_pred))
    fp = sum(not t and p for t, p in zip(y_true, y_pred))
    tn = sum(not t and not p for t, p in zip(y_true, y_pred))
    fn = sum(t and not p for t, p in zip(y_true, y_pred))
    acc  = (tp + tn) / len(y_true) if y_true else 0
    prec = tp / (tp + fp) if (tp + fp) > 0 else 0
    rec  = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
    return dict(acc=acc, prec=prec, rec=rec, f1=f1, tp=tp, fp=fp, tn=tn, fn=fn)

# ─────────────────────────────────────────────────────────────
def main():
    args = parse_args()

    c.rule("[bold cyan]🔬 Hallucination Benchmark — Quick Demo[/bold cyan]")

    # ── 1. Synthetic data ─────────────────────────────────────
    c.print("\n[bold][1/4] Generating synthetic samples...[/bold]")
    sys.path.insert(0, ".")
    from data.datasets import _make_synthetic

    samples = _make_synthetic(max_samples=30, seed=42)
    y_true  = [s.is_hallucinated for s in samples]
    hal_n   = sum(y_true)
    fac_n   = len(y_true) - hal_n
    c.print(f"  {len(samples)} samples  |  {hal_n} hallucinated  |  {fac_n} factual")

    results = {}

    # ── 2. Token similarity ───────────────────────────────────
    c.print("\n[bold][2/4] Token Similarity Detector...[/bold]")
    from detectors.token_detector import TokenSimilarityDetector
    t0  = time.time()
    det = TokenSimilarityDetector()
    tok_preds = [det.detect(s.context, s.answer).is_hallucinated for s in samples]
    elapsed   = time.time() - t0
    results["Token Similarity"] = metrics(y_true, tok_preds)
    c.print(f"  Done in {elapsed:.2f}s")

    # ── 3. Semantic similarity ────────────────────────────────
    c.print("\n[bold][3/4] Semantic Similarity Detector (sentence-transformers)...[/bold]")
    try:
        from detectors.semantic_detector import SemanticSimilarityDetector
        t0  = time.time()
        det = SemanticSimilarityDetector(
            embedding_model="sentence-transformers/all-MiniLM-L6-v2",
            device="cpu",
        )
        rs      = det.detect_batch([s.context for s in samples], [s.answer for s in samples])
        elapsed = time.time() - t0
        sem_preds = [r.is_hallucinated for r in rs]
        results["Semantic Similarity"] = metrics(y_true, sem_preds)
        c.print(f"  Done in {elapsed:.2f}s")
    except Exception as e:
        c.print(f"  [yellow]⚠ Skipped: {e}[/yellow]")
        c.print("    pip install sentence-transformers")

    # ── 4. Ollama-based detectors ─────────────────────────────
    if args.with_ollama:
        tag = args.with_ollama
        c.print(f"\n[bold][4/4] LLM + BERT Detectors via Ollama ({tag})...[/bold]")
        try:
            from models.ollama_model import OllamaModel
            from detectors.llm_detector import LLMDetector
            from detectors.bert_detector import BERTStochasticDetector

            model = OllamaModel(
                name=tag,
                config={"model": tag, "temperature": 0.7, "max_tokens": 64},
                ollama_host=args.host,
            )

            # LLM detector
            c.print("  Running LLM-based detector...")
            t0       = time.time()
            llm_det  = LLMDetector(judge_model=model, threshold=0.5)
            llm_preds = [
                llm_det.detect(s.context, s.answer).is_hallucinated
                for s in samples
            ]
            results[f"LLM-Based ({tag})"] = metrics(y_true, llm_preds)
            c.print(f"  LLM done in {time.time()-t0:.1f}s")

            # BERT stochastic detector (3 samples for speed)
            c.print("  Running BERT Stochastic detector (n=3)...")
            t0        = time.time()
            bert_det  = BERTStochasticDetector(model=model, n_samples=3, use_fast_bert=True)
            bert_preds = [
                bert_det.detect(s.question, s.context, s.answer).is_hallucinated
                for s in samples
            ]
            results[f"BERT Stochastic ({tag})"] = metrics(y_true, bert_preds)
            c.print(f"  BERT done in {time.time()-t0:.1f}s")

        except Exception as e:
            c.print(f"  [red]✗ Ollama test failed: {e}[/red]")
            c.print(f"    Make sure Ollama is running and '{tag}' is pulled.")
    else:
        c.print(
            "\n[dim][4/4] Skipping LLM + BERT detectors "
            "(pass --with-ollama llama3.2:3b to enable)[/dim]"
        )

    # ── Results table ─────────────────────────────────────────
    c.print()
    c.rule("[bold yellow]Results[/bold yellow]")

    t = Table(box=box.ROUNDED, header_style="bold cyan")
    t.add_column("Detector",  style="cyan", width=30)
    t.add_column("Accuracy",  justify="right", style="yellow")
    t.add_column("Precision", justify="right", style="green")
    t.add_column("Recall",    justify="right", style="blue")
    t.add_column("F1",        justify="right", style="bold white")
    t.add_column("TP/FP/TN/FN", justify="right", style="dim")

    for name, m in results.items():
        t.add_row(
            name,
            f"{m['acc']:.3f}",
            f"{m['prec']:.3f}",
            f"{m['rec']:.3f}",
            f"{m['f1']:.3f}",
            f"{m['tp']}/{m['fp']}/{m['tn']}/{m['fn']}",
        )
    c.print(t)

    c.print()
    c.rule("[bold green]✅ Demo complete![/bold green]")
    c.print()
    c.print("  Next steps:")
    c.print("  [cyan]python main.py --list-models[/cyan]       ← see installed models")
    c.print("  [cyan]python main.py --list-available[/cyan]    ← see models configured in config.yaml")
    c.print("  [cyan]python main.py --models <name> --datasets synthetic --samples 5 --runs 1 --no-prompt[/cyan]")
    c.print("  [dim]Tip: --models expects the display name from config.yaml (models[].name).[/dim]")
    c.print()


if __name__ == "__main__":
    main()
