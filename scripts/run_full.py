#!/usr/bin/env python3
"""Orchestrate a "full run": every requested detector, each in its own
isolated venv, plus the reduction stage, collected into one output folder
with one combined report.

Why per-detector venvs: SelfCheckGPT, MiniCheck, SummaC, and AlignScore each
pin their own (conflicting) upstream torch/transformers versions. That is a
real constraint of the upstream packages, not something this script can paper
over — see docs/HOW_TO_RUN.md §3. This script just automates creating and
reusing one venv per detector instead of doing it by hand.

Run this with the base controller environment's Python (the one used for
`python main.py --dry-run` / requirements.txt) — it only needs stdlib
`venv`/`subprocess` itself; each detector's own dependencies are installed
into that detector's child venv, not into the interpreter running this
script.

Example — the 5-sample, 2-turn smoke version of a full run:

    python scripts/run_full.py --max-samples 5 --n-samples 2 --max-iterations 2

Example — the full-size run, every detector, using config.yaml's own sample
sizes:

    python scripts/run_full.py --detectors selfcheckgpt minicheck summac
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Each entry: pip requirements file for that detector's isolated venv, extra
# main.py flags, and (alignscore only) a checkpoint that must already exist
# since it is not auto-downloaded.
DETECTOR_SETUP = {
    "selfcheckgpt": {
        "requirements": "requirements-colab-smoke.txt",
        "extra_args": lambda a: [
            "--reduce",
            "--n-samples", str(a.n_samples),
            "--max-iterations", str(a.max_iterations),
        ],
        "checkpoint": None,
    },
    "minicheck": {
        "requirements": "requirements-colab-minicheck.txt",
        "extra_args": lambda a: [],
        "checkpoint": None,
    },
    "summac": {
        "requirements": "requirements-summac.txt",
        "extra_args": lambda a: [],
        "checkpoint": None,
    },
    "alignscore": {
        "requirements": "requirements-alignscore.txt",
        "extra_args": lambda a: [],
        "checkpoint": ROOT / "external_models" / "alignscore" / "AlignScore-base.ckpt",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--output",
        help="Combined output folder (default: results/full-run-<timestamp>)",
    )
    parser.add_argument(
        "--max-samples", type=int,
        help="Cap every dataset to N samples (omit to use config.yaml's own sizes)",
    )
    parser.add_argument(
        "--n-samples", type=int, default=5,
        help="SelfCheckGPT generations per case (default: 5)",
    )
    parser.add_argument(
        "--max-iterations", type=int, default=3,
        help="Reduction feedback/refine steps (default: 3)",
    )
    parser.add_argument(
        "--detectors", nargs="+", default=["selfcheckgpt", "minicheck", "summac"],
        choices=sorted(DETECTOR_SETUP),
        help="Detectors to run, each in its own venv (default: selfcheckgpt "
             "minicheck summac). alignscore is opt-in — it also needs a "
             "manually downloaded checkpoint; see METHOD_SOURCES.md.",
    )
    parser.add_argument(
        "--skip-report", action="store_true",
        help="Only run detectors/reduction; skip chart and report generation",
    )
    return parser.parse_args()


def run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}", flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def ensure_venv(name: str, requirements: str) -> Path:
    venv_dir = ROOT / f".venv-{name}"
    python = venv_dir / "bin" / "python"
    if not python.exists():
        print(f"== creating {venv_dir.name} (installing {requirements}) ==")
        run([sys.executable, "-m", "venv", str(venv_dir)])
        run([str(venv_dir / "bin" / "pip"), "install", "-r", requirements])
    else:
        print(f"== reusing {venv_dir.name} ==")
    return python


def main() -> None:
    args = parse_args()
    output_dir = (
        Path(args.output) if args.output
        else ROOT / "results" / f"full-run-{datetime.now():%Y%m%d-%H%M%S}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    ran = []
    failed = []
    for name in args.detectors:
        setup = DETECTOR_SETUP[name]
        checkpoint = setup["checkpoint"]
        if checkpoint is not None and not checkpoint.exists():
            print(
                f"== skipping {name}: checkpoint not found at {checkpoint} — "
                f"download it first (see METHOD_SOURCES.md), then rerun with "
                f"--detectors {name} =="
            )
            continue

        detector_output = output_dir / name
        cmd_args = [
            "--detectors", name,
            "--output", str(detector_output),
        ]
        if args.max_samples is not None:
            cmd_args += ["--max-samples", str(args.max_samples)]
        cmd_args += setup["extra_args"](args)

        try:
            python = ensure_venv(name, setup["requirements"])
            print(f"== running {name} ==")
            run([str(python), "main.py", *cmd_args])
            ran.append(name)
        except subprocess.CalledProcessError as exc:
            print(f"== {name} failed (exit {exc.returncode}); continuing with "
                  f"the remaining detectors ==")
            failed.append(name)

    if not ran:
        raise SystemExit("No detector produced results; nothing to report.")

    print(f"== ran: {', '.join(ran)} ==" + (f"  (failed: {', '.join(failed)})" if failed else ""))
    if not args.skip_report:
        run([sys.executable, str(ROOT / "scripts" / "generate_report.py"),
             "--input", str(output_dir)])

    print(f"\nFull run complete: {output_dir}")


if __name__ == "__main__":
    main()
