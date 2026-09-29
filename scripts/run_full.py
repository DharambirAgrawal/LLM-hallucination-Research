#!/usr/bin/env python3
"""Orchestrate a "full run": every requested detector, each in its own
isolated venv, plus the reduction stage, collected into one output folder
with one combined report.

Why per-detector venvs: SelfCheckGPT, MiniCheck, SummaC, and AlignScore each
pin their own (conflicting) upstream torch/transformers versions. That is a
real constraint of the upstream packages, not something this script can paper
over — see docs/HOW_TO_RUN.md §3. This script just automates creating and
reusing one venv per detector instead of doing it by hand.

Run this with the base controller environment's Python (docs/HOW_TO_RUN.md
§0.3). SelfCheckGPT runs directly in that environment when it is installed
there; every other detector gets its own `.venv-<name>`, created on first use
and reinstalled automatically if its requirements file changed or a previous
install did not finish. pip output goes to `<output>/logs/`, not the screen.

Example — the 5-sample, 2-turn smoke version of a full run:

    python scripts/run_full.py --max-samples 5 --n-samples 2 --max-iterations 2

Example — the full-size run, every detector, using config.yaml's own sample
sizes:

    python scripts/run_full.py --detectors selfcheckgpt minicheck summac
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from utils import console  # noqa: E402

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
MARKER = ".requirements.sha256"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--config", default="config.yaml")
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
        "--device", choices=("cpu", "cuda"),
        help="Device for the torch-based detectors (passed to main.py --device)",
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
        help="Only run detectors/reduction; skip the combined chart and report",
    )
    return parser.parse_args()


def requirements_hash(requirements: str) -> str:
    """Hash of the requirements file plus every file it includes with -r."""
    digest = hashlib.sha256()
    pending = [ROOT / requirements]
    while pending:
        path = pending.pop()
        text = path.read_text(encoding="utf-8")
        digest.update(text.encode())
        for line in text.splitlines():
            if line.strip().startswith("-r "):
                pending.append(path.parent / line.strip()[3:].strip())
    return digest.hexdigest()


def ensure_python(name: str, requirements: str, log_dir: Path) -> Path:
    """Return the interpreter that runs `name`, installing its venv if needed."""
    if name == "selfcheckgpt" and importlib.util.find_spec("selfcheckgpt"):
        console.line("environment      this interpreter (selfcheckgpt already installed)")
        return Path(sys.executable)

    venv_dir = ROOT / f".venv-{name}"
    python = venv_dir / "bin" / "python"
    marker = venv_dir / MARKER
    wanted = requirements_hash(requirements)
    if python.exists() and marker.is_file() and marker.read_text().strip() == wanted:
        console.line(f"environment      {venv_dir.name} (up to date)")
        return python

    log = log_dir / f"pip-{name}.log"
    reason = "first use" if not python.exists() else "requirements changed or last install incomplete"
    console.line(f"environment      installing {venv_dir.name} from {requirements} ({reason})")
    console.line(f"                 this can take several minutes · pip log: {log}")
    started = time.monotonic()
    with log.open("w", encoding="utf-8") as handle:
        if not python.exists():
            subprocess.run([sys.executable, "-m", "venv", str(venv_dir)],
                           check=True, stdout=handle, stderr=subprocess.STDOUT)
        subprocess.run([str(python), "-m", "pip", "install", "--upgrade", "pip"],
                       check=True, stdout=handle, stderr=subprocess.STDOUT, cwd=ROOT)
        subprocess.run([str(python), "-m", "pip", "install", "-r", requirements],
                       check=True, stdout=handle, stderr=subprocess.STDOUT, cwd=ROOT)
    marker.write_text(wanted)
    console.line(f"                 installed in {console.duration(time.monotonic() - started)}")
    return python


def main() -> None:
    args = parse_args()
    started = time.monotonic()
    output_dir = (
        Path(args.output) if args.output
        else ROOT / "results" / f"full-run-{datetime.now():%Y%m%d-%H%M%S}"
    )
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    console.header("Full run · every detector in its own environment")
    console.kv("detectors", ", ".join(args.detectors))
    console.kv("config", args.config)
    console.kv("sample size", f"{args.max_samples} per dataset" if args.max_samples
               else "config.yaml max_samples")
    console.kv("selfcheckgpt", f"n_samples={args.n_samples} · reduction max_iterations={args.max_iterations}")
    console.kv("output", output_dir)

    results = []  # (name, status, seconds)
    total = len(args.detectors)
    for index, name in enumerate(args.detectors, 1):
        setup = DETECTOR_SETUP[name]
        print(f"\n\n{'#' * console.WIDTH}\n#  [{index}/{total}] {name}\n{'#' * console.WIDTH}",
              file=sys.stderr, flush=True)
        detector_started = time.monotonic()
        checkpoint = setup["checkpoint"]
        if checkpoint is not None and not checkpoint.exists():
            console.line(f"skipped: checkpoint not found at {checkpoint}")
            console.line("download it first (see METHOD_SOURCES.md), then rerun with "
                         f"--detectors {name}")
            results.append((name, "skipped (no checkpoint)", 0.0))
            continue

        cmd_args = ["--config", args.config, "--detectors", name,
                    "--output", str(output_dir / name)]
        if args.max_samples is not None:
            cmd_args += ["--max-samples", str(args.max_samples)]
        if args.device:
            cmd_args += ["--device", args.device]
        cmd_args += setup["extra_args"](args)

        try:
            python = ensure_python(name, setup["requirements"], log_dir)
        except subprocess.CalledProcessError:
            console.line(f"✗ install failed — see {log_dir / f'pip-{name}.log'}")
            results.append((name, "install failed", time.monotonic() - detector_started))
            continue

        env = {**os.environ, "PYTHONUNBUFFERED": "1"}
        code = subprocess.run([str(python), "main.py", *cmd_args], cwd=ROOT, env=env).returncode
        seconds = time.monotonic() - detector_started
        if code == 0:
            results.append((name, "ok", seconds))
        else:
            results.append((name, f"failed (exit {code}) — see {output_dir / name / 'run.log'}", seconds))

    ran = [name for name, status, _ in results if status == "ok"]
    produced = {}
    if ran and not args.skip_report:
        from scripts.generate_report import generate
        produced = generate(output_dir)

    console.header(f"Full run finished in {console.duration(time.monotonic() - started)}")
    for name, status, seconds in results:
        mark = "✓" if status == "ok" else "✗"
        console.line(f"{mark} {name:<14} {console.duration(seconds):>9}   {status}")
    if not ran:
        raise SystemExit("\nNo detector produced results; nothing to report.")
    console.line()
    console.kv("combined report", produced.get("report", "skipped (--skip-report)"))
    console.kv("per-detector", f"{output_dir}/<detector>/REPORT.md")


if __name__ == "__main__":
    main()
