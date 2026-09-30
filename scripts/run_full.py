#!/usr/bin/env python3
"""Orchestrate a "full run": every detector group in its own environment,
plus the reduction stage, collected into one output folder with one
combined report.

Environments (run in this order):
  core        requirements.txt: SelfCheckGPT + UQLM + the UQLM judge. They
              share the generator samples, and this environment also runs
              the reduction stage.
  minicheck   requirements/minicheck.txt   } each pins conflicting torch /
  summac      requirements/summac.txt      } transformers versions upstream,
  alignscore  requirements/alignscore.txt  } so each gets its own venv; each
              also scores core's reduction answers (--score-reduction-from).

Two phases. Phase 1 installs every environment, downloads everything, and
runs each environment's preflight (`main.py --preflight`: one real question
through the whole pipeline). Phase 2, the long runs, starts only if every
environment passed, so a broken install or missing model shows up in the
first minutes, not after hours of another environment's run.

Run this with the controller environment's Python (docs/HOW_TO_RUN.md §0.3).
core runs directly in that environment when requirements.txt is installed
there; every other environment gets its own `.venv-<name>`, created on first
use and reinstalled automatically if its requirements file changed or a
previous install did not finish. pip output goes to `<output>/logs/`.

The full run: exactly the `run:` block of config.yaml

    python scripts/run_full.py

A small smoke version of it (same outputs, less data):

    python scripts/run_full.py --runs 2 --max-samples 2 --n-samples 2 --max-iterations 1
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

# Environments, in run order. "core" holds the sampling-based detectors and
# the judge (they share the generator samples) and runs the reduction stage;
# the others each hold one detector with conflicting dependencies and score
# core's reduction answers too. Data, checkpoints and models are fetched by
# main.py; sizes come from config.yaml's `run:` block.
ENVIRONMENTS = {
    "core": {"requirements": "requirements.txt", "detectors": ("selfcheckgpt", "uqlm", "uqlm_judge")},
    "minicheck": {"requirements": "requirements/minicheck.txt", "detectors": ("minicheck",)},
    "summac": {"requirements": "requirements/summac.txt", "detectors": ("summac",)},
    "alignscore": {"requirements": "requirements/alignscore.txt", "detectors": ("alignscore",)},
}
ALL_DETECTORS = tuple(d for env in ENVIRONMENTS.values() for d in env["detectors"])
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
    parser.add_argument("--runs", type=int, help="Independent repeats (default: run.runs)")
    parser.add_argument("--max-samples", type=int,
                        help="Samples per dataset (default: run.samples_per_dataset)")
    parser.add_argument("--n-samples", type=int,
                        help="SelfCheckGPT samples per question (default: run.selfcheckgpt_samples)")
    parser.add_argument("--max-iterations", type=int,
                        help="Reduction rounds (default: run.reduction_iterations)")
    parser.add_argument("--no-reduce", action="store_true", help="Skip the reduction stage")
    parser.add_argument(
        "--device", choices=("cpu", "cuda"),
        help="Device for the torch-based detectors (passed to main.py --device)",
    )
    parser.add_argument(
        "--detectors", nargs="+", choices=ALL_DETECTORS,
        help="Detectors to run (default: run.detectors in the config)",
    )
    parser.add_argument(
        "--skip-report", action="store_true",
        help="Only run the environments; skip the top-level combined report",
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
    if name == "core" and all(importlib.util.find_spec(m) for m in ("selfcheckgpt", "uqlm")):
        console.line("environment      this interpreter (requirements.txt already installed)")
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

    import yaml
    plan = (yaml.safe_load((ROOT / args.config).read_text()) or {}).get("run", {})
    config = yaml.safe_load((ROOT / args.config).read_text()) or {}
    plan = config.get("run", {})
    selected = args.detectors or plan.get("detectors") or list(ALL_DETECTORS)
    envs = {name: [d for d in env["detectors"] if d in selected] for name, env in ENVIRONMENTS.items()}
    envs = {name: dets for name, dets in envs.items() if dets}
    args.detectors = list(envs)             # environments, in run order
    runs = args.runs if args.runs is not None else plan.get("runs", 1)
    reduce_on = not args.no_reduce and plan.get("reduce", False) and \
        any(d in envs.get("core", []) for d in ("selfcheckgpt", "uqlm"))
    console.header("Full run · each detector group in its own environment")
    console.kv("config", f"{args.config} (run plan below; flags override it)")
    console.kv("runs", f"{runs} per environment → <env>/run_01 … + combined/")
    for name, dets in envs.items():
        console.kv(f"env {name}", ", ".join(dets))
    console.kv("questions", f"{args.max_samples or plan.get('samples_per_dataset')} per dataset")
    console.kv("samples", f"{args.n_samples or plan.get('selfcheckgpt_samples')} per question per model")
    console.kv("reduction", "off" if not reduce_on else
               ", ".join(config.get("reduction", {}).get("methods", [])) + " vs. baseline (in core; "
               "every other environment also scores these answers)")
    console.kv("output", output_dir)

    def main_args(name: str) -> list[str]:
        cmd_args = ["--config", args.config, "--detectors", *envs[name],
                    "--output", str(output_dir / name)]
        if name != "core" and reduce_on:
            cmd_args += ["--score-reduction-from", str(output_dir / "core")]
        for flag, value in (("--runs", args.runs), ("--max-samples", args.max_samples),
                            ("--n-samples", args.n_samples), ("--max-iterations", args.max_iterations),
                            ("--device", args.device)):
            if value is not None:
                cmd_args += [flag, str(value)]
        if args.no_reduce:
            cmd_args.append("--no-reduce")
        return cmd_args

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    total = len(args.detectors)

    # Phase 1: install every environment, download everything, and run each
    # detector's preflight (one real case through the whole pipeline).
    # Nothing long starts unless every detector passes.
    console.header("Phase 1/2 · Prepare and check every detector before the long runs")
    pythons: dict[str, Path] = {}
    problems = []
    for index, name in enumerate(args.detectors, 1):
        console.section(f"[{index}/{total}] {name}")
        try:
            pythons[name] = ensure_python(name, ENVIRONMENTS[name]["requirements"], log_dir)
        except subprocess.CalledProcessError:
            problems.append((name, f"install failed — see {log_dir / f'pip-{name}.log'}"))
            continue
        preflight_args = [a for a in main_args(name)]
        if "--score-reduction-from" in preflight_args:   # its answers do not exist yet
            i = preflight_args.index("--score-reduction-from")
            del preflight_args[i:i + 2]
        code = subprocess.run(
            [str(pythons[name]), "main.py", *preflight_args, "--preflight"],
            cwd=ROOT, env=env,
        ).returncode
        if code != 0:
            problems.append((name, f"preflight failed — see {output_dir / name / 'run.log'}"))

    if problems:
        console.header("Stopped before the long runs")
        for name, reason in problems:
            console.line(f"✗ {name:<14} {reason}")
        console.line()
        console.line("Nothing long was started. Fix the items above and run the same command again;")
        console.line("installed environments and downloads are reused.")
        raise SystemExit(1)

    # Phase 2: the real runs. Everything was just verified.
    console.header("Phase 2/2 · Runs (all detectors passed their checks)")
    results = []  # (name, status, seconds)
    for index, name in enumerate(args.detectors, 1):
        print(f"\n\n{'#' * console.WIDTH}\n#  [{index}/{total}] {name}\n{'#' * console.WIDTH}",
              file=sys.stderr, flush=True)
        detector_started = time.monotonic()
        code = subprocess.run([str(pythons[name]), "main.py", *main_args(name)],
                              cwd=ROOT, env=env).returncode
        seconds = time.monotonic() - detector_started
        if code == 0:
            results.append((name, "ok", seconds))
        else:
            results.append((name, f"failed (exit {code}) — see {output_dir / name / 'run.log'}", seconds))

    ran = [name for name, status, _ in results if status == "ok"]
    produced = {}
    if ran and not args.skip_report:
        from reporting import generate
        console.header("Combined report · every detector, every run")
        produced = generate(output_dir, out_dir=output_dir / "combined",
                            title=f"{output_dir.name} · all detectors")

    console.header(f"Full run finished in {console.duration(time.monotonic() - started)}")
    for name, status, seconds in results:
        mark = "✓" if status == "ok" else "✗"
        console.line(f"{mark} {name:<14} {console.duration(seconds):>9}   {status}")
    if not ran:
        raise SystemExit("\nNo detector produced results; nothing to report.")
    console.line()
    console.line(f"{output_dir}/")
    for name in args.detectors:
        console.line(f"  {name}/run_01 … /combined   {', '.join(envs[name])}: each run + its own report")
    console.line("  combined/   all detectors, all runs: REPORT.md · report.html · report.docx · "
                 "takeaways.md   ← start here")
    if "report (docx)" in produced:
        console.kv("open", produced["report (docx)"])


if __name__ == "__main__":
    main()
