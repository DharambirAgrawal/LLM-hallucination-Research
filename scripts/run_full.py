#!/usr/bin/env python3
"""The full experiment: N independent runs with every detector and every
reduction method, then one combined result.

    python scripts/run_full.py                 # the config.yaml plan (run: block)
    python scripts/run_full.py --runs 5        # 5 independent runs
    python scripts/run_full.py --smoke         # quick end-to-end test (tiny numbers)

Result folder:

    results/<name>/
      run_01/ … run_N/   each a complete, independent run on the same data:
                         every detector, every reduction method, its own
                         report.docx / report.html / REPORT.md and data files
      combined/          all runs together: mean ± std, run-to-run
                         consistency, one report
      logs/              one log per detector group, pip install logs

Why detector groups: MiniCheck, SummaC and AlignScore pin conflicting
torch/transformers versions upstream, so each runs in its own Python
environment (created and reused automatically as `.venv-<name>`).
SelfCheckGPT, UQLM and the judge share the generator samples and run in the
"core" environment (requirements.txt), which also runs the reduction methods;
the other groups then score those reduction answers too. Each group writes
its part of every run into run_XX/<group>/, and this script builds each
run's report as soon as every group finishes that run, before starting the
next run. The combined report is written after the last run.

Two phases. Phase 1 installs every environment, downloads everything and
runs each group's preflight (one real question through the whole pipeline).
Phase 2, the long runs, starts only if every group passed.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from utils import console  # noqa: E402

# Detector groups, in run order. "python" lists interpreters to build the
# venv with when the current one is too new for the package's pins.
ENVIRONMENTS = {
    "core": {"requirements": "requirements.txt", "detectors": ("selfcheckgpt", "uqlm", "uqlm_judge")},
    "minicheck": {"requirements": "requirements/minicheck.txt", "detectors": ("minicheck",)},
    "summac": {"requirements": "requirements/summac.txt", "detectors": ("summac",)},
    # AlignScore pins torch<2, which has no wheels for Python ≥ 3.12.
    "alignscore": {"requirements": "requirements/alignscore.txt", "detectors": ("alignscore",),
                   "python": ("python3.11", "python3.10", "python3.9")},
}
ALL_DETECTORS = tuple(d for env in ENVIRONMENTS.values() for d in env["detectors"])
MARKER = ".requirements.sha256"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--runs", type=int, help="Independent runs (default: run.runs in config.yaml)")
    smoke_group = parser.add_mutually_exclusive_group()
    smoke_group.add_argument("--smoke", action="store_true",
                        help="Quick end-to-end test: 2 runs, 2 questions per dataset, 2 samples, "
                             "1 refine round (explicit flags still win)")
    smoke_group.add_argument("--smoke-2q", action="store_true",
                             help="Two questions total; 2 runs, all detectors and reducers")
    parser.add_argument("--output", help="Result folder (default: results/<run|smoke|smoke-2q>-<timestamp>)")
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--detectors", nargs="+", choices=ALL_DETECTORS,
                        help="Detectors to run (default: run.detectors in config.yaml)")
    parser.add_argument("--max-samples", type=int, help="Questions per dataset (default: config)")
    parser.add_argument("--n-samples", type=int, help="Samples per question per model (default: config)")
    parser.add_argument("--max-iterations", type=int, help="Self-Refine rounds (default: config)")
    parser.add_argument("--no-reduce", action="store_true", help="Skip the reduction methods")
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"),
                        help="Device for the torch-based detectors (default: auto)")
    parser.add_argument("--skip-report", action="store_true", help="Do not build the reports")
    parser.add_argument("--preflight", action="store_true",
                        help="Prepare/check all selected groups, print the total runtime estimate, then stop")
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


def base_python(name: str) -> str:
    """The interpreter to build `name`'s venv with."""
    wanted = ENVIRONMENTS[name].get("python")
    if not wanted:
        return sys.executable
    for candidate in wanted:
        found = shutil.which(candidate)
        if found:
            return found
    raise RuntimeError(
        f"{name} needs one of {', '.join(wanted)} (its pinned packages do not support "
        f"Python {sys.version_info.major}.{sys.version_info.minor}); install one, e.g. "
        f"`sudo apt install python3.11 python3.11-venv`, or leave {name} out of run.detectors"
    )


def ensure_python(name: str, requirements: str, log_dir: Path) -> Path:
    """Return the interpreter that runs `name`, installing its venv if needed."""
    if name == "core" and all(importlib.util.find_spec(m) for m in ("selfcheckgpt", "uqlm")):
        console.line("environment      this interpreter (requirements.txt already installed; "
                     "after a git pull run `pip install -r requirements.txt` again)")
        return Path(sys.executable)

    venv_dir = ROOT / f".venv-{name}"
    python = venv_dir / "bin" / "python"
    marker = venv_dir / MARKER
    wanted = requirements_hash(requirements)
    if python.exists() and marker.is_file() and marker.read_text().strip() == wanted:
        console.line(f"environment      {venv_dir.name} (up to date)")
        return python

    creator = base_python(name)
    log = log_dir / f"pip-{name}.log"
    reason = "first use" if not python.exists() else "requirements changed or last install incomplete"
    console.line(f"environment      installing {venv_dir.name} from {requirements} ({reason})")
    console.line(f"                 this can take several minutes · pip log: {log}")
    started = time.monotonic()
    with log.open("w", encoding="utf-8") as handle:
        if not python.exists():
            subprocess.run([creator, "-m", "venv", str(venv_dir)],
                           check=True, stdout=handle, stderr=subprocess.STDOUT)
        subprocess.run([str(python), "-m", "pip", "install", "--upgrade", "pip"],
                       check=True, stdout=handle, stderr=subprocess.STDOUT, cwd=ROOT)
        subprocess.run([str(python), "-m", "pip", "install", "-r", requirements],
                       check=True, stdout=handle, stderr=subprocess.STDOUT, cwd=ROOT)
    marker.write_text(wanted)
    console.line(f"                 installed in {console.duration(time.monotonic() - started)}")
    return python


def workload(config: dict, args: argparse.Namespace, runs: int, reduce_on: bool) -> tuple[int, int]:
    """Requested question slots and an upper bound on generator calls.

    This is shown before installing environments or downloading resources.
    Actual datasets can contain fewer questions; judge calls are additional.
    """
    default_questions = config.get("run", {}).get("samples_per_dataset", 50)
    questions = sum(
        args.max_samples if args.max_samples is not None else
        (2 if args.smoke else ds.get("max_samples", default_questions))
        for ds in config.get("datasets", []) if ds.get("enabled", True)
    )
    if not any(d in (args.detectors or config.get("run", {}).get("detectors") or ALL_DETECTORS)
               for d in ("selfcheckgpt", "uqlm")):
        return questions, 0
    models = len(config.get("selected_models") or config.get("models", []))
    samples = args.n_samples if args.n_samples is not None else (
        2 if args.smoke or args.smoke_2q else config.get("run", {}).get("selfcheckgpt_samples", 5))
    methods = config.get("reduction", {}).get("methods", []) if reduce_on else []
    iterations = args.max_iterations if args.max_iterations is not None else (
        1 if args.smoke or args.smoke_2q else config.get("run", {}).get("reduction_iterations", 3))
    reduction_calls = (1 + ("closed_book" in methods) + ("greedy" in methods)
                       + (2 * iterations if "self_refine_adapted" in methods else 0)
                       + (7 if "cove_adapted" in methods else 0)) if reduce_on else 0
    return questions, runs * questions * models * (samples + reduction_calls)


def main() -> None:
    args = parse_args()
    started = time.monotonic()

    import yaml
    config = yaml.safe_load((ROOT / args.config).read_text()) or {}
    if args.smoke_2q:
        if (args.max_samples is not None or args.detectors or args.no_reduce or
                args.runs not in (None, 2)):
            raise SystemExit("--smoke-2q fixes two questions, two runs, all detectors and reducers; "
                             "omit --max-samples, --detectors, and --no-reduce, "
                             "and do not override --runs")
        from utils.smoke import configure_two_question_smoke
        chosen = configure_two_question_smoke(config, ALL_DETECTORS)
    plan = config.get("run", {})
    if args.smoke or args.smoke_2q:
        args.runs = args.runs if args.runs is not None else 2
    runs = args.runs if args.runs is not None else plan.get("runs", 1)
    if runs < 1:
        raise SystemExit("--runs must be at least 1")
    selected = args.detectors or plan.get("detectors") or list(ALL_DETECTORS)
    groups = {name: [d for d in env["detectors"] if d in selected] for name, env in ENVIRONMENTS.items()}
    groups = {name: dets for name, dets in groups.items() if dets}
    reduce_on = not args.no_reduce and plan.get("reduce", False) and \
        any(d in groups.get("core", []) for d in ("selfcheckgpt", "uqlm"))

    # Absolute: main.py runs with cwd=ROOT.
    prefix = "smoke-2q" if args.smoke_2q else "smoke" if args.smoke else "run"
    stamp = f"{prefix}-{datetime.now():%Y%m%d-%H%M%S}"
    output_dir = Path(args.output).resolve() if args.output else ROOT / "results" / stamp
    old = sorted(output_dir.glob("run_[0-9]*")) if output_dir.is_dir() else []
    if old:
        raise SystemExit(f"{output_dir} already holds results ({', '.join(p.name for p in old)}). "
                         "Choose a new --output: mixing runs would make the combined result wrong.")
    log_dir = output_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    console.header("Full experiment" + (" · two-question smoke test" if args.smoke_2q
                                        else " · smoke test" if args.smoke else ""))
    console.kv("runs", f"{runs} independent runs on the same data → run_01 … run_{runs:02d} + combined/")
    console.kv("detectors", ", ".join(d for dets in groups.values() for d in dets))
    console.kv("reduction", ", ".join(config.get("reduction", {}).get("methods", [])) + " vs. baseline"
               if reduce_on else "off")
    console.kv("config", f"{args.config}" + (" + two-question smoke plan" if args.smoke_2q
                                                 else " + smoke numbers" if args.smoke else ""))
    if args.smoke_2q:
        console.kv("dataset", chosen)
    console.kv("output", output_dir)
    questions, generator_calls = workload(config, args, runs, reduce_on)
    console.kv("question slots", f"{questions} per run across "
               f"{sum(ds.get('enabled', True) for ds in config.get('datasets', []))} datasets")
    if generator_calls:
        console.kv("generator calls", f"up to {generator_calls:,} across all runs "
                   "(judge calls and preflight additional)")

    def main_args(name: str, run_index: int | None = None) -> list[str]:
        cmd = ["--config", args.config, "--detectors", *groups[name],
               "--output", str(output_dir), "--part", name, "--runs", str(runs)]
        if run_index is not None:
            cmd += ["--run-index", str(run_index)]
        if args.smoke_2q:
            cmd.append("--smoke-2q")
        elif args.smoke:
            cmd.append("--smoke")
        for flag, value in (("--max-samples", args.max_samples), ("--n-samples", args.n_samples),
                            ("--max-iterations", args.max_iterations), ("--device", args.device)):
            if value is not None:
                cmd += [flag, str(value)]
        if args.no_reduce:
            cmd.append("--no-reduce")
        return cmd

    env = {**os.environ, "PYTHONUNBUFFERED": "1"}
    total = len(groups)

    # Phase 1: install every environment, download everything, preflight.
    console.header("Phase 1/2 · Prepare and check every detector group before the long runs")
    pythons: dict[str, Path] = {}
    problems = []
    timings = {}
    for index, name in enumerate(groups, 1):
        console.section(f"[{index}/{total}] {name}: {', '.join(groups[name])}")
        try:
            pythons[name] = ensure_python(name, ENVIRONMENTS[name]["requirements"], log_dir)
        except (subprocess.CalledProcessError, RuntimeError) as exc:
            reason = (str(exc) if isinstance(exc, RuntimeError)
                      else f"install failed — see {log_dir / f'pip-{name}.log'}")
            console.line(f"✗ {reason}")
            problems.append((name, reason))
            continue
        timing_path = log_dir / f"preflight-timing-{name}.json"
        # An existing preflight-only folder may contain old timings. Accept
        # only a file written by the current successful worker.
        previous_timing = timing_path.read_bytes() if timing_path.is_file() else None
        timing_args = []
        if name != "core" and reduce_on:
            from utils.runtime_estimate import reduction_answers_per_question
            timing_args = ["--estimate-reduction-answers-per-question", str(reduction_answers_per_question(config))]
        code = subprocess.run([str(pythons[name]), "main.py", *main_args(name), "--preflight", *timing_args],
                              cwd=ROOT, env=env).returncode
        if code != 0:
            problems.append((name, f"preflight failed — see {log_dir / f'{name}.log'}"))
        elif timing_path.is_file() and timing_path.read_bytes() != previous_timing:
            try:
                timings[name] = json.loads(timing_path.read_text(encoding="utf-8"))
            except (ValueError, OSError) as exc:
                console.line(f"Timing estimate unavailable for {name}: {exc}")

    if problems:
        console.header("Stopped before the long runs")
        for name, reason in problems:
            console.line(f"✗ {name:<11} {reason}")
        console.line()
        console.line("Nothing long was started. Fix the items above and run the same command again;")
        console.line("installed environments and downloads are reused.")
        console.line("To share the errors for help: python scripts/share_logs.py --upload")
        raise SystemExit(1)

    if set(timings) == set(groups):
        from utils.runtime_estimate import combine_estimates, print_estimate, save_estimate
        try:
            timing = combine_estimates(timings, runs, elapsed_seconds=time.monotonic() - started)
            timing_path = output_dir / "runtime_estimate.json"
            save_estimate(timing_path, timing)
            print_estimate(timing, timing_path)
        except (ValueError, KeyError, TypeError, OSError) as exc:
            console.line(f"Total runtime estimate unavailable: {exc}")
    else:
        missing = sorted(set(groups) - set(timings))
        console.line(f"Total runtime estimate unavailable: missing current timing data for {', '.join(missing)}")
    if args.preflight:
        console.header("Preflight passed · measured runs were not started")
        return

    # Complete all groups for one run, publish its reports, then start the next.
    # Only one environment is loaded at a time. Each run recomputes every
    # detector's scores and draws fresh generator samples.
    console.header("Phase 2/2 · Runs (every detector group passed its checks)")
    results = []
    produced = {}
    report_failures = []
    if not args.skip_report:
        from reporting import generate
    for run_index in range(1, runs + 1):
        run_name = f"run_{run_index:02d}"
        run_dir = output_dir / run_name
        console.header(f"Run {run_index}/{runs} · every detector group → {run_dir}")
        core_ok = True
        run_ok = False
        failed_groups = []
        for index, name in enumerate(groups, 1):
            console.section(f"{run_name} · [{index}/{total}] {name}")
            # Phase 1 already checked this group; run only this repetition.
            cmd = [*main_args(name, run_index), "--skip-preflight"]
            if name != "core" and reduce_on and core_ok:
                cmd += ["--score-reduction-from", str(output_dir)]
            group_started = time.monotonic()
            code = subprocess.run([str(pythons[name]), "main.py", *cmd], cwd=ROOT, env=env).returncode
            seconds = time.monotonic() - group_started
            label = f"{run_name}/{name}"
            if code == 0:
                results.append((label, "ok", seconds))
                run_ok = True
            else:
                results.append((label, f"failed (exit {code}) — see {log_dir / f'{name}.log'}", seconds))
                failed_groups.append(name)
                if name == "core" and reduce_on:
                    core_ok = False
                    console.line("⚠ core failed in this run: the other groups run Stage A only "
                                 "(there are no complete reduction answers to score)")

        if not args.skip_report:
            console.section(f"{run_name} · Reports")
            if run_ok:
                try:
                    title = f"{output_dir.name} · {run_name}"
                    if failed_groups:
                        title += f" · incomplete (failed groups: {', '.join(failed_groups)})"
                    report_files = generate(run_dir, title=title)
                    if any(key not in report_files for key in ("report (docx)", "report (html)")):
                        raise RuntimeError("Word or HTML report was not produced")
                    console.line(f"✓ {run_name} reports ready" + (" (incomplete detector coverage)" if failed_groups else ""))
                    console.kv("Word", report_files["report (docx)"])
                    console.kv("HTML", report_files["report (html)"])
                except (Exception, SystemExit) as exc:
                    console.line(f"✗ {run_name}: report failed: {exc}")
                    report_failures.append(run_name)
            else:
                console.line(f"✗ {run_name}: no detector group succeeded; no report available")
                report_failures.append(run_name)

    # The combined report is generated only after the last run's reports.
    ran = [name for name, status, _ in results if status == "ok"]
    if ran and not args.skip_report:
        console.header("Combined report · all runs")
        try:
            title = f"{output_dir.name} · combined over {runs} runs"
            if any(status != "ok" for _, status, _ in results):
                title += " · incomplete detector coverage"
            produced = generate(output_dir, out_dir=output_dir / "combined",
                                title=title)
            if any(key not in produced for key in ("report (docx)", "report (html)")):
                raise RuntimeError("Word or HTML report was not produced")
            console.line("✓ combined/report.docx")
        except (Exception, SystemExit) as exc:
            console.line(f"✗ combined: report failed: {exc}")
            report_failures.append("combined")

    console.header(f"Finished in {console.duration(time.monotonic() - started)}")
    for name, status, seconds in results:
        console.line(f"{'✓' if status == 'ok' else '✗'} {name:<20} {console.duration(seconds):>9}   {status}")
    if any(status != "ok" for _, status, _ in results):
        console.line("To share the errors for help: python scripts/share_logs.py --upload")
    if not ran:
        raise SystemExit("\nNo detector group produced results; nothing to report.")
    console.line()
    console.line(f"{output_dir}/")
    for index in range(1, runs + 1):
        console.line(f"  run_{index:02d}/     run data" + (" · reports" if not args.skip_report else ""))
    if not args.skip_report:
        console.line("  combined/   all runs together: report.docx · report.html · takeaways.md   ← start here")
    console.line("  logs/       one log per detector group")
    if "report (docx)" in produced:
        console.kv("open", produced["report (docx)"])
    if any(status != "ok" for _, status, _ in results) or report_failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
