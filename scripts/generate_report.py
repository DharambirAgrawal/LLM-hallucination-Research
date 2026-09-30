#!/usr/bin/env python3
"""Rebuild the reports of an existing results folder.

    python scripts/generate_report.py --input results/full-run-X/selfcheckgpt/run_01
    python scripts/generate_report.py --input results/full-run-X --combined

main.py and scripts/run_full.py already call this for every run folder and
for the combined folder; use it only to regenerate reports afterwards.
See reporting/build.py for what is written.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from reporting import generate  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True, help="A run folder, or a folder of runs with --combined")
    parser.add_argument("--combined", action="store_true",
                        help="Combine every run folder under --input into <input>/combined")
    args = parser.parse_args()
    source = Path(args.input)
    for label, path in generate(source, source / "combined" if args.combined else None).items():
        print(f"  {label:<14} {path}")


if __name__ == "__main__":
    main()
