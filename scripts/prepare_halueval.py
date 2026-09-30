#!/usr/bin/env python3
"""Optional: pre-download the official HaluEval files without running
anything else (for example on a machine that will later run offline).

You do not need this before a normal run: main.py fetches every missing
file itself (utils/resources.py) at the commit pinned in
provenance/sources.yaml and verifies its sha256.

HaluEval's general_data.json is not fetched: it labels a single response
without a matched right/hallucinated pair, so it does not fit this
harness's paired-case schema (data/datasets.py:detection_cases).
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from utils import console  # noqa: E402
from utils.resources import HALUEVAL_REVISION, REMOTE_FILES, ensure_file  # noqa: E402


def main() -> None:
    console.section(f"HaluEval @ {HALUEVAL_REVISION[:12]}")
    for path in REMOTE_FILES:
        if path.startswith("external_data/HaluEval/"):
            ensure_file(path)


if __name__ == "__main__":
    main()
