#!/usr/bin/env python3
"""Fetch the official HaluEval data files at the commit pinned in
provenance/sources.yaml, straight from GitHub (no git clone needed — this
pulls only the 3 files this harness actually loads, not the whole repo).

Downloads, into external_data/HaluEval/data/:
    qa_data.json             (~6 MB)  - question, right/hallucinated answer
    dialogue_data.json       (~7 MB)  - knowledge, dialogue history, right/hallucinated response
    summarization_data.json  (~45 MB) - document, right/hallucinated summary

These are the three HaluEval task types with a matched right/hallucinated
pair, which is what this harness's paired detector-validation protocol needs
(see data/datasets.py:detection_cases). HaluEval's fourth file,
general_data.json, labels a single response hallucinated or not without a
matched pair — it does not fit this harness's schema and is not fetched here.

Run this once, on the data/execution machine (docs/REPRODUCIBILITY.md), then
enable `halueval_qa`, `halueval_dialogue`, and `halueval_summarization` in
config.yaml (already done if you pulled the config shipped with this script).
"""
from __future__ import annotations

import hashlib
import sys
import urllib.request
from pathlib import Path

REVISION = "b7253db3cdaa0ab2c382f92b26b390109174f77e"  # pinned in provenance/sources.yaml
BASE_URL = f"https://raw.githubusercontent.com/RUCAIBox/HaluEval/{REVISION}/data"
FILES = ["qa_data.json", "dialogue_data.json", "summarization_data.json"]

ROOT = Path(__file__).resolve().parent.parent
DEST = ROOT / "external_data" / "HaluEval" / "data"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def is_valid_jsonl(path: Path) -> bool:
    import json
    with path.open(encoding="utf-8") as handle:
        first_line = handle.readline()
    try:
        json.loads(first_line)
        return True
    except json.JSONDecodeError:
        return False


def main() -> None:
    DEST.mkdir(parents=True, exist_ok=True)
    print(f"Fetching HaluEval data at pinned revision {REVISION[:12]} into {DEST}")

    for filename in FILES:
        dest_path = DEST / filename
        if dest_path.exists():
            print(f"  = {filename} already present, skipping (delete it to re-fetch)")
            continue
        url = f"{BASE_URL}/{filename}"
        print(f"  > downloading {filename} ...")
        try:
            urllib.request.urlretrieve(url, dest_path)
        except Exception as exc:
            dest_path.unlink(missing_ok=True)
            print(f"  ✗ failed to fetch {filename}: {exc}", file=sys.stderr)
            sys.exit(1)

        if not is_valid_jsonl(dest_path):
            print(f"  ✗ {filename} does not look like valid JSON Lines after download", file=sys.stderr)
            sys.exit(1)

        size_mb = dest_path.stat().st_size / (1024 * 1024)
        print(f"  ✓ {filename} ({size_mb:.1f} MB) sha256={sha256(dest_path)}")

    print(
        "\nDone. `halueval_qa`, `halueval_dialogue`, and `halueval_summarization` "
        "are enabled in config.yaml and now point at real files.\n"
        "Record the sha256 values above alongside any reported run "
        "(docs/REPRODUCIBILITY.md: 'record a checksum of the input file')."
    )


if __name__ == "__main__":
    main()
