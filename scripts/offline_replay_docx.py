"""Export real report blocks using python-docx in an existing separate runtime."""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("replay_document_export", ROOT / "reporting/document.py")
document = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = document
spec.loader.exec_module(document)

if __name__ == "__main__":
    source, destination = map(Path, sys.argv[1:])
    content = json.loads(source.read_text(encoding="utf-8"))
    report = document.Report(content["title"], content["subtitle"])
    for item in content["blocks"]:
        frame = pd.DataFrame(item["records"], columns=item["columns"]) if "records" in item else None
        report.blocks.append(document.Block(item["kind"], text=item["text"], items=item["items"],
                                             frame=frame, path=Path(item["path"]) if item.get("path") else None))
    report.to_docx(destination)
