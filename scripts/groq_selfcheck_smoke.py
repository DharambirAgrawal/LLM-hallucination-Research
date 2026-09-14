#!/usr/bin/env python3
"""Small real Groq + official SelfCheckGPT n-gram integration test."""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detectors.selfcheckgpt_detector import SelfCheckGPTDetector
from models.openai_compatible_model import OpenAICompatibleModel
from scripts.groq_smoke import BASE_URL, api_json, select_models
from utils.env_loader import load_env_file


CONTEXT = "Python was created by Guido van Rossum and first released in 1991."
QUESTION = "When was Python first released?"
CASES = (
    ("factual", "Python was first released in 1991."),
    ("hallucinated", "Python was created by James Gosling and released in 1985."),
)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=int, default=2, choices=(1, 2))
    parser.add_argument("--samples", type=int, default=2, choices=(2, 3))
    parser.add_argument("--output", default="results/groq-selfcheck-smoke.json")
    args = parser.parse_args()

    load_env_file(ROOT / ".env")
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise SystemExit("GROQ_API_KEY was not found in .env or the environment")

    available = {
        item["id"] for item in api_json("/models", api_key).get("data", [])
        if item.get("active", True)
    }
    selected = select_models(available, args.models)
    if len(selected) < args.models:
        raise SystemExit(f"Groq returned only {len(selected)} usable model(s)")

    report = {
        "kind": "official-selfcheckgpt-integration-smoke",
        "research_result": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "SelfCheckGPT official n-gram scorer",
        "models": selected,
        "samples_per_case": args.samples,
        "cases": [],
    }
    for model_id in selected:
        generator = OpenAICompatibleModel(
            name=f"groq:{model_id}",
            config={
                "model": model_id,
                "base_url": BASE_URL,
                "api_key_env": "GROQ_API_KEY",
                "temperature": 1.0,
                "max_tokens": 64,
                "timeout": 30,
            },
        )
        detector = SelfCheckGPTDetector(
            model=generator,
            method="ngram",
            n_samples=args.samples,
            temperature=1.0,
            threshold=3.0,
        )
        for label, answer in CASES:
            result = detector.detect(QUESTION, CONTEXT, answer)
            item = {
                "model": model_id,
                "case": label,
                "answer": answer,
                "score": result.score,
                "sentence_scores": result.sentence_scores,
                "threshold": detector.threshold,
                "flagged": result.is_hallucinated,
            }
            report["cases"].append(item)
            print(
                f"{model_id} | {label}: score={result.score:.4f}, "
                f"flagged={result.is_hallucinated}"
            )

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved integration report: {output.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
