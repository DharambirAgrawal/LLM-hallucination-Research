#!/usr/bin/env python3
"""Dependency-free Groq API smoke test; does not compute research scores."""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from models.openai_compatible_model import OpenAICompatibleModel
from utils.env_loader import load_env_file


BASE_URL = "https://api.groq.com/openai/v1"
PREFERRED_MODELS = (
    "llama-3.1-8b-instant",
    "openai/gpt-oss-20b",
    "qwen/qwen3-32b",
    "llama-3.3-70b-versatile",
)
PROMPT = (
    "Answer using only the context. If the answer is absent, say 'not provided'.\n\n"
    "Context: Python was created by Guido van Rossum and first released in 1991.\n"
    "Question: When was Python first released?\nAnswer:"
)


def api_json(path: str, api_key: str) -> dict:
    request = Request(
        f"{BASE_URL}{path}",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "User-Agent": "llm-hallucination-research/1.0",
        },
    )
    try:
        with urlopen(request, timeout=30) as response:
            return json.loads(response.read().decode("utf-8"))
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")[:500]
        raise RuntimeError(f"Groq HTTP {exc.code}: {detail}") from exc
    except (URLError, TimeoutError) as exc:
        raise RuntimeError(f"Groq connection failed: {exc}") from exc


def select_models(available: set[str], count: int) -> list[str]:
    selected = [model for model in PREFERRED_MODELS if model in available][:count]
    if len(selected) < count:
        selected.extend(sorted(available - set(selected))[: count - len(selected)])
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", type=int, default=2, choices=(1, 2))
    parser.add_argument("--samples", type=int, default=2, choices=(1, 2, 3))
    parser.add_argument("--output", default="results/groq-smoke.json")
    args = parser.parse_args()

    load_env_file(".env")
    api_key = os.environ.get("GROQ_API_KEY", "")
    if not api_key:
        raise SystemExit("GROQ_API_KEY was not found in the environment or .env")

    available = {
        item["id"] for item in api_json("/models", api_key).get("data", [])
        if item.get("active", True)
    }
    selected = select_models(available, args.models)
    if not selected:
        raise SystemExit("Groq returned no available models")

    report = {
        "kind": "api-smoke-not-research-evaluation",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "provider": "Groq OpenAI-compatible API",
        "models": selected,
        "samples_per_model": args.samples,
        "fixture": {"expected_year": "1991", "forbidden_year": "1985"},
        "results": [],
    }
    for model_id in selected:
        model = OpenAICompatibleModel(
            name=f"groq:{model_id}",
            config={
                "model": model_id,
                "base_url": BASE_URL,
                "api_key_env": "GROQ_API_KEY",
                "temperature": 0.7,
                "max_tokens": 64,
                "timeout": 30,
            },
        )
        for sample_index in range(args.samples):
            started = time.perf_counter()
            answer = model.generate(PROMPT)
            latency = time.perf_counter() - started
            fixture_pass = "1991" in answer and "1985" not in answer
            report["results"].append({
                "model": model_id,
                "sample": sample_index + 1,
                "latency_seconds": round(latency, 3),
                "fixture_pass": fixture_pass,
                "answer": answer,
            })
            print(
                f"{model_id} sample {sample_index + 1}: "
                f"fixture_pass={fixture_pass}, latency={latency:.2f}s"
            )

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    passed = sum(item["fixture_pass"] for item in report["results"])
    print(f"Smoke fixture: {passed}/{len(report['results'])} passed")
    print(f"Saved non-research smoke report: {output}")


if __name__ == "__main__":
    main()
