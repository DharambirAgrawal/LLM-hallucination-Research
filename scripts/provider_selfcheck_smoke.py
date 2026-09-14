#!/usr/bin/env python3
"""Tiny free-tier API + official SelfCheckGPT n-gram integration test.

This is deliberately a plumbing test, not a benchmark. It makes four short
completion requests per provider (two samples for each of two fixed cases).
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from detectors.selfcheckgpt_detector import SelfCheckGPTDetector
from models.openai_compatible_model import OpenAICompatibleModel
from utils.env_loader import load_env_file


CONTEXT = "Python was created by Guido van Rossum and first released in 1991."
QUESTION = "When was Python first released, and who created it?"
CASES = (
    ("factual", "Python was created by Guido van Rossum and first released in 1991."),
    ("hallucinated", "Python was created by James Gosling and first released in 1985."),
)

PROVIDERS = {
    "openrouter": {
        "key_env": "OPENROUTER_API_KEY",
        "base_url": "https://openrouter.ai/api/v1",
        "model": "openrouter/free",
        "extra_headers": {
            "HTTP-Referer": "https://github.com/DharambirAgrawal/LLM-hallucination-Research",
            "X-Title": "LLM Hallucination Research",
        },
        "cost_note": "OpenRouter free-model router; provider rate limits apply.",
    },
    "gemini": {
        "key_env": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "model": "gemini-3.5-flash-lite",
        "extra_headers": {},
        "cost_note": "Gemini free tier when enabled for this API project; quota applies.",
    },
    "gemini-flash": {
        "key_env": "GEMINI_API_KEY",
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "model": "gemini-3.5-flash",
        "extra_headers": {},
        "request_fields": {"reasoning_effort": "minimal"},
        "max_tokens": 96,
        "cost_note": "Gemini free tier when enabled for this API project; quota applies.",
    },
    "mistral": {
        "key_env": "MISTRAL_API_KEY",
        "base_url": "https://api.mistral.ai/v1",
        "model": "mistral-small-latest",
        "extra_headers": {},
        "cost_note": "Mistral Studio free mode when enabled for this account; limits apply.",
    },
}

DEFAULT_PROVIDERS = "gemini,gemini-flash"


class RecordingModel(OpenAICompatibleModel):
    """Record bounded call metadata and samples without recording credentials."""

    def __init__(self, name: str, config: dict):
        super().__init__(name, config)
        self.calls: list[dict] = []

    def generate(self, prompt: str, **kwargs) -> str:
        started = time.monotonic()
        text = super().generate(prompt, **kwargs)
        self.calls.append(
            {
                "latency_seconds": round(time.monotonic() - started, 3),
                "response": text,
            }
        )
        return text


def parse_providers(value: str) -> list[str]:
    selected = [item.strip().lower() for item in value.split(",") if item.strip()]
    unknown = sorted(set(selected) - set(PROVIDERS))
    if unknown:
        raise argparse.ArgumentTypeError(f"unknown provider(s): {', '.join(unknown)}")
    if not selected:
        raise argparse.ArgumentTypeError("select at least one provider")
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--providers",
        type=parse_providers,
        default=parse_providers(DEFAULT_PROVIDERS),
        help=f"comma-separated providers (default: {DEFAULT_PROVIDERS})",
    )
    parser.add_argument("--samples", type=int, default=2, choices=(2, 3))
    parser.add_argument("--output", default="results/provider-selfcheck-smoke.json")
    args = parser.parse_args()

    load_env_file(ROOT / ".env")
    report = {
        "kind": "official-selfcheckgpt-free-tier-integration-smoke",
        "research_result": False,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "method": "SelfCheckGPT official n-gram scorer",
        "method_revision": "19b492a2a380931bf1ed0ca94a9565c9aa7b03e1",
        "samples_per_case": args.samples,
        "maximum_completion_calls": len(args.providers) * len(CASES) * args.samples,
        "providers": [],
    }

    for provider_name in args.providers:
        provider = PROVIDERS[provider_name]
        provider_result = {
            "provider": provider_name,
            "requested_model": provider["model"],
            "cost_note": provider["cost_note"],
            "status": "pending",
            "cases": [],
        }
        report["providers"].append(provider_result)
        if not os.environ.get(provider["key_env"]):
            provider_result.update(
                status="skipped",
                error=f"{provider['key_env']} is not set",
            )
            print(f"{provider_name}: skipped (missing {provider['key_env']})")
            continue

        generator = RecordingModel(
            name=f"{provider_name}:{provider['model']}",
            config={
                "model": provider["model"],
                "base_url": provider["base_url"],
                "api_key_env": provider["key_env"],
                "extra_headers": provider["extra_headers"],
                "request_fields": provider.get("request_fields", {}),
                "temperature": 1.0,
                "max_tokens": provider.get("max_tokens", 48),
                "timeout": 45,
            },
        )
        detector = SelfCheckGPTDetector(
            model=generator,
            method="ngram",
            n_samples=args.samples,
            temperature=1.0,
            threshold=3.0,
        )
        try:
            for label, answer in CASES:
                first_call = len(generator.calls)
                result = detector.detect(QUESTION, CONTEXT, answer)
                provider_result["cases"].append(
                    {
                        "case": label,
                        "answer": answer,
                        "score": result.score,
                        "sentence_scores": result.sentence_scores,
                        "uncalibrated_smoke_threshold": detector.threshold,
                        "smoke_flag_only": result.is_hallucinated,
                        "samples": generator.calls[first_call:],
                    }
                )
                print(
                    f"{provider_name} | {label}: score={result.score:.4f}, "
                    f"smoke_flag={result.is_hallucinated}"
                )
            factual, hallucinated = provider_result["cases"]
            provider_result["expected_score_direction"] = (
                hallucinated["score"] > factual["score"]
            )
            provider_result["completion_calls"] = len(generator.calls)
            provider_result["status"] = "passed"
        except Exception as exc:
            provider_result["status"] = "failed"
            provider_result["completion_calls"] = len(generator.calls)
            provider_result["error"] = str(exc)
            print(f"{provider_name}: failed: {exc}")

    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"Saved integration report: {output.relative_to(ROOT)}")

    if not any(item["status"] == "passed" for item in report["providers"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
