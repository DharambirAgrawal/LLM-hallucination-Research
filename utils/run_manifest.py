"""Everything needed to say exactly what produced a results folder.

docs/REPRODUCIBILITY.md asks every reported run to archive its
configuration, dependency versions, dataset checksums, model artifacts and
the Git SHA of this repository. `write_run_files` writes those next to the
results as run_manifest.json, config_used.yaml and environment.txt.
"""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path
from typing import Iterable

import yaml

ROOT = Path(__file__).resolve().parent.parent

# Packages whose version changes a score or a generation.
KEY_PACKAGES = (
    "selfcheckgpt", "uqlm", "minicheck", "summac", "alignscore", "torch",
    "transformers", "sentence-transformers", "bert-score", "spacy", "nltk",
    "ollama", "langchain-ollama", "openai", "numpy", "pandas", "scikit-learn",
)


# Hugging Face models the detectors download through their own packages.
# The packages load "main", so the revision each run actually used is read
# from the local Hugging Face cache and recorded (it cannot be pinned without
# changing upstream code).
HF_MODELS = (
    "roberta-large",                               # SelfCheckGPT BERTScore
    "potsawee/deberta-v3-large-mnli",              # SelfCheckGPT NLI
    "microsoft/deberta-large-mnli",                # UQLM NLI scorers / semantic entropy
    "sentence-transformers/all-MiniLM-L6-v2",      # UQLM cosine similarity
    "lytang/MiniCheck-Flan-T5-Large",              # MiniCheck
    "tals/albert-xlarge-vitaminc-mnli",            # SummaC vitc
)


def _hf_revisions() -> dict:
    import os
    hub = Path(os.environ.get("HF_HUB_CACHE") or
               Path(os.environ.get("HF_HOME", Path.home() / ".cache" / "huggingface")) / "hub")
    found = {}
    for repo in HF_MODELS:
        ref = hub / f"models--{repo.replace('/', '--')}" / "refs" / "main"
        if ref.is_file():
            found[repo] = ref.read_text().strip()
    return found


def _git(*args: str) -> str | None:
    try:
        return subprocess.run(
            ["git", *args], cwd=ROOT, capture_output=True, text=True, check=True
        ).stdout.strip()
    except Exception:
        return None


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def write_run_files(
    output_dir: Path,
    config: dict,
    argv: Iterable[str],
    datasets: dict,
    generators: list,
    started: datetime,
    stage_seconds: dict,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_entries = []
    for cfg in config.get("datasets", []):
        if not cfg.get("enabled", True) or cfg["name"] not in datasets:
            continue
        entry = {
            "name": cfg["name"],
            "source": cfg.get("source"),
            "n_samples": len(datasets[cfg["name"]]),
        }
        if cfg.get("path") and Path(cfg["path"]).is_file():
            entry["path"] = cfg["path"]
            entry["sha256"] = _sha256(Path(cfg["path"]))
        if cfg.get("hf_path"):
            entry["hf_path"] = cfg["hf_path"]
            entry["revision"] = cfg.get("revision")
        dataset_entries.append(entry)

    model_entries = [
        {
            "name": g.name,
            "provider": g.config.get("provider", "ollama"),
            "model": g.config.get("model"),
            "digest": getattr(g, "digest", None),
            "parameter_size": getattr(g, "parameter_size", None),
            "quantization": getattr(g, "quantization", None),
            "think": g.config.get("think"),
            "max_tokens": g.config.get("max_tokens"),
        }
        for g in generators
    ]

    finished = datetime.now(timezone.utc)
    manifest = {
        "started_at": started.isoformat(timespec="seconds"),
        "finished_at": finished.isoformat(timespec="seconds"),
        "duration_seconds": round((finished - started).total_seconds(), 1),
        "stage_seconds": {k: round(v, 1) for k, v in stage_seconds.items()},
        "run_protocol": {
            "detector_scores": "recomputed_each_run",
            "generator_samples": "fresh_each_run",
            "preflight_samples": "discarded",
            "shared_inputs": "fixed_questions_and_labeled_answers",
            "selfcheck_bertscore_short_samples": "retain_nonempty_sentences_if_upstream_filter_is_empty",
        },
        "command": " ".join(argv),
        "git": {
            "commit": _git("rev-parse", "HEAD"),
            "uncommitted_changes": bool(_git("status", "--porcelain")),
        },
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "packages": {name: _version(name) for name in KEY_PACKAGES if _version(name)},
        "huggingface_models": _hf_revisions(),
        "datasets": dataset_entries,
        "models": model_entries,
        "seed": config.get("benchmark", {}).get("seed"),
    }
    completeness = output_dir / "score_completeness.json"
    if completeness.is_file():
        manifest["score_completeness"] = json.loads(completeness.read_text(encoding="utf-8"))
    path = output_dir / "run_manifest.json"
    path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    (output_dir / "config_used.yaml").write_text(
        yaml.safe_dump(config, sort_keys=False), encoding="utf-8"
    )
    frozen = sorted(
        f"{dist.metadata['Name']}=={dist.version}"
        for dist in metadata.distributions()
        if dist.metadata["Name"]
    )
    (output_dir / "environment.txt").write_text("\n".join(frozen) + "\n", encoding="utf-8")
    return path
