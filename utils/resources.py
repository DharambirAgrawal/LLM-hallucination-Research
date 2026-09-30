"""Everything a run needs from outside the repository, fetched on demand.

`main.py` checks each required resource before it starts. Anything missing
is downloaded from its pinned official source with a progress bar and
verified against the SHA-256 recorded here, so a run never depends on a
separate manual download step. With `--dry-run` it only reports what is
missing and how much would be downloaded.

Remote files (pinned, checksummed):
  - HaluEval QA / dialogue / summarization: RUCAIBox/HaluEval at the commit
    in provenance/sources.yaml
  - RAGTruth responses + sources: ParticleMedia/RAGTruth at a fixed commit
  - HaluBench test split: PatronusAI/HaluBench at a fixed dataset revision
  - AlignScore-base checkpoint: the authors' Hugging Face repo yzha/AlignScore
    at a fixed revision
Package data: spaCy `en_core_web_sm` (SelfCheckGPT, AlignScore) and NLTK
`punkt`/`punkt_tab` (MiniCheck, SummaC, AlignScore).
Ollama models are pulled by models/model_factory.py.
"""
from __future__ import annotations

import hashlib
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.request import Request, urlopen

from utils import console

ROOT = Path(__file__).resolve().parent.parent

HALUEVAL_REVISION = "b7253db3cdaa0ab2c382f92b26b390109174f77e"
HALUEVAL_URL = f"https://raw.githubusercontent.com/RUCAIBox/HaluEval/{HALUEVAL_REVISION}/data"
ALIGNSCORE_REVISION = "8509e78d25bb914939fc585c626500c9b2944249"
RAGTRUTH_REVISION = "c103204b9ce28d6bbad859304bf30de72b8ed8fe"
RAGTRUTH_URL = f"https://raw.githubusercontent.com/ParticleMedia/RAGTruth/{RAGTRUTH_REVISION}/dataset"
HALUBENCH_REVISION = "5966a87929f51c204ab3cbef986b449495cc97b6"
SPACY_MODEL = "en_core_web_sm"
SPACY_MODEL_WHEEL = (
    "https://github.com/explosion/spacy-models/releases/download/"
    "en_core_web_sm-3.8.0/en_core_web_sm-3.8.0-py3-none-any.whl"
)


@dataclass(frozen=True)
class RemoteFile:
    label: str
    url: str
    sha256: str
    size: int


# Keyed by the repository-relative path that config.yaml refers to.
REMOTE_FILES = {
    "external_data/HaluEval/data/qa_data.json": RemoteFile(
        "HaluEval qa_data.json", f"{HALUEVAL_URL}/qa_data.json",
        "89ed139ec5e3a3169a0b30e45569ac1283846f76f27f7bb5e908ee6deed57e88", 6_164_420,
    ),
    "external_data/HaluEval/data/dialogue_data.json": RemoteFile(
        "HaluEval dialogue_data.json", f"{HALUEVAL_URL}/dialogue_data.json",
        "9c461df2691e4362837f66fceaaff3bc260453350c3a1cae76a5a52e5e338bfd", 6_984_266,
    ),
    "external_data/HaluEval/data/summarization_data.json": RemoteFile(
        "HaluEval summarization_data.json", f"{HALUEVAL_URL}/summarization_data.json",
        "86d2561eeb4271eb50b6bd90c9fd38636d30eaf4e98b648045f221adb1c2c758", 46_992_527,
    ),
    "external_data/RAGTruth/dataset/response.jsonl": RemoteFile(
        "RAGTruth response.jsonl", f"{RAGTRUTH_URL}/response.jsonl",
        "e4c2e4ac24fff676d8984cc61c35d791612fadc58015335d97dd632375e18073", 21_458_735,
    ),
    "external_data/RAGTruth/dataset/source_info.jsonl": RemoteFile(
        "RAGTruth source_info.jsonl", f"{RAGTRUTH_URL}/source_info.jsonl",
        "0dffc26ea9f3c1c3d7c7e8336b56ef1646e3cec876edffcca3c9c624d12d578b", 15_117_971,
    ),
    "external_data/HaluBench/test-00000-of-00001.parquet": RemoteFile(
        "HaluBench test parquet",
        f"https://huggingface.co/datasets/PatronusAI/HaluBench/resolve/{HALUBENCH_REVISION}/data/test-00000-of-00001.parquet",
        "c7e9cf966085ffae88d2947744418a05a26ea94380c35c238b9fc12ecb874cdc", 7_512_526,
    ),
    "external_models/summac/summac_conv_vitc_sent_perc_e.bin": RemoteFile(
        "SummaC-Conv vitc weights",
        "https://github.com/tingofurro/summac/raw/c1f3da93cd074c24d8033eb27a88b5a7cc5c08fa/summac_conv_vitc_sent_perc_e.bin",
        "cd880581f59bd47b8968c7f48d754cd617dfe9b256d8d76fdcbc0fa4a3b3c6fe", 1_811,
    ),
    "external_models/alignscore/AlignScore-base.ckpt": RemoteFile(
        "AlignScore-base checkpoint",
        f"https://huggingface.co/yzha/AlignScore/resolve/{ALIGNSCORE_REVISION}/AlignScore-base.ckpt",
        "6aedb637f0596ab29baef91e94466a57f032e02feea654978518919fe0981607", 1_966_965_771,
    ),
}


def _size(num: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if num < 1024 or unit == "GB":
            return f"{num:.0f} {unit}" if unit == "B" else f"{num:.1f} {unit}"
        num /= 1024
    return f"{num:.1f} GB"


def _key(path: str | Path) -> str:
    path = Path(path)
    if path.is_absolute():
        try:
            path = path.relative_to(ROOT)
        except ValueError:
            pass
    return path.as_posix()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(remote: RemoteFile, destination: Path) -> None:
    """Stream `remote` to `destination` with a progress bar; verify SHA-256
    before the file appears under its final name."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    digest = hashlib.sha256()
    request = Request(remote.url, headers={"User-Agent": "llm-hallucination-research/1.0"})
    try:
        with urlopen(request, timeout=60) as response, partial.open("wb") as handle:
            total = int(response.headers.get("Content-Length") or remote.size)
            with console.download_bar(remote.label, total) as bar:
                while chunk := response.read(1 << 20):
                    handle.write(chunk)
                    digest.update(chunk)
                    bar.update(len(chunk))
    except Exception:
        partial.unlink(missing_ok=True)
        raise
    if digest.hexdigest() != remote.sha256:
        partial.unlink(missing_ok=True)
        raise RuntimeError(
            f"{remote.label}: checksum mismatch (got {digest.hexdigest()[:16]}…, "
            f"expected {remote.sha256[:16]}…); the download was discarded"
        )
    partial.replace(destination)


def ensure_file(path: str | Path, dry_run: bool = False) -> Optional[int]:
    """Make sure a required file exists. Returns the bytes that were (or,
    in a dry run, would be) downloaded; None when it is already present."""
    target = Path(path) if Path(path).is_absolute() else ROOT / path
    remote = REMOTE_FILES.get(_key(path))
    label = remote.label if remote else _key(path)
    if target.is_file() and remote is None:
        console.line(f"✓ {label:<34} {_size(target.stat().st_size):>9}  present")
        return None
    if target.is_file() and target.stat().st_size == remote.size and _sha256(target) == remote.sha256:
        console.line(f"✓ {label:<34} {_size(target.stat().st_size):>9}  present, sha256 verified")
        return None
    if remote is None:
        raise SystemExit(
            f"\nRequired file {target} is missing and has no known official "
            "download. Put it there, or disable what uses it in the config."
        )
    if dry_run:
        console.line(f"↓ {label:<34} {_size(remote.size):>9}  will be downloaded")
        return remote.size
    if target.exists():
        console.line(f"↻ {label}: file does not match its recorded SHA-256, downloading again")
    download(remote, target)
    console.line(f"✓ {label:<34} {_size(remote.size):>9}  downloaded, sha256 verified")
    return remote.size


def ensure_spacy_model(dry_run: bool = False) -> None:
    try:
        import spacy.util
    except ImportError:
        return  # the detector's own import reports the missing package
    if spacy.util.is_package(SPACY_MODEL):
        console.line(f"✓ {'spaCy ' + SPACY_MODEL:<34} {'':>9}  present")
        return
    if dry_run:
        console.line(f"↓ {'spaCy ' + SPACY_MODEL:<34} {'13 MB':>9}  will be installed")
        return
    console.line(f"↓ spaCy {SPACY_MODEL}: installing pinned 3.8.0 wheel")
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", SPACY_MODEL_WHEEL],
        check=True,
    )
    console.line(f"✓ {'spaCy ' + SPACY_MODEL:<34} {'':>9}  installed")


def ensure_nltk_punkt(dry_run: bool = False) -> None:
    try:
        import nltk
    except ImportError:
        return
    for resource in ("punkt", "punkt_tab"):
        label = f"NLTK {resource}"
        try:
            nltk.data.find(f"tokenizers/{resource}")
            console.line(f"✓ {label:<34} {'':>9}  present")
        except LookupError:
            if dry_run:
                console.line(f"↓ {label:<34} {'':>9}  will be downloaded")
                continue
            nltk.download(resource, quiet=True)
            console.line(f"✓ {label:<34} {'':>9}  downloaded")


def prepare(config: dict, dry_run: bool = False) -> None:
    """Fetch every resource this configuration needs (see module docstring)."""
    detectors = config.get("detectors", {})
    enabled = {name for name, cfg in detectors.items()
               if isinstance(cfg, dict) and cfg.get("enabled")}
    pending = 0

    seen = set()
    for ds in config.get("datasets", []):
        if not ds.get("enabled", True):
            continue
        for key in ("path", "responses_path", "sources_path"):
            if ds.get(key) and ds.get("source") != "synthetic" and ds[key] not in seen:
                seen.add(ds[key])
                pending += ensure_file(ds[key], dry_run) or 0
    if "alignscore" in enabled:
        pending += ensure_file(detectors["alignscore"]["checkpoint_path"], dry_run) or 0
    if "summac" in enabled and detectors["summac"].get("model_name", "vitc") != "zs":
        pending += ensure_file(detectors["summac"].get(
            "conv_weights", "external_models/summac/summac_conv_vitc_sent_perc_e.bin"), dry_run) or 0
    if enabled & {"selfcheckgpt", "alignscore"}:
        ensure_spacy_model(dry_run)
    if enabled & {"minicheck", "summac", "alignscore"}:
        ensure_nltk_punkt(dry_run)
        console.line("· detector model weights (MiniCheck/SummaC) are fetched by their "
                     "own packages on first use")
    if dry_run and pending:
        console.line(f"total to download: {_size(pending)}")

