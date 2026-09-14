#!/usr/bin/env python3
"""One-pair smoke tests for official upstream detector packages."""
from __future__ import annotations

import argparse

from models import ReplayModel


CONTEXT = "Python was created by Guido van Rossum and first released in 1991."
FACTUAL = "Python was first released in 1991."
HALLUCINATED = "Python was created by James Gosling and released in 1985."
QUESTION = "When was Python first released?"


def selfcheckgpt_smoke() -> None:
    from detectors import SelfCheckGPTDetector

    replay = ReplayModel([
        "Python was first released in 1991 by Guido van Rossum.",
        "Guido van Rossum released Python in 1991.",
        "Python's initial release was in 1991.",
    ])
    detector = SelfCheckGPTDetector(
        model=replay, method="ngram", n_samples=3, threshold=3.0
    )
    print("SelfCheckGPT official n-gram")
    print("  factual risk:", detector.detect(QUESTION, CONTEXT, FACTUAL).score)
    print("  hallucinated risk:", detector.detect(QUESTION, CONTEXT, HALLUCINATED).score)


def minicheck_smoke() -> None:
    import nltk

    from detectors import MiniCheckDetector

    # MiniCheck calls NLTK internally. New NLTK releases split the sentence
    # tables into `punkt_tab`; older releases may still look for `punkt`.
    for resource in ("punkt", "punkt_tab"):
        nltk.download(resource, quiet=True)
    detector = MiniCheckDetector(cache_dir="/content/minicheck-checkpoints")
    print("MiniCheck official Flan-T5-Large")
    print("  factual risk:", detector.detect(CONTEXT, FACTUAL).score)
    print("  hallucinated risk:", detector.detect(CONTEXT, HALLUCINATED).score)


def summac_smoke() -> None:
    from detectors import SummaCDetector

    detector = SummaCDetector(model_name="vitc")
    print("SummaC official Conv-VitC")
    print("  factual risk:", detector.detect(CONTEXT, FACTUAL).score)
    print("  hallucinated risk:", detector.detect(CONTEXT, HALLUCINATED).score)


def alignscore_smoke(checkpoint: str) -> None:
    from detectors import AlignScoreDetector

    detector = AlignScoreDetector(checkpoint_path=checkpoint)
    print("AlignScore official checkpoint")
    print("  factual risk:", detector.detect(CONTEXT, FACTUAL).score)
    print("  hallucinated risk:", detector.detect(CONTEXT, HALLUCINATED).score)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "detector",
        choices=("selfcheckgpt", "minicheck", "summac", "alignscore"),
    )
    parser.add_argument("--checkpoint", help="Required for AlignScore")
    args = parser.parse_args()
    if args.detector == "selfcheckgpt":
        selfcheckgpt_smoke()
    elif args.detector == "minicheck":
        minicheck_smoke()
    elif args.detector == "summac":
        summac_smoke()
    elif args.checkpoint:
        alignscore_smoke(args.checkpoint)
    else:
        parser.error("--checkpoint is required for AlignScore")


if __name__ == "__main__":
    main()
