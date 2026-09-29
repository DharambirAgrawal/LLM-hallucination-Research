"""NLTK sentence-tokenizer data needed by upstream MiniCheck, SummaC, and
AlignScore (each calls ``nltk.tokenize.sent_tokenize`` internally).

pip installs NLTK but not its data files; without them every scored case
fails with ``LookupError: Resource punkt_tab not found``. Newer NLTK
releases read ``punkt_tab``; older ones read ``punkt``, so both are fetched.
"""
from __future__ import annotations

from loguru import logger


def ensure_sentence_tokenizer() -> None:
    try:
        import nltk
    except ImportError:
        return  # the detector's own import will report the missing package
    for resource in ("punkt", "punkt_tab"):
        try:
            nltk.data.find(f"tokenizers/{resource}")
        except LookupError:
            logger.info(f"Downloading NLTK '{resource}' sentence tokenizer (one time)")
            nltk.download(resource, quiet=True)
