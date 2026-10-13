"""Remove the model's commentary around a rewritten answer.

Asked to revise an answer, small models often wrap it: "Here is a revised
answer that addresses the feedback:\\n\\n<answer>\\n\\nThis response removes the
unsupported claim ...". The detectors would score that commentary as part of
the answer (in an earlier run, 47% of llama3.2-3b's self-refine answers had
it). Self-refine and CoVe therefore pass their final answer through
`strip_commentary`, which removes only:

  * a first line that introduces the answer and ends with a colon
    ("Here is the revised answer:", "Revised answer:", "Sure! Here's ...:")
  * a leading "Revised answer:" / "Final answer:" label on the answer line
  * trailing paragraphs that talk about the answer instead of answering
    ("This revised answer ...", "I removed ...")

Nothing in the middle of the answer is touched, and if removing these would
leave nothing, the text is returned unchanged. Whether anything was removed is
recorded per answer (details.commentary_removed in reduction_comparison.csv).
"""
from __future__ import annotations

import re

_INTRO = re.compile(
    r"^\s*(?:(?:sure|certainly|okay|ok)[!,.]?\s*)?"
    r"(?:here(?:'s| is| are)\b[^\n]*|(?:the |my )?(?:revised|rewritten|corrected|updated|final|improved)"
    r"(?: version of the)?(?: candidate)? (?:answer|response|version)[^\n]*)[:\-–]\s*$",
    re.IGNORECASE)
_LABEL = re.compile(r"^\s*\**(?:revised|rewritten|corrected|updated|final|improved) (?:answer|response)\**\s*:\s*",
                    re.IGNORECASE)
_COMMENT = re.compile(
    r"^\s*[\(\*]*(?:this (?:revised |rewritten |updated |new |corrected )?(?:answer|response|version)\b|"
    r"the (?:revised|rewritten|original) (?:answer|response)\b|"
    r"i (?:have |'ve )?(?:removed|revised|rewrote|rewritten|changed|kept|corrected|added|made|focused)\b|"
    r"i've\b|changes? made\b|explanation\s*:|in this (?:revised |rewritten )?(?:answer|response|version)\b|"
    r"by (?:removing|making|focusing)\b)",
    re.IGNORECASE)


def strip_commentary(text: str) -> tuple[str, bool]:
    """(cleaned answer, whether anything was removed)."""
    if not isinstance(text, str):
        return text, False
    original = text.strip()
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", original) if p.strip()]
    if not paragraphs:
        return text, False
    # leading introduction: its own paragraph, or the first line of the first one
    first_lines = paragraphs[0].splitlines()
    if _INTRO.match(first_lines[0]):
        rest = "\n".join(first_lines[1:]).strip()
        paragraphs = ([rest] if rest else []) + paragraphs[1:]
    if paragraphs:
        paragraphs[0] = _LABEL.sub("", paragraphs[0], count=1).strip()
    # trailing commentary paragraphs (never the only one left)
    while len(paragraphs) > 1 and _COMMENT.match(paragraphs[-1]):
        paragraphs.pop()
    cleaned = "\n\n".join(p for p in paragraphs if p).strip()
    if not cleaned or _squash(cleaned) == _squash(original):
        return original, False      # nothing removed: keep the text exactly as written
    return cleaned, True


def _squash(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()
