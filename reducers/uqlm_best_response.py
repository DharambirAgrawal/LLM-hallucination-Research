"""Best-response selection with UQLM's official semantic-entropy scorer.

Upstream: https://github.com/cvs-health/uqlm (SemanticEntropy with
use_best=True), v0.6.6. Candidates are the grounded baseline answer plus the
model's samples for the same prompt, so no extra generation is needed.
Official implementation; nothing is reimplemented.

What UQLM does (uqlm/nli/cluster.py, verified): answers are grouped with an
NLI model into meaning clusters (UQLM's own clustering rule, adapted from
semantic entropy, Farquhar et al., Nature 2024); the "best" answer is the
first member of the most probable cluster, where members are ordered with the
most repeated exact string first and otherwise by LENGTH, longest first. The
method therefore has a length bias whenever no answer repeats verbatim; the
report states this.

prompts_in_nli: we pass no prompts, so UQLM compares the answers alone (its
code uses the prompt only when `prompts_in_nli and prompts`). Passing our
long grounded prompts made every answer fall into one cluster in a check on
uqlm 0.6.6, so "best" was just the first candidate; prompts_in_nli=False is
set explicitly to make that choice visible.

Scoring: an answer picked from the samples is scored leave-one-out (never
against evidence that contains itself); see benchmark/reduction_runner.py.
"""
from __future__ import annotations

import contextlib
import io
from typing import List, Optional


class UQLMBestResponseReducer:
    METHOD_ID = "uqlm_best_response"
    REPRODUCTION_STATUS = "official_uqlm_implementation"
    SOURCE_PAPER = "https://arxiv.org/abs/2507.06196"

    def __init__(self, device: Optional[str] = None, nli=None):
        self.device = device
        self.nli = nli          # reuse an already loaded UQLM NLI model (same weights)
        self._se = None

    def _load(self):
        if self._se is None:
            try:
                from uqlm import SemanticEntropy
            except ImportError as exc:
                raise RuntimeError("UQLM is not installed: pip install -r requirements.txt") from exc
            with contextlib.redirect_stdout(io.StringIO()):
                self._se = SemanticEntropy(device=self.device, use_best=True, prompts_in_nli=False,
                                           nli=self.nli)

    def select(self, baseline: str, samples: List[str]) -> str:
        self._load()
        with contextlib.redirect_stdout(io.StringIO()):
            result = self._se.score(responses=[baseline], sampled_responses=[samples],
                                    show_progress_bars=False)
        best = result.data["responses"][0]
        if not isinstance(best, str) or not best.strip():
            raise ValueError("UQLM returned no best response")
        return best
