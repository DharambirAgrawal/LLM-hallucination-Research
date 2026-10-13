"""Thin adapter around the official SelfCheckGPT package.

Upstream: https://github.com/potsawee/selfcheckgpt (commit pinned in
provenance/sources.yaml). Paper: Manakul et al., EMNLP 2023.

Every official scorer the package ships is available, each run on the SAME
sampled passages (detectors/sampling.py):

  ngram      SelfCheckNgram(n=1)           unigram negative log-probability
                                           (unbounded; not a probability)
  bertscore  SelfCheckBERTScore(rescale_with_baseline=True)
  nli        SelfCheckNLI                  DeBERTa-v3-large MNLI; the paper's
                                           strongest non-LLM variant
  prompt     SelfCheckAPIPrompt            the official LLM-prompt variant
                                           ("Is the sentence supported by the
                                           context above? Answer Yes or No."),
                                           with a local Ollama model as the
                                           judge through Ollama's OpenAI-
                                           compatible endpoint

The response is split into sentences with spaCy en_core_web_sm as in the
upstream README; the response score is the mean of the upstream sentence
scores (the paper's passage-level average). Values are not clipped. The
BERTScore short-sentence compatibility repair is documented below.

One call is routed, not changed: upstream SelfCheckBERTScore calls
``bert_score.score(...)`` once per sample, which reloads roberta-large from
disk every time and always puts it on the GPU when one exists (it passes no
device). With Ollama sharing the GPU that fails intermittently with "CUDA out
of memory". The module-level ``bert_score`` seen by upstream is replaced by
``_LoadedBERTScore``: the same library's ``BERTScorer`` with the same
arguments (lang, rescale_with_baseline; same model, layer, baseline file,
idf=False, batch size 64), loaded once on the configured device. Same numbers.

Upstream BERTScore drops sample sentences with <=3 spaCy tokens and crashes
when a sample has no longer sentences. For that sample only, retain its
nonempty short sentences and apply the same best-sentence F1 and sample-mean
formula. Other samples still use upstream unchanged. No samples are redrawn,
discarded, or replaced. This preprocessing repair is recorded in the report
and manifest; it is an adaptation on these edge cases, not exact upstream
preprocessing.

The prompt scorer's request is routed, not changed: upstream
SelfCheckAPIPrompt sends each (sample, sentence) prompt through its
``completion(prompt)`` method to an OpenAI client at temperature 0 with 5
output tokens. Here ``completion`` sends the same message with the same
settings to the local Ollama server's native API, adding the run's context
window (num_ctx; the OpenAI-compatible endpoint cannot set it, and a
different value makes Ollama reload the judge between requests). A prompt that
was already sent is answered from memory instead of again (temperature 0:
same prompt, same answer); identical sentences recur when a reduction method
keeps sentences of the baseline answer.
"""
from __future__ import annotations

import contextlib
import threading
from collections import OrderedDict
import io
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, TYPE_CHECKING

import numpy as np

from detectors.sampling import SampleBank

if TYPE_CHECKING:
    from models.base_model import BaseModel

METHODS = ("ngram", "bertscore", "nli", "prompt")


class _LoadedBERTScore:
    """Stands in for the ``bert_score`` module inside upstream
    SelfCheckBERTScore: ``score(...)`` with upstream's arguments, served by one
    ``bert_score.BERTScorer`` per setting, kept loaded on ``device``."""

    def __init__(self, device: Optional[str]):
        self.device = device
        self._scorers: Dict[tuple, object] = {}

    def score(self, cands, refs, lang=None, verbose=False, rescale_with_baseline=False):
        key = (lang, rescale_with_baseline)
        if key not in self._scorers:
            from bert_score import BERTScorer
            self._scorers[key] = BERTScorer(lang=lang, rescale_with_baseline=rescale_with_baseline,
                                            device=self.device)
        return self._scorers[key].score(cands, refs, verbose=verbose)


@dataclass
class SelfCheckGPTResult:
    score: float                                   # first configured method
    is_hallucinated: bool
    sentence_scores: List[float] = field(default_factory=list)
    scores: Dict[str, float] = field(default_factory=dict)
    errors: Dict[str, str] = field(default_factory=dict)   # scorer -> reason


class SelfCheckGPTDetector:
    """Run one or more official SelfCheckGPT scorers on shared samples."""

    def __init__(
        self,
        model: Optional["BaseModel"] = None,
        method: str | Sequence[str] = "ngram",
        n_samples: int = 5,
        temperature: float = 1.0,
        threshold: float = 0.5,
        device: str = "cpu",
        judge_model: Optional[str] = None,
        judge_host: str = "http://localhost:11434",
        bank: Optional[SampleBank] = None,
        thresholds: Optional[Dict[str, float]] = None,
        judge_num_ctx: Optional[int] = None,
        judge_parallel: int = 1,
    ):
        self.methods = [method] if isinstance(method, str) else list(method)
        unknown = set(self.methods) - set(METHODS)
        if unknown or not self.methods:
            raise ValueError(f"SelfCheckGPT methods must be among {METHODS}, got {self.methods}")
        if "prompt" in self.methods and not judge_model:
            raise ValueError("SelfCheckGPT 'prompt' needs a judge model (judge.model in config)")
        self.method = self.methods[0]
        self.model = model
        self.n_samples = n_samples
        self.temperature = temperature
        self.threshold = threshold
        self.thresholds = thresholds or {}
        self.device = device
        self.judge_model = judge_model
        self.judge_host = judge_host
        self.bank = bank or SampleBank(n_samples, temperature)
        self.judge_num_ctx = judge_num_ctx
        self.judge_parallel = max(1, int(judge_parallel))
        self._judge_lock = threading.Lock()
        self._scorers: Dict[str, object] = {}
        self._nlp = None
        self._bert = None                      # the _LoadedBERTScore in use (torch models live there)
        self._judge_cache: "OrderedDict[str, str]" = OrderedDict()
        self._judge_client = None

    TORCH_METHODS = ("bertscore", "nli")

    def release(self) -> None:
        """Drop the torch models (BERTScore, NLI) from memory; they are
        loaded again on next use. The n-gram scorer and the prompt scorer
        (and its memory of judge answers) stay."""
        for method in self.TORCH_METHODS:
            self._scorers.pop(method, None)
        if self._bert is not None:
            # upstream's module also points at this object: empty it, so the
            # BERTScore model is really freed
            self._bert._scorers.clear()
        self._bert = None

    def _load(self):
        if all(method in self._scorers for method in self.methods):
            return
        try:
            import selfcheckgpt.modeling_selfcheck as upstream
            from selfcheckgpt.modeling_selfcheck import (
                SelfCheckBERTScore,
                SelfCheckNLI,
                SelfCheckNgram,
            )
        except ImportError as exc:
            raise RuntimeError(
                "SelfCheckGPT is not installed: pip install -r requirements.txt"
            ) from exc
        # Upstream prints status lines on construction; keep them off screen.
        with contextlib.redirect_stdout(io.StringIO()):
            for method in self.methods:
                if method in self._scorers:
                    continue
                if method == "ngram":
                    self._scorers[method] = SelfCheckNgram(n=1)
                elif method == "bertscore":
                    self._scorers[method] = SelfCheckBERTScore(rescale_with_baseline=True)
                    self._bert = _LoadedBERTScore(self.device)
                    upstream.bert_score = self._bert   # see module docstring
                elif method == "nli":
                    self._scorers[method] = SelfCheckNLI(device=self.device)
                elif method == "prompt":
                    from selfcheckgpt.modeling_selfcheck_apiprompt import SelfCheckAPIPrompt
                    # The official class builds `OpenAI()` from the environment;
                    # point it at the local Ollama server's OpenAI-compatible API.
                    os.environ["OPENAI_BASE_URL"] = f"{self.judge_host.rstrip('/')}/v1"
                    os.environ.setdefault("OPENAI_API_KEY", "ollama")
                    scorer = SelfCheckAPIPrompt(client_type="openai", model=self.judge_model)
                    scorer.completion = self._judge_completion   # see module docstring
                    self._scorers[method] = scorer

    JUDGE_CACHE_SIZE = 200_000

    def _judge_completion(self, prompt: str) -> str:
        """Upstream SelfCheckAPIPrompt.completion's request (temperature 0,
        5 output tokens, the prompt as one user message) to Ollama's native
        chat API with this run's num_ctx; repeated prompts come from memory."""
        with self._judge_lock:
            cached = self._judge_cache.get(prompt)
            if cached is not None:
                self._judge_cache.move_to_end(prompt)
                return cached
        text = self._judge_request(prompt)
        with self._judge_lock:
            self._judge_cache[prompt] = text
            if len(self._judge_cache) > self.JUDGE_CACHE_SIZE:
                self._judge_cache.popitem(last=False)
        return text

    def _judge_request(self, prompt: str) -> str:
        if self._judge_client is None:
            import ollama
            self._judge_client = ollama.Client(host=self.judge_host, timeout=600)
        options = {"temperature": 0.0, "num_predict": 5}
        if self.judge_num_ctx:
            options["num_ctx"] = int(self.judge_num_ctx)
        response = self._judge_client.chat(model=self.judge_model,
                                           messages=[{"role": "user", "content": prompt}],
                                           options=options, stream=False)
        return response["message"]["content"] or ""

    def _prefetch_judge(self, scorer, sentences: List[str], samples: List[str]) -> None:
        """Send the prompts upstream predict() is about to send, in an order
        that is fast for Ollama, so its sequential completion() calls find the
        answers in memory.

        Upstream loops sentence by sentence over the samples, and its prompt
        starts with the sample ("Context: {sample}\\n\\nSentence: ..."), so
        every request starts with different text. Grouped by sample, the
        requests of one group share that long beginning, which Ollama reuses
        from the previous request instead of reading it again. Up to
        `judge_parallel` groups run at once (one per Ollama slot). Prompts are
        built exactly as upstream builds them (newlines in a sample become
        spaces); one built differently would just be sent again by upstream,
        so this can only save time, never change a score."""
        template = getattr(scorer, "prompt_template", None)
        if not template:
            return
        groups = []
        with self._judge_lock:
            for sample in dict.fromkeys(samples):
                context = sample.replace("\n", " ")
                group = [p for p in dict.fromkeys(template.format(context=context, sentence=sentence)
                                                  for sentence in sentences)
                         if p not in self._judge_cache]
                if group:
                    groups.append(group)
        if not groups:
            return

        def send(group):
            for prompt in group:
                self._judge_completion(prompt)
        if self.judge_parallel < 2 or len(groups) < 2:
            for group in groups:
                send(group)
            return
        from concurrent.futures import ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=self.judge_parallel) as pool:
            list(pool.map(send, groups))

    def _sentences(self, answer: str) -> List[str]:
        """Split the response the way the upstream README does:
        ``[sent.text.strip() for sent in nlp(passage).sents]`` with spaCy
        ``en_core_web_sm``. The n-gram scorer builds its vocabulary with that
        same spaCy segmentation, so a different splitter can produce tokens
        it never counted, which score -log(0) = inf. Falls back to a regex
        only when spaCy or the model is missing (offline tests)."""
        if self._nlp is None:
            try:
                import spacy
                self._nlp = spacy.load("en_core_web_sm")
            except Exception:
                self._nlp = False
        if self._nlp:
            parts = [sent.text.strip() for sent in self._nlp(answer.strip()).sents]
        else:
            parts = re.split(r"(?<=[.!?])\s+", answer.strip())
        return [part.strip() for part in parts if part.strip()]

    def _score_one(self, method: str, sentences: List[str], answer: str, samples: List[str]) -> List[float]:
        scorer = self._scorers[method]
        with contextlib.redirect_stdout(io.StringIO()):
            if method == "ngram":
                result = scorer.predict(sentences=sentences, passage=answer, sampled_passages=samples)
                values = result["sent_level"]["avg_neg_logprob"]
            elif method == "prompt":
                self._prefetch_judge(scorer, sentences, samples)
                values = scorer.predict(sentences=sentences, sampled_passages=samples, verbose=False)
            elif method == "bertscore":
                values = self._bertscore(sentences, samples)
            else:
                values = scorer.predict(sentences=sentences, sampled_passages=samples)
        return [float(v) for v in np.asarray(values).reshape(-1)]

    def _bertscore(self, sentences: List[str], samples: List[str]):
        """Keep upstream's result except for its empty short-sentence filter."""
        if not samples or not sentences:
            raise ValueError("BERTScore requires sentences and evidence samples")
        scorer = self._scorers["bertscore"]
        # Test doubles without spaCy continue through the public upstream API.
        if not hasattr(scorer, "nlp"):
            return scorer.predict(sentences=sentences, sampled_passages=samples)
        normal, short = [], []
        for sample in samples:
            spans = list(scorer.nlp(sample).sents)
            if any(len(span) > 3 for span in spans):
                normal.append(sample)
            else:
                texts = [span.text.strip() for span in spans if span.text.strip()]
                if not texts:
                    raise ValueError("BERTScore sample contains no nonempty sentences")
                short.append(texts)
        if not short:
            return scorer.predict(sentences=sentences, sampled_passages=samples)
        total = np.zeros(len(sentences), dtype=float)
        if normal:
            total += np.asarray(scorer.predict(sentences=sentences, sampled_passages=normal)) * len(normal)
        import selfcheckgpt.modeling_selfcheck as upstream
        for texts in short:
            refs = [sentence for sentence in sentences for _ in texts]
            cands = texts * len(sentences)
            _, _, f1 = upstream.bert_score.score(
                cands, refs, lang=scorer.default_model, verbose=False,
                rescale_with_baseline=scorer.rescale_with_baseline)
            # Same best-match and 1-F1 calculation as upstream predict().
            best = f1.reshape(len(sentences), len(texts)).max(axis=1).values.numpy()
            total += 1.0 - best
        return total / len(samples)

    def detect(
        self,
        question: str,
        context: str,
        answer: str,
        model: Optional["BaseModel"] = None,
        methods: Optional[Sequence[str]] = None,
    ) -> SelfCheckGPTResult:
        """Score one answer with every configured method, or with `methods`
        only (the run scores the torch methods and the prompt method in
        separate phases). `model` overrides the constructor's generator for
        this call only."""
        self._load()
        if not answer.strip():
            # No neutral score exists: n-gram scores are unbounded, so a
            # placeholder such as 1.0 would read as "very factual". Record
            # the case as a failure instead.
            raise ValueError("empty answer cannot be scored")
        generator = model or self.model
        if generator is None:
            raise ValueError(
                "SelfCheckGPT needs a generator model, either bound at "
                "construction or passed to detect(model=...)"
            )
        samples = self.bank.get(generator, question, context)
        sentences = self._sentences(answer)
        # Preserve other scores if one scorer fails; completeness checks
        # reject the run, with the original per-scorer reason retained.
        scores, errors, first_values = {}, {}, []
        for method in (self.methods if methods is None else [m for m in self.methods if m in methods]):
            try:
                values = self._score_one(method, sentences, answer, samples)
                if not values or not np.isfinite(values).all():
                    raise ValueError(f"non-finite score: {values}")
            except Exception as exc:
                errors[method] = f"{type(exc).__name__}: {str(exc).strip() or repr(exc)}"
                continue
            scores[method] = float(np.mean(values))
            if method == self.method:
                first_values = values
        if not scores:
            raise RuntimeError("; ".join(f"{m}: {e}" for m, e in errors.items()))
        score = scores.get(self.method, next(iter(scores.values())))
        return SelfCheckGPTResult(
            score=score,
            is_hallucinated=score >= self.thresholds.get(self.method, self.threshold),
            sentence_scores=first_values,
            scores=scores,
            errors=errors,
        )

    def export_samples(self, path: Path) -> int:
        return self.bank.export(path)
