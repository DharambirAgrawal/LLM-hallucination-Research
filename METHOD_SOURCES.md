# Method sources and status

The machine-readable authority is [`provenance/sources.yaml`](provenance/sources.yaml).
A GitHub link alone is not enough for reproducibility: every executed method
needs a commit SHA, license, dependency environment, checkpoint identifier/hash,
configuration, and recorded adapter behavior.

## Detection

### SelfCheckGPT — official adapter

- Paper: [SelfCheckGPT](https://arxiv.org/abs/2303.08896)
- Code: [potsawee/selfcheckgpt](https://github.com/potsawee/selfcheckgpt), MIT
- Adapter: [`detectors/selfcheckgpt_detector.py`](detectors/selfcheckgpt_detector.py)
- Uses the upstream `SelfCheckNLI`, `SelfCheckBERTScore`, or `SelfCheckNgram`.
- The n-gram method returns an unbounded negative-log-probability score; it must
  not be treated as a probability. NLI/BERTScore require evaluator checkpoints.
- Samples may come from remote Ollama or an OpenAI-compatible endpoint. That
  provider substitution affects generation, not the upstream scoring function.

### MiniCheck — official adapter

- Paper: [MiniCheck](https://arxiv.org/abs/2404.10774)
- Code: [Liyan06/MiniCheck](https://github.com/Liyan06/MiniCheck), Apache-2.0
- Adapter: [`detectors/minicheck_detector.py`](detectors/minicheck_detector.py)
- Calls the upstream `MiniCheck.score(docs, claims)` API. The upstream guidance
  says multi-sentence responses should first be separated into sentences.
- The adapter reports `1 - minimum(sentence support probability)` as its declared
  response-level aggregation. This aggregation is local benchmark policy, not a
  new detector and not a value claimed by the MiniCheck paper.

### SummaC — official adapter

- Paper: [SummaC](https://arxiv.org/abs/2111.09525)
- Code: [tingofurro/summac](https://github.com/tingofurro/summac), Apache-2.0
- Adapter: [`detectors/summac_detector.py`](detectors/summac_detector.py)
- Calls the upstream `SummaCZS` or author-recommended `SummaCConv` and its
  `score([document], [claim])` API.
- Run separately because the upstream stack is older and should be reproduced
  with a documented compatible Torch/Transformers environment.

### AlignScore — official adapter

- Paper: [AlignScore](https://arxiv.org/abs/2305.07035)
- Code: [yuh-zha/AlignScore](https://github.com/yuh-zha/AlignScore), MIT
- Adapter: [`detectors/alignscore_detector.py`](detectors/alignscore_detector.py)
- Calls upstream `AlignScore(..., evaluation_mode="nli_sp").score(...)`.
- Requires the official checkpoint, spaCy sentence model, and a reproduction-
  compatible environment. The repository does not download these by default.

## Excluded local detection heuristics

The former token overlap, semantic cosine, prompted LLM judge, BERT stochastic
consistency, and ensemble implementations were removed. They must not appear
in results as official detectors; Git history is historical work only.

## Reduction

No reducer is active. See [`docs/MITIGATION_METHODS.md`](docs/MITIGATION_METHODS.md).

- [Self-Refine](https://github.com/madaan/self-refine) is official but task-
  specific. A generic QA prompt port is an adaptation, not a reproduction.
- [Self-RAG](https://github.com/AkariAsai/self-rag) is official but requires its
  trained checkpoint, reflection tokens, and retrieval/inference workflow.
- [RARR](https://github.com/anthonywchen/RARR) is relevant but is reference-only
  because no license is declared at its repository root.
- [AWS Bedrock contextual grounding](https://docs.aws.amazon.com/bedrock/latest/userguide/guardrails-contextual-grounding-check.html)
  can be a separately labeled managed-service baseline.

## Data and evaluation

- [HaluEval](https://github.com/RUCAIBox/HaluEval): fixed correct/hallucinated
  pairs; official repository pinned.
- [RAGTruth](https://github.com/ParticleMedia/RAGTruth): human response- and
  span-level annotations; planned.
- [TRUE](https://github.com/google-research/true): factual-consistency
  meta-evaluation resources; planned.
- [LLM-AggreFact](https://huggingface.co/datasets/lytang/LLM-AggreFact): useful
  official MiniCheck evaluation collection; dataset revision still needs to be
  pinned before use.

Synthetic cases in this repository are smoke fixtures, not evidence for a paper.
