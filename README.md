# LLM Hallucination Research Harness

This repository validates hallucination detectors and later compares reduction
methods without presenting locally invented heuristics as published methods.
Detector algorithms come from pinned upstream packages; the local Python files
are thin adapters, data normalization, model connectivity, and metric reporting.

No LLM or checkpoint is stored or downloaded by this repository's default
workflow. Run `python main.py --dry-run` to check configuration and labeled data
without loading a detector or contacting a model server.

## Active official detectors

| Detector | Source used | Local role | Runtime |
|---|---|---|---|
| SelfCheckGPT | [official repository](https://github.com/potsawee/selfcheckgpt) | Calls official NLI, BERTScore, or n-gram scorer; obtains samples through the common model interface | N-gram Colab smoke or separate evaluator environment |
| MiniCheck | [official repository](https://github.com/Liyan06/MiniCheck) | Calls official sentence-level `MiniCheck.score` and converts support probability to risk | GPU Colab/evaluator; downloads official checkpoint there |
| SummaC | [official repository](https://github.com/tingofurro/summac) | Calls official SummaC-ZS or SummaC-Conv API | Separate compatible legacy ML environment |
| AlignScore | [official repository](https://github.com/yuh-zha/AlignScore) | Calls official `AlignScore.score` API | Separate environment plus explicit official checkpoint |

Exact Git revisions, licenses, status, and adapter paths are recorded in
[`provenance/sources.yaml`](provenance/sources.yaml). “Integrated” means the
adapter matches the upstream API; it does not mean runtime reproduction has
already been completed. Successful environments and checkpoint hashes must be
recorded before results are reported.

Semantic cosine similarity, token overlap, prompted LLM judging, the former
BERT-consistency heuristic, and the weighted ensemble are not active methods.
They were written locally and are excluded from the runner and configuration.

## Correct evaluation design

The harness first creates two fixed cases per dataset item: the known factual
answer (`label=0`) and known hallucinated answer (`label=1`). An official detector
scores those same cases. The output includes AUROC, average precision, accuracy,
precision, recall, and F1. Thresholds must be calibrated on validation data and
reported on a separate held-out test set.

Synthetic cases are only plumbing tests. Research claims should use official,
pinned data such as HaluEval, RAGTruth, TRUE, or LLM-AggreFact and should include
a human-reviewed subset.

## Quick checks

```bash
# On this computer: config/data only; no detector model and no API request
python main.py --dry-run
```

For Colab, copy the cells from [`docs/COLAB.md`](docs/COLAB.md). The lightest
official-package smoke test is:

```bash
pip install -r requirements-colab-smoke.txt
python official_smoke.py selfcheckgpt
```

The smoke script replays saved sample responses and therefore does not need an
LLM. It proves adapter/package connectivity only, not detector quality.

## Local model on another computer or an API

SelfCheckGPT needs repeated outputs from the model being checked. The same
adapter supports:

- Ollama at a remote `ollama.host`; or
- any server exposing an OpenAI-compatible `/chat/completions` endpoint.

Start from [`config.remote.example.yaml`](config.remote.example.yaml). Secrets
are read from the environment variable named by `api_key_env` and must never be
committed. MiniCheck, SummaC, and AlignScore score an existing context/answer
pair and do not need access to the generator.

## Reduction methods

There is intentionally no active reducer right now. The former “RAG,” restricted
sampling, self-verification, and Self-Refine-inspired code were local prompt or
sampling baselines—not official reproductions. Candidate verified approaches
and the reasons they cannot all be plugged into an arbitrary API model are in
[`docs/MITIGATION_METHODS.md`](docs/MITIGATION_METHODS.md).

The next defensible experiment is to run one upstream method in its supported
environment, preserve its original condition, and label any API/prompt port as
an adaptation. Detection and reduction results must remain separate.

## Repository map

```text
main.py                         fixed-pair validation CLI
official_smoke.py               one-pair upstream integration checks
config.yaml                     official detectors, all disabled by default
config.remote.example.yaml      remote/API model example
benchmark/runner.py             orchestration only
benchmark/detector_validation.py standard labeled metrics
detectors/                      thin official-package adapters
models/                         remote Ollama, OpenAI-compatible, and replay adapters
data/datasets.py                normalized fixed labeled cases
provenance/sources.yaml         immutable source register
docs/COLAB.md                   copy/paste Colab workflow
docs/REPRODUCIBILITY.md         research provenance policy
docs/MITIGATION_METHODS.md      reduction-method decision record
docs/PROFESSOR_EMAIL.md         email draft
```

See [`METHOD_SOURCES.md`](METHOD_SOURCES.md) for method-by-method research
status. Historical outputs from locally implemented methods must not be cited as
results of this official-source harness.
