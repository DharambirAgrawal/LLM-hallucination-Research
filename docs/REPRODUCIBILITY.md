# Reproducibility and Upstream-Code Policy

This repository is a benchmark harness. Published algorithms remain owned and
maintained by their upstream projects. The harness should contain only adapters,
experiment orchestration, result schemas, and the proposed research method.

## What "verified upstream" means

A method may be labeled `upstream` only when all of the following are recorded:

1. the paper and an author-associated or institution-associated repository;
2. the upstream license;
3. a full Git commit SHA or immutable package artifact;
4. exact model and tokenizer artifact revisions;
5. the official example or evaluation reproduced without algorithm changes;
6. every local adapter and behavior-changing deviation;
7. prompts, sampling settings, seeds, dataset revision, and runtime environment.

The machine-readable source register is
[`provenance/sources.yaml`](../provenance/sources.yaml). A moving branch such as
`main`, an unbounded requirement such as `package>=1.0`, or an unversioned model
name is not sufficient for a reported research run.

## Integration order

Use the least invasive integration available:

1. official released package pinned in an environment lock file;
2. official Git package pinned to a full commit SHA;
3. read-only Git submodule under `third_party/` when no package interface exists;
4. a local adapter around the upstream public API.

Do not copy individual upstream source files into `detectors/` or `reducers/`.
Do not edit a submodule in place. If a patch is unavoidable, store it under
`patches/<method>/`, explain it in the source register, and report the condition
as a modified reproduction.

Model weights, dataset caches, API credentials, and generated responses are not
committed. They belong on the execution machine or in an external artifact store.

## Experiment separation

### Stage A: detector validation

Use fixed, labeled factual and hallucinated responses. HaluEval, RAGTruth, TRUE,
or LLM-AggreFact are suitable depending on the task. Tune thresholds on a
validation split and report AUROC, AUPRC, F1, precision, recall, and confusion
matrices on a held-out test split. Never call an unlabeled acceptance rate
"accuracy."

### Stage B: reduction evaluation

Generate paired baseline and mitigated answers for the same model, sample, and
seed. Record failures instead of converting them into empty answers. Measure at
least:

- groundedness/faithfulness against the supplied evidence;
- answer correctness against the reference answer;
- answer relevance and abstention rate;
- latency and API/token cost;
- paired effect size and bootstrap confidence interval.

The detector used for the final claim should be validated in Stage A. Prefer a
judge independent of the generator, and retain blinded human review for a
representative subset.

### Stage C: proposed method

Freeze the datasets, detector thresholds, baselines, and analysis plan before
evaluating the proposed method. Report upstream reproductions, local inspired
baselines, provider services, and the proposed method as separate conditions.

## Execution topology

The source repository does not need an LLM or model checkpoint. The intended
topology is:

```text
controller / analysis machine
    -> remote OpenAI-compatible endpoint or remote Ollama
    -> optional evaluator service with SelfCheckGPT, SummaC, AlignScore, etc.
    -> JSONL/CSV result artifacts with complete run metadata
```

Ollama can be accessed remotely through its native API or OpenAI-compatible
endpoint. API-backed detectors such as Amazon Bedrock contextual grounding and
OpenAI Guardrails must be reported as provider baselines, not as open-source
algorithm reproductions.

## Environment layout

Keep dependency groups separate:

- `requirements.txt`: benchmark controller only;
- `requirements-colab-*.txt`: one pinned official method per environment;
- `requirements-upstream.txt`: index explaining the environment split;
- a dedicated lock/container for each legacy reproduction;
- no automatic model pulls in committed configurations.

Before publishing a run, archive the resolved dependency lock, configuration,
source registry, prompts, raw generations, raw detector outputs, and the Git SHA
of this repository alongside the report.

## Preparing official data on the execution machine

Do this on the data/execution machine, not on the lightweight controller:

```bash
git clone https://github.com/RUCAIBox/HaluEval.git external_data/HaluEval
git -C external_data/HaluEval checkout b7253db3cdaa0ab2c382f92b26b390109174f77e
```

Then enable `halueval_qa` in `config.yaml`. The path already points to the
official `qa_data.json`. Keep `external_data/` out of Git; record a checksum of
the input file with each reported run.
