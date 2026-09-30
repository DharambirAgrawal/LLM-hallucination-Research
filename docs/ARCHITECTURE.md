# Architecture

How the pieces fit together, from the command you type to the report you
read. For setup and commands see [`HOW_TO_RUN.md`](HOW_TO_RUN.md); for where
each method comes from see [`../METHOD_SOURCES.md`](../METHOD_SOURCES.md).

## 1. The big picture

```mermaid
flowchart LR
    cfg["config.yaml<br/>run plan · models · datasets ·<br/>detectors · reduction methods"]
    rf["scripts/run_full.py<br/>one process per environment"]
    main["main.py<br/>one environment, end to end"]
    cfg --> rf --> main
    cfg --> main

    subgraph IN["Inputs (downloaded + checksummed)"]
        data["Datasets<br/>HaluEval · RAGTruth · HaluBench"]
        llm["Ollama models<br/>5 generators + 1 judge"]
        pkgs["Official detector packages<br/>+ their model weights"]
    end

    subgraph RUN["What main.py does"]
        pre["Preflight"]
        sa["Stage A<br/>detector validation"]
        sb["Stage B<br/>reduction methods"]
    end

    subgraph OUT["Outputs"]
        csv["raw CSVs + samples + manifest"]
        rep["reports: docx · html · md<br/>charts · tables · takeaways"]
    end

    IN --> RUN
    main --> RUN
    RUN --> csv --> rep
```

## 2. Environments

The official detector packages pin conflicting versions of `torch` and
`transformers`, so they cannot all live in one Python environment.
`scripts/run_full.py` therefore runs `main.py` once per environment and
merges everything at the end.

```mermaid
flowchart TD
    rf(["scripts/run_full.py"])
    rf --> p1["Phase 1 · for every environment:<br/>install / reuse venv → setup → preflight"]
    p1 -->|any environment fails| stop(["Stop before any long run,<br/>with the reason per environment"])
    p1 -->|all pass| core

    subgraph P2["Phase 2 · the runs"]
        core["core (requirements.txt)<br/>SelfCheckGPT · UQLM · UQLM judge<br/>Stage A + Stage B (produces the reduction answers)"]
        mc["minicheck (.venv-minicheck)<br/>Stage A + scores core's reduction answers"]
        su["summac (.venv-summac)<br/>Stage A + scores core's reduction answers"]
        al["alignscore (.venv-alignscore, opt-in)<br/>Stage A + scores core's reduction answers"]
        core --> mc --> su --> al
    end

    al --> comb["combined/<br/>every detector · every run · mean ± std"]
```

`core` runs first because the other environments score the answers it
produced (`--score-reduction-from core`). The reports join those scores back
onto core's answers by run, question, model and method.

## 3. Inside `main.py`

```mermaid
flowchart TD
    a["Resolve the run plan<br/>flag, then run: block (details in main.apply_plan)"] --> b["Setup · utils/resources.py<br/>download missing data + weights (SHA-256 checked),<br/>pull missing Ollama models"]
    b --> c["Load data · data/datasets.py<br/>seeded subset, same questions every run"]
    c --> d["Build detectors · benchmark/runner.py<br/>and generators · models/model_factory.py"]
    d --> e["Preflight · benchmark/preflight.py<br/>1 real question through everything → time estimate"]
    e --> f{"for run 1 … N"}
    f --> g["Stage A · runner.validate()"]
    g --> h["Stage B · ReductionRunner.run()<br/>(or score another environment's answers)"]
    h --> i["Archive: selfcheckgpt_samples.jsonl ·<br/>run_manifest.json · config_used.yaml · environment.txt"]
    i --> j["Report for this run · reporting/"]
    j --> f
    f -->|done| k["Combined report · reporting/<br/>mean ± std over runs"]
```

## 4. Stage A: can the detectors be trusted?

Every question comes with answers whose label is known. Detectors that do
not need a model score each answer once; the sampling-based detectors run one
model at a time over every answer, so Ollama keeps a single model loaded.

```mermaid
flowchart LR
    subgraph Q["For each question"]
        known["Known answers<br/>label 0 = faithful · 1 = hallucinated"]
        ctx["Context"]
    end

    subgraph IND["Model-independent (once per answer)"]
        j["UQLM judge"]
        m["MiniCheck"]
        s["SummaC"]
        al["AlignScore"]
    end

    subgraph DEP["Per generator model"]
        bank["SampleBank<br/>5 answers from the model,<br/>drawn once per question"]
        sc["SelfCheckGPT ×4"]
        uq["UQLM consistency ×5"]
        bank --> sc
        bank --> uq
    end

    known --> IND
    ctx --> IND
    known --> sc
    known --> uq
    IND --> met["AUROC · AUPRC · accuracy ·<br/>precision · recall · F1 · failures"]
    DEP --> met
```

The model-independent scores are deterministic, so later runs reuse the first
run's scores instead of recomputing them; only the sampling-based detectors
vary between runs.

## 5. Stage B: do the methods reduce hallucination?

```mermaid
sequenceDiagram
    participant R as ReductionRunner
    participant M as Generator model
    participant B as SampleBank
    participant D as Every detector

    R->>M: grounded prompt (question + context)
    M-->>R: baseline answer
    R->>M: closed_book: question only
    R->>M: greedy: grounded prompt, temperature 0
    R->>M: self_refine_adapted: feedback → rewrite (≤ N rounds)
    R->>M: cove_adapted: plan checks → answer each → revise
    R->>B: samples for this question (already drawn in Stage A)
    Note over R,B: uqlm_best_response picks the most consistent<br/>answer among baseline + samples (no new generation)
    loop every answer (baseline + 5 methods)
        R->>D: score answer against the same samples / context
        D-->>R: risk per detector
    end
    Note over R: report: change vs. baseline on the same<br/>question and model, share better / worse
```

The report then judges each method only with the detectors that passed
Stage A (AUROC ≥ 0.65): the mean change vs. the baseline per question × model
pair, with a 95% bootstrap confidence interval. Every detector's view is
still in the appendix and the CSVs.

Two details keep this comparison fair:

- **Same evidence.** Every answer to a question is scored against the same
  samples, so differences come from the answers, not from new randomness.
- **Leave-one-out.** When `uqlm_best_response` picks one of the samples, every
  exact copy of it is taken out of its evidence and the baseline answer is
  added; otherwise it would be scored against a copy of itself.

## 6. What a results folder contains

```text
<output>/
  run_01/ … run_N/        one complete, independent run each
    detector_validation_raw.csv      every labeled answer × every detector
    detector_validation_summary.csv  metrics per detector (per model where relevant)
    reduction_comparison.csv         every method's answer × every detector (core)
    reduction_scores.csv             another environment's scores of those answers
    selfcheckgpt_samples.jsonl       every sampled answer used as evidence
    run_manifest.json · config_used.yaml · environment.txt
    report.docx · report.html · REPORT.md · takeaways.md · charts/ · tables/
  combined/                mean ± std over runs: the same reports + all rows
  run.log                  full log with tracebacks
```

`scripts/run_full.py` puts one such folder per environment (`core/`,
`minicheck/`, …) and adds a top-level `combined/` over all of them.

## 7. Design rules

| Rule | Where | Why |
|---|---|---|
| Official code only; local implementations labeled | `detectors/`, `reducers/`, `reproduction_status` column | results must be citable |
| Every download pinned and SHA-256 checked | `utils/resources.py` | same data and weights on every machine |
| Fail fast | `benchmark/preflight.py`, 10-failures-in-a-row stop | find problems in minutes, not hours |
| Failures are recorded, never scored as 0 | `*_error` columns, `n_failed` | a crash must not look like a result |
| One sample set per model and question | `detectors/sampling.py` | paired comparisons on identical evidence (except the leave-one-out case, §5) |
| One model at a time | `benchmark/runner.py` | Ollama keeps one model loaded |
| Same outputs for smoke and full runs | `scripts/run_full.py` with small numbers | a smoke test checks every environment the full run uses |
| Never mix runs | `main.py` refuses an output folder that already holds runs | a combined report must only contain this run's repeats |

## 8. Extending it

- **A detector:** write a thin adapter in `detectors/` around the official
  package, register it as a family in `benchmark/runner.py` (score columns,
  thresholds, whether it needs a generator), add its source to
  `provenance/sources.yaml` and `METHOD_SOURCES.md`; if its dependencies
  conflict, add a `requirements/<name>.txt` and an entry in
  `scripts/run_full.py`'s `ENVIRONMENTS`.
- **A dataset:** add a loader in `data/datasets.py` that returns questions with
  labeled answers, its pinned files in `utils/resources.py`, and an entry in
  `config.yaml`.
- **A reduction method:** add it in `reducers/`, wire it in
  `ReductionRunner._produce` with a `reproduction_status` that says exactly
  what it is, and list it in `config.yaml` under `reduction.methods`.
