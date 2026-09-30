# How to run

How the pieces fit together, with diagrams: [`ARCHITECTURE.md`](ARCHITECTURE.md).

## 0. First-time setup on a new Linux machine

You install two things by hand: the Python packages and Ollama itself.
Everything else a run needs is fetched by the run (§0.4).

### 0.1 Clone the repo

```bash
git clone <this-repo-url>
cd LLM-hallucination-Research
```

### 0.2 Get `pip` if it's missing

`python3` without `pip` is common on Debian/Ubuntu, which ships them
separately:

```bash
sudo apt update && sudo apt install -y python3-pip python3-venv   # Debian/Ubuntu
# or: sudo dnf install -y python3-pip                             # Fedora/RHEL
```

### 0.3 Create a venv and install the Python dependencies

This is the **core** environment: it runs `main.py` and
`scripts/run_full.py`, and holds SelfCheckGPT, UQLM and the UQLM judge (they
share the generator samples). MiniCheck / SummaC / AlignScore get their own
venvs, created automatically by `run_full.py` (see [§4](#4-full-run) for
why):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 0.4 Install and start Ollama

Ollama is a system install, not a Python package: follow ollama.com for your
distribution (version 0.9 or newer, for the reasoning-model `think`
setting), then make sure it is running (`ollama serve`, or the system
service). `config.yaml`'s `ollama.host` points at `http://localhost:11434`.

You do **not** pull models or download data yourself. Every run starts with
a **Setup** step that checks each resource and fetches whatever is missing:

| Resource | Source | Check |
|---|---|---|
| HaluEval QA / dialogue / summarization (~58 MB) | RUCAIBox/HaluEval, pinned commit | SHA-256 |
| RAGTruth responses + sources (~36 MB) | ParticleMedia/RAGTruth, pinned commit | SHA-256 |
| HaluBench test parquet (~7 MB) | PatronusAI/HaluBench, pinned revision | SHA-256 |
| SummaC-Conv weights (2 KB, with `summac`) | tingofurro/summac, pinned commit | SHA-256 |
| AlignScore-base checkpoint (~1.9 GB, with `alignscore`) | authors' Hugging Face repo `yzha/AlignScore`, pinned revision | SHA-256 |
| spaCy `en_core_web_sm`, NLTK `punkt` | pinned wheel / NLTK | installed |
| Every Ollama model in `selected_models` + the judge (`judge.model`) | `ollama pull` (off with `ollama.auto_pull: false`) | listed by the server |

Downloads show a progress bar; a file with the wrong checksum is discarded,
never used. The detectors' neural models (RoBERTa / DeBERTa for SelfCheckGPT
and UQLM, MiniCheck's Flan-T5, SummaC's ALBERT) are downloaded by their own
packages from Hugging Face the first time they load, which happens in the
preflight (§1); the exact revision used is recorded in each run's
`run_manifest.json`.

To see what is present and what would be fetched, without downloading
anything:

```bash
python main.py --dry-run --detectors selfcheckgpt uqlm uqlm_judge
```

## 1. How every run protects your time

Every `main.py` run goes through the same order, and stops at the first
problem, before anything long starts:

1. **Setup**: download/pull whatever is missing (§0.4).
2. **Preflight**: one real question through the whole pipeline. Every
   detector loads and scores it, every model answers and draws its samples,
   every sampling-based detector scores with them, and with the reduction
   stage on, every reduction method runs once per model and all its answers
   are scored. Each check prints ✓ or ✗ with the reason. Any ✗ stops the run.
3. **Estimate**: the preflight timings give the expected time per run and in
   total.
4. The long stages.

`python main.py ... --preflight` does steps 1–3 and stops, so you can check
a machine (and see the time estimate) before committing to a run.
`scripts/run_full.py` runs the preflight of **every** environment before
starting any long run (§4).

## 2. What the full run does

Everything about the size of a run lives in one place, the `run:` block at
the top of `config.yaml`. `python scripts/run_full.py` runs exactly this:

| Setting (`run:`) | Default | Meaning |
|---|---|---|
| `runs` | 3 | independent repeats: `run_01`, `run_02`, `run_03`, then `combined/` with mean ± std |
| `samples_per_dataset` | 20 | questions per dataset, 10 datasets (HaluEval ×3, RAGTruth ×3, HaluBench ×4) |
| `detectors` | selfcheckgpt, uqlm, uqlm_judge, minicheck, summac | add `alignscore` to include it |
| `selfcheckgpt_samples` | 5 | answers sampled per question, per model, per run (shared by SelfCheckGPT and UQLM) |
| `reduce` | true | Stage B: every method in `reduction.methods` vs. the baseline answer |
| `reduction_iterations` | 3 | max rounds for `self_refine_adapted` |

Models are every entry in `selected_models` (5 by default) plus the judge
(`judge.model`, Mistral 7B, used by the UQLM judge and SelfCheckGPT's prompt
scorer); datasets are every `enabled` entry under `datasets:`. With the
defaults, one run is:

- **data**: 10 datasets × 20 = **200 questions**. HaluEval gives 2 labeled
  answers per question, RAGTruth about 6 (one per LLM it collected), HaluBench
  1–2: about **570 labeled answers**. The same questions (seed 42) are used in
  every run.
- **Stage A, detector validation**: each model samples 5 answers per question
  (200 × 5 × 5 models = 5,000 generations); every detector then scores all
  ~570 labeled answers (the sampling-based ones once per model). SelfCheckGPT's
  prompt scorer asks the judge once per sentence per sample, which is the
  most expensive part.
- **Stage B, reduction**: each model answers the 200 questions six ways
  (baseline, closed-book, greedy, Self-Refine, CoVe, UQLM best response;
  up to 16 generations per question), and every detector scores all six
  answers. MiniCheck/SummaC score them in their own environments afterwards.
- **× 3 runs**, fresh sampling in each, then the combined report.

This is a long run on real hardware. The preflight measures every step on
one real question and prints the expected time per run and in total before
anything long starts; lower `samples_per_dataset` or `runs`, or remove the
`prompt` scorer from `detectors.selfcheckgpt.methods`, if it is too long.

Command-line flags override the plan for one run only (§ Useful flags); the
plan actually used is printed at the start ("Run plan") and saved in every
run's `config_used.yaml` and in the report.

## 3. Smoke test first — always

The full run with tiny numbers: **2 runs**, 2 questions per dataset, 2
samples, 1 refine round, **every environment** (core, MiniCheck, SummaC).
It produces the same folders and files as the full run:

```bash
python scripts/run_full.py --runs 2 --max-samples 2 --n-samples 2 --max-iterations 1 \
  --output results/smoke-test
```

(`--max-samples` is questions **per dataset**; `--runs` is the number of
independent repeats.)

Check: the final table shows ✓ for every environment, the results tables
show `n_failed` 0 (a few failures for SelfCheckGPT BERTScore on very short
answers are an upstream limitation, recorded per case; anything else is a
problem to look at in `run.log`), and
`results/smoke-test/combined/report.docx` (or `report.html`) opens with its
tables and charts. If a check fails, the preflight stops the run within
minutes and the environment's `run.log` has the full traceback.

A quicker check of the core environment alone (no MiniCheck / SummaC, so no
`reduction_scores.csv`):

```bash
python main.py --detectors selfcheckgpt uqlm uqlm_judge --reduce \
  --runs 2 --max-samples 2 --n-samples 2 --max-iterations 1 --output results/smoke-core
```

## What the terminal shows

```text
══ LLM hallucination benchmark ═════════════════════════════════════
── Run plan ────  runs, questions per dataset, detectors, samples,
                  reduction methods, judge, models
── Setup ───────  ✓ present / ↓ downloading (data, weights, models)
── Data ────────  per dataset: questions → labeled answers (faithful / hallucinated)
── Detectors / Generators / Workload
── Preflight ───  ✓/✗ per detector, model and method + time estimate
══ Run 1/3 → results/.../run_01 ══════════════════════════════════
── Run 1/3 · Stage 1/2 · Detector validation
  [1/5] llama3.2-3b   42%|██████████      | 240/570 [05:10<07:07, 1.3case/s]
  ✓ [1/5] llama3.2-3b · selfcheckgpt, uqlm · 570 cases in 12m17s
── Run 1/3 · Detector validation results   (table, n_failed per row)
── Run 1/3 · Stage 2/2 · Reduction          (one bar per model)
── Run 1/3 · Reduction results             (mean change vs. baseline, method × detector)
══ Run 2/3 … ══ Run 3/3 …
══ Combined · 3 run(s) ══   key takeaways, mean ± std over runs
── Done in … ──  folder layout, which report to open
```

Each bar is one model, so `[elapsed<remaining]` is that model's own ETA.
Only one-line warnings reach the screen; tracebacks, retries and
third-party warnings go to `run.log`. A model or detector that fails 10
cases in a row is stopped early with the reason.

## 4. Full run

```bash
tmux new -s run                  # a multi-hour run should survive a closed terminal
python scripts/run_full.py       # exactly the `run:` block of config.yaml
```

Why several environments: MiniCheck, SummaC and AlignScore pin conflicting
`torch`/`transformers` versions upstream, so each gets its own
`.venv-<name>` (created and reused by `run_full.py`, pip output in
`<output>/logs/`, reinstalled automatically if its requirements changed).
SelfCheckGPT, UQLM and the judge share the generator samples and run
together in the **core** environment, which is `requirements.txt` (§0.3)
and also runs the reduction stage. The other environments then score the
core run's reduction answers, so every detector judges every method.

It runs in two phases:

- **Phase 1 · prepare and check**: every environment is installed and its
  `main.py --preflight` runs (download everything, one real question through
  the whole pipeline). If any environment fails, the script stops and lists
  why; nothing long has started. Rerun the same command after fixing it;
  environments and downloads are reused.
- **Phase 2 · runs**: only when every environment passed: core first (it
  produces the reduction answers), then the others, then the top-level
  combined report.

`alignscore` is opt-in: uncomment it in `run.detectors`, or pass all six
detectors, `--detectors selfcheckgpt uqlm uqlm_judge minicheck summac
alignscore`. Its legacy environment is heavy and its ~1.9 GB checkpoint is
downloaded (and checksum-verified) on first use. If one environment fails in
phase 2, the others still run and its line in the final table shows `✗` with
the path of its `run.log`.

## What a run writes

```text
results/full-run-<time>/                       (scripts/run_full.py)
  combined/                ← start here: all detectors, all runs
  core/                                        SelfCheckGPT + UQLM + judge + reduction
    run_01/  run_02/  run_03/                  each a complete, independent run
    combined/                                  this environment, mean ± std over its runs
    run.log
  minicheck/   (same layout; also scores core's reduction answers)
  summac/      (same layout; also scores core's reduction answers)
  logs/pip-<env>.log

results/smoke-core/                            (main.py, one environment)
  run_01/  run_02/  combined/  run.log
```

**Every report folder** (`run_XX/`, `<env>/combined/`, `combined/`) has:

| File | What it holds |
|---|---|
| `report.docx` | the full report as a Word document: run plan, key takeaways, all tables, all charts |
| `report.html` | the same report as one self-contained web page (charts embedded) |
| `REPORT.md` | the same report in Markdown (charts in `charts/`) |
| `takeaways.md` | just the key findings, in plain sentences |
| `charts/*.png` | how the experiment works (flow diagram), detectors ranked by AUROC, AUROC per detector × dataset, change in risk per method judged by each validated detector (95% CI), and in the appendix: every detector's view of Stage B and AUROC in every run |
| `tables/*.csv` | every table in the report |

The report reads top to bottom as a research report; every chart has a
"How to read this chart" box and a "What it shows" sentence:

1. **Summary**: the answers in plain sentences (most reliable detector,
   which detectors are validated, for each method whether it lowered risk,
   with 95% confidence intervals).
2. **How the experiment works**: flow diagram, datasets, detectors and
   methods in one line each.
3. **Stage A**: detectors ranked by AUROC (chance and validation lines), one
   compact table, AUROC per dataset and (with several models) per generator
   model. A detector that gives every answer the same score is flagged as
   "no signal" and never used as a judge.
4. **Stage B**: each method's change in risk vs. the baseline answer, judged
   only by the **validated detectors** (AUROC ≥ 0.65 in Stage A; if none
   passes, the three best, flagged as indicative), with 95% bootstrap
   confidence intervals; the same by generator model and by dataset; one
   example answer per method.
5. **Reliability**: run-to-run variation, failures by cause.
6. **Appendix**: exactly what ran (models, versions, commit), Stage B as
   seen by every detector, AUROC per run, and every data file.

The full detail (every answer, every score, every table) is in the CSV
files, not in the document.

**A `combined/` folder** reports every metric as mean ± std over the runs,
and adds `tables/*_by_run.csv` (each run's numbers side by side),
`tables/*_mean_std.csv`, `raw_all_runs.csv` and `reduction_all_runs.csv`
(every row of every run, with a `run` column).

**Each `run_XX/` folder** also keeps that run's raw data:

| File | What it holds |
|---|---|
| `detector_validation_summary.csv` | AUROC, AUPRC, accuracy, precision, recall, F1, `n_cases`, `n_failed` per detector (per model for SelfCheckGPT and UQLM) |
| `detector_validation_raw.csv` | every labeled answer: dataset, question, context, answer, label, every detector's score, errors |
| `reduction_comparison.csv` | (core) one row per question × model × method: the answer, every detector's score, generation calls, seconds, method details (feedback, verification questions, chosen candidate), `reproduction_status`, errors |
| `reduction_scores.csv` | (minicheck/summac/alignscore) their scores of core's reduction answers, joined into the reports automatically |
| `selfcheckgpt_samples.jsonl` | every sampled answer SelfCheckGPT and UQLM compared against (one line per model × question) |
| `run_manifest.json` | command, start/end time, per-stage seconds, Git commit (+ uncommitted flag), package versions, dataset sha256s, Ollama model digests, Hugging Face model revisions |
| `config_used.yaml`, `environment.txt` | the resolved config (incl. the run plan) and every installed package version |

To rebuild reports later: `python scripts/generate_report.py --input
<run folder>` or `--input <parent> --combined`.

What stays constant and what varies between runs: the questions (seed) are
the same in every run; the samples and every reduction answer are drawn
fresh each run, so the std shows how stable each number is under the models'
own randomness. Within a run, the samples are drawn once per model and
question and every answer to that question is scored against them (the
labeled answers in Stage A, every method's answer in Stage B). `max_samples`
picks a seeded random subset of each dataset, and `sample_id` traces back to
the source row. If any enabled dataset fails to load, the run stops instead
of silently continuing on partial data.

## GPU or no GPU?

- **Ollama** uses the machine's GPU automatically if there is one.
- **Detectors**: SelfCheckGPT (BERTScore, NLI), UQLM, SummaC and AlignScore
  run torch models. The config default `device: "auto"` uses the GPU when
  torch sees one and the CPU otherwise; `--device cpu|cuda` (to `main.py` or
  `run_full.py`) forces one. The device used is printed in the Detectors
  section. MiniCheck uses the GPU by itself when one is present.
  SelfCheckGPT's n-gram and prompt scorers need no GPU.

## Useful flags

| Flag | `main.py` | `run_full.py` | What it does |
|---|---|---|---|
| `--runs N` | ✓ | ✓ | Independent repeats (`run.runs`) |
| `--max-samples N` | ✓ | ✓ | Questions per dataset (`run.samples_per_dataset`) |
| `--n-samples N` | ✓ | ✓ | Samples per question per model (`run.selfcheckgpt_samples`) |
| `--max-iterations N` | ✓ | ✓ | Self-Refine rounds (`run.reduction_iterations`) |
| `--detectors …` | ✓ | ✓ | Which detectors run (`run.detectors`) |
| `--no-reduce` | ✓ | ✓ | Skip the reduction stage (`run.reduce`) |
| `--reduce` | ✓ | | Force the reduction stage on |
| `--device auto\|cpu\|cuda` | ✓ | ✓ | Device for the torch-based detectors (default `auto`: GPU if present) |
| `--output DIR` | ✓ | ✓ | Where results are written |
| `--config FILE` | ✓ | ✓ | Another config file (default `config.yaml`) |
| `--preflight` | ✓ | | Download everything, check the pipeline on one question, print the estimate, stop |
| `--dry-run` | ✓ | | Show what is present / would be downloaded; download and load nothing |
| `--score-reduction-from DIR` | ✓ | | Score another environment's reduction answers (used by `run_full.py`) |
| `--skip-report` | | ✓ | Skip the top-level combined report |

Which models run and which datasets are enabled come from `config.yaml`
(`selected_models`, `datasets[].enabled`).

## What the numbers are, and what they are not

- **Stage A** tells you how well each detector separates faithful from
  hallucinated answers on labeled data (AUROC/AUPRC need no threshold; the
  other metrics use uncalibrated thresholds).
- **Stage B** tells you, per method, how the risk score of each validated
  detector changes compared with the model's own baseline answer to the same
  question. Repeated runs of one question × model are averaged first, so each
  pair counts once; the 95% confidence interval is a percentile bootstrap
  over those pairs (2,000 resamples, fixed seed), given only when there are
  at least 10 pairs. A method is reported as
  "lower risk" only when the whole interval is below zero. Every row of the
  CSVs says what the method is in `reproduction_status` (official
  implementation, local implementation of a paper, local inspired
  adaptation, ablation, or decoding setting); see
  [`METHOD_SOURCES.md`](../METHOD_SOURCES.md).

Neither is publishable evidence on its own yet: detector thresholds are not
calibrated on a held-out split, the validation bar is a fixed choice, small
runs give wide intervals, and there is no human review (see
[`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)).
