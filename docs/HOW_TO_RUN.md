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

Check: every results table in the terminal shows `n_failed` 0, the final
table shows ✓ for every environment, and
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
| `charts/*.png` | AUROC and AUPRC per detector × model, AUROC per detector × dataset, AUROC in every run (consistency), change vs. baseline per method × detector and per method × model, share of questions improved, seconds per answer |
| `tables/*.csv` | every table in the report |

The report sections: **Run plan** (runs, datasets and sizes, models with
size/quantization, detectors and their settings, judge, reduction methods,
seed, code commit, package versions, run time) · **Key takeaways** ·
**Stage A** detector validation (per detector × model, per dataset with
failure counts) · **Stage B** reduction (change vs. baseline per method ×
detector, by model, by dataset, cost, example answers that improved most /
got worse most) · **Run by run** (combined only) · **Failures** · **Files**.

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
  run torch models. Pass `--device cuda` (to `main.py` or `run_full.py`) on a
  GPU machine; the default is `cpu`, which works but is much slower.
  MiniCheck uses the GPU by itself when one is present. SelfCheckGPT's
  n-gram and prompt scorers need no GPU.

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
| `--device cpu\|cuda` | ✓ | ✓ | Device for the torch-based detectors |
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
- **Stage B** tells you, per method, how each detector's risk score changes
  compared with the model's own baseline answer to the same question. Every
  row says what the method is in `reproduction_status` (official
  implementation, local implementation of a paper, local inspired
  adaptation, ablation, or decoding setting); see
  [`METHOD_SOURCES.md`](../METHOD_SOURCES.md).

Neither is publishable evidence on its own yet: there is no held-out
calibration split, no confidence intervals beyond run-to-run std, and no
human review (see [`REPRODUCIBILITY.md`](REPRODUCIBILITY.md)).
