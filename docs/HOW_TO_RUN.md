# How to run

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

This is the "controller" environment used for `main.py` and the
`scripts/run_full.py` orchestrator. It includes the pinned official
SelfCheckGPT package; MiniCheck/SummaC/AlignScore get their own venvs,
created automatically by `run_full.py` (see
[§4](#4-full-run) for why):

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
| HaluEval QA / dialogue / summarization (~58 MB) | RUCAIBox/HaluEval at the pinned commit | SHA-256 |
| AlignScore-base checkpoint (~1.9 GB, only with `alignscore`) | authors' Hugging Face repo `yzha/AlignScore`, pinned revision | SHA-256 |
| spaCy `en_core_web_sm`, NLTK `punkt` | pinned wheel / NLTK | installed |
| Every Ollama model in `selected_models` | `ollama pull` (off with `ollama.auto_pull: false`) | listed by the server |

Downloads show a progress bar; a file with the wrong checksum is discarded,
never used. MiniCheck/SummaC fetch their own model weights from Hugging
Face the first time they load, which happens in the preflight (§1).

To see what is present and what would be fetched, without downloading
anything:

```bash
python main.py --dry-run --detectors selfcheckgpt
```

## 1. How every run protects your time

Every `main.py` run goes through the same order, and stops at the first
problem, before anything long starts:

1. **Setup**: download/pull whatever is missing (§0.4).
2. **Preflight**: one real case through the whole pipeline. Every detector
   loads and scores it, every model answers, SelfCheckGPT scores the case
   with every model, and with `--reduce` one full reduction round runs per
   model. Each check prints ✓ or ✗ with the reason. Any ✗ stops the run.
3. **Estimate**: the preflight timings give the expected time per stage.
4. The long stages.

`python main.py ... --preflight` does steps 1–3 and stops, so you can
check a machine (and see the time estimate) before committing to a run.
`scripts/run_full.py` runs the preflight of **every** detector, each in its
own environment, before starting any long run (§4).

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
  up to ~15 generations per question), and every detector scores all six
  answers. MiniCheck/SummaC score them in their own environments afterwards.
- **× 3 runs**, fresh sampling in each, then the combined report.

This is a long run on real hardware. The preflight measures every step on
one real question and prints the expected time per run and in total before
anything long starts; lower `samples_per_dataset` or `runs`, or remove the
`prompt` scorer from `detectors.selfcheckgpt.methods`, if it is too long.

Every run prints this plan at the start ("Run plan"), and the preflight
prints the estimated time per run and in total before the long part starts.
Command-line flags override the plan for one run only (`--runs`,
`--max-samples`, `--n-samples`, `--max-iterations`, `--detectors`,
`--no-reduce`); the plan actually used is saved in every run's
`config_used.yaml` and in the report.

## 3. Smoke test first — always

The full run with tiny numbers: **2 runs**, 2 samples per dataset, 2
SelfCheckGPT samples, 1 refine round. It produces exactly the same folders
and files as the full run:

```bash
python main.py --detectors selfcheckgpt --reduce \
  --runs 2 --max-samples 2 --n-samples 2 --max-iterations 1 \
  --output results/smoke-test
```

(`--max-samples` is samples **per dataset**; `--runs` is the number of
independent repeats.)

Check: every results table in the terminal shows `n_failed` 0, and
`results/smoke-test/combined/report.docx` (or `report.html`) opens with its
tables and charts. If a check fails, the preflight stops the run within
minutes and `results/smoke-test/run.log` has the full traceback.

The same smoke test for every detector, each in its own environment:

```bash
python scripts/run_full.py --runs 2 --max-samples 2 --n-samples 2 --max-iterations 1
```

## What the terminal shows

```text
══ LLM hallucination benchmark ═════════════════════════════════════
── Run plan ────  runs, samples per dataset, detectors, SelfCheckGPT
                  samples, reduction rounds, models
── Setup ───────  ✓ present / ↓ downloading (data, models, tokenizers)
── Data ────────  one line per dataset: samples → labeled cases
── Detectors / Generators / Workload
── Preflight ───  ✓/✗ per detector, model and stage + time estimate
══ Run 1/3 → results/.../run_01 ══════════════════════════════════
── Run 1/3 · Stage 1/2 · Detector validation
  [1/5] llama3.2-3b   42%|██████████      | 133/316 [01:10<01:37, 1.9case/s]
  ✓ [1/5] llama3.2-3b · 316 cases in 2m47s
── Run 1/3 · Detector validation results   (table, n_failed per row)
── Run 1/3 · Stage 2/2 · Reduction          (one bar per model)
── Run 1/3 · Reduction results             (baseline → refined, better/worse)
══ Run 2/3 … ══ Run 3/3 …
══ Combined · 3 run(s) ══   key takeaways, mean ± std over runs
── Done in 3h12m ──  folder layout, which report to open
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

- **Phase 1 · prepare and check**: every detector's environment is
  installed and its `main.py --preflight` runs (download everything, one
  real case through the whole pipeline). If any detector fails, the script
  stops and lists why; nothing long has started. Rerun the same command
  after fixing it; environments and downloads are reused.
- **Phase 2 · runs**: only when every detector passed, each detector runs
  all its runs, then the top-level combined report is built.

`alignscore` is opt-in (add it to `run.detectors` or pass
`--detectors selfcheckgpt minicheck summac alignscore`): its legacy
environment is heavy and its ~1.9 GB checkpoint is downloaded (and
checksum-verified) on first use. If one detector's run fails in phase 2,
the others still run and its line in the final table shows `✗` with the
path of its `run.log`.

A single detector without the orchestrator: `python main.py --detectors
selfcheckgpt` (uses the plan's runs/samples/reduction).

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

results/smoke-test/                            (main.py)
  run_01/  run_02/  combined/  run.log
```

**Every report folder** (`run_XX/`, `<detector>/combined/`, `combined/`)
has:

| File | What it holds |
|---|---|
| `report.docx` | the full report as a Word document: run plan, key takeaways, all tables, all charts |
| `report.html` | the same report as one self-contained web page (charts embedded) |
| `REPORT.md` | the same report in Markdown (charts in `charts/`) |
| `takeaways.md` | just the key findings, in plain sentences |
| `charts/*.png` | detector validation (AUROC/AUPRC/F1), AUROC per dataset, run-to-run consistency, reduction scores by model and by dataset, better/worse share, latency |
| `tables/*.csv` | every table in the report |

The report sections: **Run plan** (runs, datasets and sizes, models with
size/quantization, detector settings, reduction settings, seed, code
commit, package versions, run time) · **Key takeaways** · **Stage A**
detector validation (overall and per dataset) · **Stage B** reduction (by
model, by dataset, by dataset × model, why the loop stopped, latency,
example answers that improved most / got worse most) · **Run by run**
(combined only) · **Failures** · **Files**.

**A `combined/` folder** reports every metric as mean ± std over the runs,
and adds `tables/*_by_run.csv` (each run's numbers side by side),
`tables/*_mean_std.csv`, `raw_all_runs.csv` and `reduction_all_runs.csv`
(every row of every run, with a `run` column).

**Each `run_XX/` folder** also keeps that run's raw data:

| File | What it holds |
|---|---|
| `detector_validation_summary.csv` | AUROC, AUPRC, accuracy, precision, recall, F1, `n_cases`, `n_failed` per detector (per model for SelfCheckGPT) |
| `detector_validation_raw.csv` | every case: dataset, question, context, answer, label, score, error |
| `reduction_comparison.csv` | (core) one row per question × model × method: the answer, every detector's score, generation calls, seconds, method details (feedback, verification questions, chosen candidate), `reproduction_status`, errors |
| `reduction_scores.csv` | (minicheck/summac/alignscore) their scores of core's reduction answers, joined into the reports automatically |
| `selfcheckgpt_samples.jsonl` | every sampled answer SelfCheckGPT compared against (one line per model × question) |
| `run_manifest.json` | command, start/end time, per-stage seconds, Git commit (+ uncommitted flag), package versions, dataset sha256s, Ollama model digests |
| `config_used.yaml`, `environment.txt` | the resolved config (incl. the run plan) and every installed package version |

To rebuild reports later: `python scripts/generate_report.py --input
<run folder>` or `--input <parent> --combined`.

What stays constant and what varies between runs: the data subset (seed)
is the same in every run; SelfCheckGPT samples and reduction answers are
drawn fresh each run, so the std shows how stable each number is under the
models' own randomness. Within a run, SelfCheckGPT draws its samples once
per model and question and scores every answer to that question against
them (factual vs. hallucinated in Stage A, baseline vs. refined in Stage
B). `max_samples` picks a seeded random subset of each HaluEval file, and
`sample_id` keeps the original line number. If any enabled dataset fails
to load, the run stops instead of silently continuing on partial data.

## GPU or no GPU?

- **Ollama**: uses the machine's GPU automatically if there is one — nothing
  to configure either way.
- **Detectors**: the default (`selfcheckgpt` method `ngram`) is pure CPU, no
  GPU needed. If you switch to a neural method (`nli`/`bertscore`, or enable
  SummaC/AlignScore), pass `--device cuda` if a GPU is present, otherwise
  leave it as `--device cpu` (default).

## Useful flags

| Flag | What it does |
|---|---|
| `--runs N` | Independent repeats (`run.runs`) |
| `--max-samples N` | Samples per dataset (`run.samples_per_dataset`) |
| `--n-samples N` | SelfCheckGPT samples per question (`run.selfcheckgpt_samples`) |
| `--max-iterations N` | Reduction feedback → refine rounds (`run.reduction_iterations`) |
| `--detectors selfcheckgpt ...` | Which detector(s) to run (`run.detectors`) |
| `--reduce` / `--no-reduce` | Force the reduction stage on / off (`run.reduce`) |
| `--device cpu\|cuda` | Device for SelfCheckGPT/SummaC/AlignScore's torch models |
| `--output DIR` | Where results are written |
| `--config FILE` | Another config file (default `config.yaml`) |
| `--preflight` | Download everything, check the pipeline on one case, print the estimate, stop |
| `--dry-run` | Show what is present / would be downloaded; download and load nothing |

Which model(s) run and which dataset(s) are enabled still come from
`config.yaml` (`selected_models`, `datasets[].enabled`) — there's no flag for
those since they're not something you'd want to fat-finger on the command
line.

## What you get, and what it actually is

- Detector validation — AUROC/precision/recall/F1 per model for
  SelfCheckGPT. This calls the real, installed upstream `selfcheckgpt`
  package — verified against it directly, not a mock.
- Reduction — baseline vs. refined answer, scored with that
  same detector. Every row is stamped `method: self_refine_adapted`,
  `reproduction_status: local_inspired_baseline_NOT_an_upstream_reproduction`
  — this part is my own prompting code, not upstream Self-Refine (their repo
  has no importable API and no QA task). Full citation and reasoning:
  [`docs/MITIGATION_METHODS.md`](MITIGATION_METHODS.md#active-integration-self-refine-adaptation-local-inspired-baseline).

Neither number is publishable evidence on its own — no held-out split,
confidence intervals, or human review yet. They tell you the pipeline works.
