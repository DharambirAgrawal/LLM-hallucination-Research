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

The dry run skips datasets whose local files are missing (and remote
Hugging Face datasets); it lists them for the real run instead of trying
to load them. If Ollama is unavailable, the dry run marks model tags as
unchecked and continues. A run exits with an error if a requested Word report fails,
so a missing report cannot look like a successful experiment.

### 0.5 Updating a machine that is already set up

After new code has been pushed:

```bash
cd LLM-hallucination-Research
git pull
source .venv/bin/activate
pip install -r requirements.txt     # the core environment picks up new packages
```

The MiniCheck / SummaC / AlignScore environments need nothing by hand:
`run_full.py` reinstalls one automatically when its `requirements/<name>.txt`
changed. Make sure Ollama is running, then start with the smoke test (§3).

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

The whole experiment with smaller numbers (2 runs, 2 questions **per
dataset**, 2 samples, 1 Self-Refine round), every detector group, every
reduction method. The default config has 10 enabled datasets and 5 generator
models: that is up to 2,800 generator calls across both runs, plus judge
calls, detector scoring, and preflight. The command prints this workload
before setup starts. It produces exactly the same folders and files as the
full run:

```bash
python scripts/run_full.py --smoke
```

For the same end-to-end coverage with **two questions total**, use:

```bash
python scripts/run_full.py --smoke-2q
```

This selects two seeded questions from the first enabled dataset (currently
HaluEval QA), runs twice, and includes all six detectors, all five configured
reduction methods, and all five selected generator models. It prepares only
that dataset, but first use can still install detector packages and download
their weights; AlignScore's checkpoint is about 1.9 GB. The generator budget
is up to 280 calls across both runs, plus judge calls and preflight. Missing
Ollama model tags are still pulled when `ollama.auto_pull` is true.

For a shorter check of the sampling detectors, use one run and skip the
reduction stage:

```bash
python scripts/run_full.py --smoke --runs 1 --max-samples 1 --n-samples 1 \
  --detectors selfcheckgpt uqlm --no-reduce
```

This still uses every selected generator model and every enabled dataset.

Results go to `results/smoke-<date-time>/` or `results/smoke-2q-<date-time>/`
(or pass `--output
results/<name>`; a folder that already holds runs is refused, so old and new
runs can never mix).

To include the opt-in AlignScore too (its ~1.9 GB checkpoint is downloaded
on first use; it needs Python 3.9–3.11 installed next to your main Python,
because its pinned `torch<2` does not exist for newer Python; `run_full.py`
finds it and says what to install if it is missing):

```bash
python scripts/run_full.py --smoke \
  --detectors selfcheckgpt uqlm uqlm_judge minicheck summac alignscore
```

Check: the final table shows ✓ for every detector group, and
`results/smoke-<date-time>/combined/report.docx` (or `report.html`) opens
with its summary, charts and tables; each `run_01/`, `run_02/` has its own.
A few failures for SelfCheckGPT BERTScore on very short answers are an
upstream limitation, recorded per case; anything else is a problem to look
at in `logs/`. If a check fails, phase 1 stops the run within minutes.

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
third-party warnings go to the log (`logs/<group>.log`). A model or detector that fails 10
cases in a row is stopped early with the reason.

## 4. Full run

```bash
tmux new -s run                        # a multi-hour run should survive a closed terminal
python scripts/run_full.py             # the run: block of config.yaml (3 runs by default)
python scripts/run_full.py --runs 5    # or: 5 independent runs
```

Every run is the same experiment on the same data (same questions, same
detectors, same methods); only the models' own sampling differs, so the
combined report can show how consistent each result is.

Why detector groups: MiniCheck, SummaC and AlignScore pin conflicting
`torch`/`transformers` versions upstream, so each gets its own
`.venv-<name>` (created and reused by `run_full.py`, pip output in
`<output>/logs/`, reinstalled automatically if its requirements changed).
SelfCheckGPT, UQLM and the judge share the generator samples and run
together in the **core** group, which is `requirements.txt` (§0.3) and also
runs the reduction methods; the other groups then score those answers too,
so every detector judges every method. Each group writes its part of every
run into `run_XX/<group>/`, and `run_full.py` then builds each run's report
and the combined one from all groups together.

It runs in two phases:

- **Phase 1 · prepare and check**: every group's environment is installed
  and its preflight runs (download everything, one real question through the
  whole pipeline). If any group fails, the script stops and lists why;
  nothing long has started. Rerun the same command after fixing it;
  environments and downloads are reused.
- **Phase 2 · runs**: only when every group passed: core first (it produces
  the reduction answers), then the others, then the reports.

`alignscore` is opt-in (see §3). If one group fails in phase 2, the others
still run and its line in the final table shows `✗` with the path of its log.

## What a run writes

```text
results/<name>/
  run_01/                  one complete, independent run
    report.docx · report.html · REPORT.md · takeaways.md · charts/ · tables/
    core/                  SelfCheckGPT + UQLM + judge data, and every reduction answer
    minicheck/             MiniCheck's scores (of the labeled answers and the reduction answers)
    summac/                SummaC's scores (same)
  run_02/ … run_N/         the same, for every run
  combined/                ← start here: all runs together (mean ± std, consistency)
    report.docx · report.html · REPORT.md · takeaways.md · charts/ · tables/
    raw_all_runs.csv · reduction_all_runs.csv
  logs/                    core.log · minicheck.log · summac.log · pip-<group>.log
```

**Every report folder** (`run_XX/` and `combined/`) has:

| File | What it holds |
|---|---|
| `report.docx` | the full report as a Word document: run plan, key takeaways, all tables, all charts |
| `report.html` | the same report as one self-contained web page (charts embedded) |
| `REPORT.md` | the same report in Markdown (charts in `charts/`) |
| `takeaways.md` | just the key findings, in plain sentences |
| `charts/*.png` | experiment flow, detector AUROC, matched baseline versus method risk, paired risk changes, one method-comparison chart per generator model, and detailed dataset/detector views |
| `tables/*.csv` | every table in the report |

The report reads top to bottom as a research report and explains that higher
AUROC is better for a detector, while lower risk and a negative method-minus-
baseline change are better for an answer:

1. **Summary**: the answers in plain sentences (observed detector ranking,
   how each method changed measured risk, and whether the data support a
   conclusion).
2. **How the experiment works**: flow diagram, datasets, detectors and
   methods in one line each.
3. **Stage A**: detectors ranked by AUROC (chance and validation lines), one
   compact table, AUROC per dataset and (with several models) per generator
   model. A detector that gives every answer the same score is flagged as
   "no signal" and never used as a judge.
4. **Stage B**: matched baseline-versus-method bars and numbers, paired risk
   changes, per-model comparisons, and an example answer per method. The
   primary detector passes the Stage A AUROC screen on a sufficiently sized
   run. If the screen fails, or the run has fewer than 10 distinct questions,
   results are labeled exploratory. Confidence intervals resample questions
   with their model answers kept together and require at least 10 distinct
   questions and 10 question × model pairs.
5. **Reliability**: run-to-run variation, failures by cause.
6. **Appendix**: exactly what ran (models, versions, commit), Stage B as
   seen by every detector, AUROC per run, and every data file.

The full detail (every answer, every score, every table) is in the CSV
files, not in the document.

**A `combined/` folder** reports every metric as mean ± std over the runs,
and adds `tables/*_by_run.csv` (each run's numbers side by side),
`tables/*_mean_std.csv`, `raw_all_runs.csv` and `reduction_all_runs.csv`
(every row of every run, with a `run` column).

**Each `run_XX/<group>/` folder** keeps that group's raw data for the run:

| File | What it holds |
|---|---|
| `detector_validation_summary.csv` | AUROC, AUPRC, accuracy, precision, recall, F1, `n_cases`, `n_failed` per detector (per model for SelfCheckGPT and UQLM) |
| `detector_validation_raw.csv` | every labeled answer: dataset, question, context, answer, label, every detector's score, errors |
| `reduction_comparison.csv` | (core) one row per question × model × method: the answer, every detector's score, generation calls, seconds, method details (feedback, verification questions, chosen candidate), `reproduction_status`, errors |
| `reduction_scores.csv` | (minicheck / summac / alignscore) their scores of core's reduction answers, joined into the reports automatically |
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

### GPU memory: how models are loaded

Ollama and the detectors share the GPU. Models are used **one at a time**:
each generator is loaded when it is first asked, and unloaded
(`keep_alive=0`) as soon as its part is done, in the preflight, the detector
validation and the reduction stage. So at most one generator, plus the judge
(`mistral:7b`) and the detectors' torch models (~6 GB in total), share the
GPU at any moment.

**Which device `device: auto` picks.** On a GPU with less than 16 GB (e.g. an
8 GB RTX 5060), the detectors (~6 GB) and a 4–5 GB Ollama model do not fit
together, so the GPU is left to Ollama and the detectors run on the CPU. The
detector process then does not see the GPU at all, so no library can take GPU
memory by itself. Ollama keeps the whole GPU and puts what does not fit on
the CPU by itself (`ollama ps` shows e.g. `20%/80% CPU/GPU`). The Detectors
section of the terminal says which was chosen and why. The scores are the
same; the detectors are slower, and the preflight's time estimate includes
that. On 16 GB or more the detectors use the GPU. `--device cuda` or
`--device cpu` overrides the choice.

`gpt-oss:20b` needs ~13 GB by itself; the other generators need 2–5 GB. If
memory still runs out while a model loads, the run recovers on its own instead
of failing:

1. **An Ollama model does not fit** ("model failed to load … resource
   limitations"): the other models loaded in Ollama are unloaded, the torch
   detectors are moved to the CPU for the rest of the run, and the request is
   retried.
2. **A detector runs out of GPU memory** ("CUDA out of memory"): the torch
   cache is freed and the case retried; if it fails again, that detector
   is reloaded on the CPU for the rest of the run.

Each move is logged as a warning, and the device used is recorded in
`config_used.yaml`. The scores are the same on the CPU; only slower. If a model
still does not fit, the error says so; check `nvidia-smi` for other programs
using the GPU, or replace the model with a smaller one. To keep the detectors
off the GPU from the start, run with `--device cpu` (the default below 16 GB).

While a run is going you can watch memory with `nvidia-smi` and see which
models Ollama holds with `ollama ps`.

## Sending the errors when something fails

The GPU machine does not need a git account for this. After a failed (or
strange) run, from the project folder:

```bash
python scripts/share_logs.py --upload
```

It writes `debug_report.txt` (git commit, GPU and Ollama status, the config,
every distinct error with a count, the last tracebacks, failed cases per CSV,
the end of each log) and uploads it to paste.rs. It then prints a short link
such as `https://paste.rs/AbC1`: send that link, and the report can be read
from any computer. Without `--upload` nothing leaves the machine. The report
contains no API keys or `.env` content and hides the home folder path, but
anyone with the link can read it. Add a results folder
(`python scripts/share_logs.py results/<run> --upload`) to report an older run
instead of the newest.

## Useful flags

| Flag | `main.py` | `run_full.py` | What it does |
|---|---|---|---|
| `--smoke` | ✓ | ✓ | Quick test: 2 runs, 2 questions per dataset, 2 samples, 1 refine round |
| `--smoke-2q` | ✓ | ✓ | 2 questions total, 2 runs, every detector and configured reducer |
| `--runs N` | ✓ | ✓ | Independent runs (`run.runs`) |
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
