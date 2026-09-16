# How to run

## 0. First-time setup on a new Linux machine

Needs `python3`, `pip`, `git`, and a separately installed, already-running
Ollama with the models listed in `config.yaml`'s `selected_models` pulled.
**`pip install` only installs Python packages** — it does not install
Ollama, pull models, or fetch the real HaluEval data. Those are separate
steps, done once, below.

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

No `sudo`? `python3 -m ensurepip --upgrade` works on most distributions, but
Debian/Ubuntu's system Python often ships that module disabled — the `apt
install` above is the reliable path there.

### 0.3 Create a venv and install the Python dependencies

This is the "controller" environment used for the smoke test and the
`scripts/run_full.py` orchestrator below. It includes the pinned official
SelfCheckGPT package (the only detector wired into the multi-model +
reduction pipeline; see
[§3](#3-adding-the-other-detectors-minicheck--summac--alignscore) for why
MiniCheck/SummaC/AlignScore get their own separate venvs instead):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-colab-smoke.txt
```

This step covers every Python package this repo needs to run detector
validation + reduction with SelfCheckGPT. It does **not** cover the next two
steps — both are required and neither is a `pip install`:

### 0.4 Confirm Ollama is installed, running, and has the models

Ollama itself is a separate system install, not a Python package — install
it however your distro/`ollama.com` docs say to, independently of this repo,
before continuing. Then pull whatever `config.yaml`'s `selected_models`
lists (`ollama pull llama3.2:3b`, etc.) and confirm:

```bash
ollama list                                # models pulled locally
curl -s http://localhost:11434/api/tags    # Ollama API is up
```

`config.yaml`'s `ollama.host` already points at `http://localhost:11434`,
which is correct when Ollama runs on the same Linux machine as this repo.

### 0.5 Fetch real data (one-time, not part of `pip install`)

```bash
python scripts/prepare_halueval.py
```

This downloads the official HaluEval QA, dialogue, and summarization files
(three different task types, ~58 MB total) to `external_data/HaluEval/data/`.
`config.yaml` already has `halueval_qa`, `halueval_dialogue`, and
`halueval_summarization` enabled and pointed at these files, alongside the
small `synthetic` fixture. Skip this only if you deliberately want to test
against the tiny synthetic-only fixture (disable the three `halueval_*`
entries in `config.yaml` first, or every run will log a "Failed to load"
warning per missing file and just fall back to `synthetic`).

### 0.6 You're set — jump to §1

Everything else below (§1 smoke test, §4 `run_full.py`) just runs from here.
`run_full.py` creates its own additional venvs for MiniCheck/SummaC/
AlignScore automatically the first time you use them — you don't need to
`pip install` those by hand unless you're running one directly per §3.

## 1. Smoke test first — always

Tiny data, all your models, both stages. Run it, check nothing errored, then
run it again (should give the same shape of output):

```bash
python main.py --detectors selfcheckgpt --reduce \
  --max-samples 2 --n-samples 2 --max-iterations 1 \
  --output results/smoke-test
```

Check: `results/smoke-test/detector_validation_summary.csv` has one row per
model, and `reduction_comparison.csv` has no `error` values. If that's clean,
move to the full run. If not, you just found the problem in seconds instead
of partway through a run that could take hours.

## 2. Full run

Same command, without the size overrides — uses whatever `max_samples` and
`n_samples` are actually set in `config.yaml`:

```bash
python main.py --detectors selfcheckgpt --reduce --output results/full-run
```

Runs every model in `config.yaml`'s `selected_models`, one after another.

## 3. Adding the other detectors (MiniCheck / SummaC / AlignScore)

`--detectors` accepts more than one name
(`--detectors selfcheckgpt minicheck`), but only SelfCheckGPT needs a
generator, feeds the reduction stage, and installs from
`requirements-colab-smoke.txt`. The other three:

- score already-fixed context/answer pairs (no generator, no `--reduce`
  support — `main.py` hard-errors if `reduction.detector` is anything but
  `selfcheckgpt`);
- each need their own upstream package, pinned separately because their git
  dependencies fix different, conflicting `torch`/`transformers` versions:
  - MiniCheck: `pip install -r requirements-colab-minicheck.txt`
  - SummaC: `pip install summac` (isolated legacy environment; see
    `provenance/sources.yaml`)
  - AlignScore: see `METHOD_SOURCES.md` — needs a downloaded checkpoint too.

Do not install all four into the same virtualenv — that's not a limitation
in this repo's code, it's the actual upstream packages conflicting. Validate
each one in its own venv, e.g.:

```bash
python3 -m venv .venv-minicheck
source .venv-minicheck/bin/activate
pip install -r requirements-colab-minicheck.txt
python main.py --detectors minicheck --max-samples 5 --output results/minicheck-smoke
deactivate
```

Then compare each detector's `detector_validation_summary.csv` side by side —
you don't need them in one process to get one report.

## 4. One command: every detector + reduction + a combined report

`scripts/run_full.py` automates exactly what §3 does by hand: it creates (and
reuses) one venv per detector, runs `main.py` in each, and then runs
`scripts/generate_report.py` to merge every detector's
`detector_validation_summary.csv` and SelfCheckGPT's `reduction_comparison.csv`
into one folder with a combined CSV, PNG charts, and a `REPORT.md`. It needs
only the base controller environment (step 0) — each detector's own
dependencies go into that detector's own child venv, not into the interpreter
running the script.

5-sample, 2-turn smoke version — checks the whole pipeline end to end fast.
`--max-samples 5` applies per dataset (§0's `prepare_halueval.py` step gives
you 4 enabled datasets: HaluEval QA, dialogue, summarization, plus the small
synthetic fixture), so this is 5 QA + 5 dialogue + 5 summarization + 5
synthetic samples per model — real variety, not one dataset repeated:

```bash
python scripts/run_full.py --max-samples 5 --n-samples 2 --max-iterations 2
```

Full-size version, every detector installed so far, using `config.yaml`'s own
sample sizes (50 per HaluEval dataset, 8 synthetic):

```bash
python scripts/run_full.py --detectors selfcheckgpt minicheck summac
```

`--detectors` defaults to `selfcheckgpt minicheck summac`. `alignscore` is
opt-in (`--detectors selfcheckgpt minicheck summac alignscore`) because it
also needs a checkpoint downloaded first — the script checks for
`external_models/alignscore/AlignScore-base.ckpt` and skips it with a message
if that file isn't there yet; see `METHOD_SOURCES.md`.

Output layout:

```text
results/full-run-<timestamp>/
  selfcheckgpt/detector_validation_summary.csv, detector_validation_raw.csv, reduction_comparison.csv
  minicheck/detector_validation_summary.csv, detector_validation_raw.csv
  summac/detector_validation_summary.csv, detector_validation_raw.csv
  combined_summary.csv                 all detectors, one table (every dataset blended)
  combined_per_dataset_breakdown.csv   same metrics, broken out per dataset (QA/dialogue/summarization/synthetic)
  combined_reduction.csv               SelfCheckGPT baseline vs. refined, every row (no per-run size limit)
  combined_reduction_summary.csv       baseline/refined means + win rate, by dataset x model
  charts/*.png                         detector comparison, per-model, per-dataset breakdown,
                                        reduction by model (2-panel: scores + win rate),
                                        reduction by dataset (2-panel: scores + win rate)
  REPORT.md                            tables + embedded charts, one file to read or link from docs
```

The blended `combined_summary.csv` answers "how did each detector do
overall"; `combined_per_dataset_breakdown.csv` and its chart answer "how did
each detector do on QA vs. dialogue vs. summarization" — that per-dataset
split is recomputed from each run's own `detector_validation_raw.csv`, not
something `main.py` itself writes. Reduction's `win_rate` (fraction of
samples that actually scored better after refining) is a clearer improvement
signal than the mean score alone when only some samples flip — both live in
`combined_reduction_summary.csv` and the two reduction charts. `REPORT.md`
embeds the two summary tables, not the row-level `combined_reduction.csv` —
that file can have thousands of rows on a big-batch run and isn't meant to be
read inline. Latency is recorded in `combined_reduction.csv` but not
surfaced in any chart or summary table — this report is about output
quality, not runtime.

If one detector's run fails (a real dependency or checkpoint problem, not a
bug in this script), `run_full.py` logs it and keeps going with the rest —
check the printed `failed:` list at the end and that detector's own directory
for the traceback.

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
| `--max-samples N` | Cap every dataset to N samples (use for the smoke test) |
| `--n-samples N` | SelfCheckGPT generations per case (use for the smoke test) |
| `--max-iterations N` | Reduction feedback/refine steps (use for the smoke test) |
| `--device cpu\|cuda` | Device for SelfCheckGPT/SummaC/AlignScore's torch models |
| `--detectors selfcheckgpt ...` | Which detector(s) to run |
| `--reduce` | Also run the reduction stage |
| `--output DIR` | Where results are written |

Which model(s) run and which dataset(s) are enabled still come from
`config.yaml` (`selected_models`, `datasets[].enabled`) — there's no flag for
those since they're not something you'd want to fat-finger on the command
line.

## What you get, and what it actually is

- `detector_validation_summary.csv` — AUROC/precision/recall/F1 per model for
  SelfCheckGPT. This calls the real, installed upstream `selfcheckgpt`
  package — verified against it directly, not a mock.
- `reduction_comparison.csv` — baseline vs. refined answer, scored with that
  same detector. Every row is stamped `method: self_refine_adapted`,
  `reproduction_status: local_inspired_baseline_NOT_an_upstream_reproduction`
  — this part is my own prompting code, not upstream Self-Refine (their repo
  has no importable API and no QA task). Full citation and reasoning:
  [`docs/MITIGATION_METHODS.md`](MITIGATION_METHODS.md#active-integration-self-refine-adaptation-local-inspired-baseline).

Neither number is publishable evidence on its own — no held-out split,
confidence intervals, or human review yet. They tell you the pipeline works.
