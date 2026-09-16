# How to run

## 0. First-time setup on a new Linux machine

Needs: `python3`, `pip`, `git`, and Ollama already installed with the models
listed in `config.yaml`'s `selected_models` pulled (`ollama pull llama3.2:3b`,
etc. — check with `ollama list`).

If `python3` is present but `pip` is not (common on Debian/Ubuntu, which ships
Python without pip):

```bash
sudo apt update && sudo apt install -y python3-pip python3-venv   # Debian/Ubuntu
# or: sudo dnf install -y python3-pip                             # Fedora/RHEL
```

If you don't have `sudo`, `python3 -m ensurepip --upgrade` works on most
distributions, but Debian/Ubuntu's system Python often ships that module
disabled — the `apt install` above is the reliable path there.

Then, from the repository root, create an isolated environment and install
the controller plus the pinned official SelfCheckGPT package (this is the
only detector wired into the multi-model + reduction pipeline below; see
[§3](#3-adding-the-other-detectors-minicheck--summac--alignscore) for why
MiniCheck/SummaC/AlignScore are separate):

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements-colab-smoke.txt
```

Confirm Ollama is reachable before running anything:

```bash
ollama list                # models pulled locally
curl -s http://localhost:11434/api/tags   # Ollama API is up
```

`config.yaml`'s `ollama.host` already points at `http://localhost:11434`,
which is correct when Ollama runs on the same Linux machine as this repo.

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
