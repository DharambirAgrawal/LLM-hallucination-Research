# Google Colab: official detector smoke tests

The development computer does not need an LLM or detector checkpoint. Colab is
the evaluator machine and its temporary disk holds any downloaded weights.

## 1. Clone the GitHub repository

Replace the URL only if you use a fork or another branch.

```python
!git clone https://github.com/DharambirAgrawal/LLM-hallucination-Research.git
%cd LLM-hallucination-Research
```

## 2. First smoke test: SelfCheckGPT n-gram

This uses the official pinned SelfCheckGPT package and replayed sample responses,
so it does not download or call an LLM. The package still installs its Python
dependencies in Colab.

```python
!pip install -q -r requirements-colab-smoke.txt
!python official_smoke.py selfcheckgpt
```

This is an integration check, not a research result. The n-gram score is
unbounded and its threshold must be calibrated on a labeled validation split.

## 3. MiniCheck smoke test

For this step, choose a GPU runtime (`Runtime > Change runtime type > T4 GPU`).
MiniCheck automatically downloads its official Flan-T5-Large checkpoint into
Colab on first use.

```python
!pip install -q -r requirements-colab-minicheck.txt
!python official_smoke.py minicheck
```

`official_smoke.py` now downloads both required NLTK sentence-tokenizer
resources before running MiniCheck. If the repository has not yet been updated
and you see `Resource punkt_tab not found`, run this repair cell, then rerun the
smoke test. The already downloaded 3.13 GB MiniCheck checkpoint should be reused
from the Colab cache.

```python
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")
!python official_smoke.py minicheck
```

## 4. Full labeled validation

Enable only detectors installed in the current Colab environment:

```python
!python main.py --detectors minicheck --output results/colab-minicheck
```

The default synthetic pairs only test the plumbing. For reportable results,
run `!python scripts/prepare_halueval.py` (fetches `qa_data.json`,
`dialogue_data.json`, `summarization_data.json` at the revision pinned in
`provenance/sources.yaml`; `halueval_qa`/`halueval_dialogue`/
`halueval_summarization` are already enabled in `config.yaml`), disable the
synthetic dataset, and use a train/validation/test threshold protocol.

## 5. Test with a small model running locally in Colab (no API key)

Choose a GPU runtime (`Runtime > Change runtime type > T4 GPU`). This example
uses the official Apache-2.0
[`Qwen/Qwen2.5-0.5B-Instruct`](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct)
checkpoint pinned to the immutable revision recorded in
`config.local-colab.example.yaml`. The model is downloaded into Colab and runs
there; nothing is downloaded onto the development computer.

If the repository is already open in Colab, update it first:

```python
%cd /content/LLM-hallucination-Research
!git pull
```

Install the pinned official SelfCheckGPT package and run one factual/
hallucinated pair. SelfCheckGPT draws two responses per case from the local
Qwen model, so this needs no API secret:

```python
!pip install -q -r requirements-colab-smoke.txt
!python main.py \
  --config config.local-colab.example.yaml \
  --detectors selfcheckgpt \
  --output results/local-qwen-smoke
```

The first run downloads about 1 GB of Qwen weights. Messages about Hugging Face
authentication or tied weights are warnings, not failures. A real failure is a
nonzero command exit followed by `All enabled detectors failed`.

Inspect the generated scores and metric summary:

```python
import pandas as pd

raw = pd.read_csv("results/local-qwen-smoke/detector_validation_raw.csv")
display(raw[["case_id", "label", "selfcheckgpt_score", "selfcheckgpt_error"]])

from pathlib import Path
summary_path = Path("results/local-qwen-smoke/detector_validation_summary.csv")
if summary_path.exists():
    display(pd.read_csv(summary_path))
else:
    print("No summary was created. Read selfcheckgpt_error above for the cause.")
```

If a previous checkout produced a blank SelfCheckGPT error, update the
repository and rerun the command. The adapter now follows Qwen's official
generation form by passing both `input_ids` and `attention_mask`, and the runner
preserves a traceback even for exceptions whose message is empty.

This local model is only the **generator used by SelfCheckGPT**. MiniCheck,
SummaC, and AlignScore are separate detector models and must be tested with
their own official packages/checkpoints.

## 6. What “test everything” currently means

Run the repository-wide plumbing tests without downloading additional model
weights:

```python
!python -m unittest discover -s tests -v
```

Then run the available official runtime checks separately:

| Component | Colab command | What it verifies |
|---|---|---|
| SelfCheckGPT n-gram + replay | `!python official_smoke.py selfcheckgpt` | Official scorer import and scoring |
| SelfCheckGPT + local Qwen | `!python main.py --config config.local-colab.example.yaml --detectors selfcheckgpt` | Local generation, scoring, CSV and metrics |
| MiniCheck | `!python official_smoke.py minicheck` | Official Flan-T5-Large detector |
| SummaC | Separate legacy environment | Official SummaC runtime compatibility |
| AlignScore | Separate legacy environment plus checkpoint | Official AlignScore runtime compatibility |
| Reduction (local inspired baseline) | `!python main.py --config config.local-colab.example.yaml --detectors selfcheckgpt --reduce` | Self-Refine-adapted loop, NOT an upstream reproduction — see `docs/MITIGATION_METHODS.md`. Self-RAG and RARR remain reference-only candidates, not integrated. |

Do not run all neural detectors in one Colab environment: their pinned upstream
dependencies conflict, and loading every checkpoint together can exhaust RAM or
GPU memory. “All tests passed” should distinguish offline adapter tests from
actual official checkpoint runs.

## 7. Connect a model running elsewhere

Only SelfCheckGPT requires repeated generations. Copy the model block from
`config.remote.example.yaml` into a new Colab config, then set the endpoint URL.
For an authenticated endpoint, put the key in a Colab secret/environment
variable and set `api_key_env`; never paste the key into YAML or commit it.

```python
import os
from google.colab import userdata
os.environ["RESEARCH_MODEL_API_KEY"] = userdata.get("RESEARCH_MODEL_API_KEY")
```

```python
!python main.py --config config.colab.yaml --detectors selfcheckgpt --output results/colab-selfcheckgpt
```

The remote endpoint must be reachable from Colab. A private LAN address is not
reachable unless you deliberately expose it through a secured network route.

## 8. Why SummaC and AlignScore are separate

Do not install every detector into one Colab environment. SummaC and AlignScore
were published against older PyTorch/Transformers stacks; AlignScore also needs
an explicit upstream checkpoint and spaCy model. Use method-specific Colab
notebooks/environments following the upstream READMEs and exact revisions in
`provenance/sources.yaml`, then run:

```python
!python official_smoke.py summac
!python official_smoke.py alignscore --checkpoint /content/AlignScore-base.ckpt
```

Record the Python, Torch, Transformers, CUDA, package commit, and checkpoint
hash for every successful research run.

## 9. Free-tier API sampling

Add `GEMINI_API_KEY` to Colab Secrets and load it without printing it:

```python
import os
from google.colab import userdata
os.environ["GEMINI_API_KEY"] = userdata.get("GEMINI_API_KEY")
```

After step 2 has installed the pinned package, run both tested Gemini free-tier
models with two samples per case (eight short API calls total):

```python
!python3 scripts/provider_selfcheck_smoke.py
```

The command also supports `--providers openrouter` after loading an
`OPENROUTER_API_KEY`, and `--providers mistral` after loading a
`MISTRAL_API_KEY`. OpenRouter selects its no-cost router; Mistral availability
depends on Studio account activation. Provider failures are recorded without
stopping another selected provider.

For the full synthetic plumbing run, copy `config.gemini.example.yaml`, then:

```python
!python main.py --config config.gemini.example.yaml --detectors selfcheckgpt
```

Do not report tiny-fixture scores as detector performance. Use held-out labeled
data and calibrate the n-gram threshold on the validation split.
