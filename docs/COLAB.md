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

## 4. Full labeled validation

Enable only detectors installed in the current Colab environment:

```python
!python main.py --detectors minicheck --output results/colab-minicheck
```

The default synthetic pairs only test the plumbing. For reportable results,
prepare the official HaluEval data at the revision in
`provenance/sources.yaml`, enable `halueval_qa` in `config.yaml`, disable the
synthetic dataset, and use a train/validation/test threshold protocol.

## 5. Connect a model running elsewhere

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

## Why SummaC and AlignScore are separate

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

## Free-tier API sampling

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
