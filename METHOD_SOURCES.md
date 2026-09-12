# Method Sources and Reproducibility

This document records the published methods and public implementations that motivate
this benchmark. It also states exactly what the current repository implements. A
method is not described as an upstream implementation unless its upstream code and
algorithm are actually integrated and pinned.

## Status key

- **Implemented locally**: the repository contains its own implementation of the
  described idea. This is a reimplementation, not copied upstream source code.
- **Reference only**: the paper or repository is a candidate source, but is not yet
  integrated into the benchmark.
- **Adapter required**: the upstream implementation uses a runtime or model interface
  that is different from Ollama.

## Reduction methods

### Retrieval-Augmented Generation

- Paper: [Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401)
- Reference implementation: [Hugging Face RAG documentation](https://huggingface.co/docs/transformers/model_doc/rag)
- Alternative retrieval framework: [Haystack](https://github.com/deepset-ai/haystack)
- Current file: [reducers/rag.py](reducers/rag.py)
- Current status: **Implemented locally, context-prompting only**
- Important difference: the current reducer receives a reference passage from the
  dataset. It does not retrieve documents, rank passages, or build a vector index.
  Results must therefore be described as a context-grounding baseline until a real
  retriever is added.

### Self-Refine / Self-Verification

- Paper: [Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651)
- Reference implementation: [madaan/self-refine](https://github.com/madaan/self-refine)
- License reported by GitHub: Apache-2.0
- Current file: [reducers/self_verification.py](reducers/self_verification.py)
- Current status: **Legacy local baseline; retained for comparison**
- Difference from Self-Refine: the current code performs one critique prompt and
  returns either the original response or the raw verification response. It does not
  implement the complete structured feedback-and-revision loop from the paper.
- Required adapter work: connect the reference refinement loop to the project model
  interface and ensure the final output contains only the revised answer.

- Ollama adapter: [reducers/self_refine.py](reducers/self_refine.py)
- Adapter status: **Runnable published-algorithm adapter**
- Configuration: `reducers.self_refine` in [config.yaml](config.yaml)
- The adapter uses the paper's generate-feedback-revise stages. The upstream
  repository does not provide a generic pip package, so the Ollama calls are
  necessarily local integration code rather than copied upstream source.

### Restricted sampling baseline

- Sampling reference: [The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751)
- Current file: [reducers/constrained_decoding.py](reducers/constrained_decoding.py)
- Current status: **Implemented locally; restricted-sampling baseline**
- Difference from formal constrained decoding: temperature, top-p, and top-k narrow
  the sampling distribution, but do not impose a grammar or token constraint.
- Candidate implementations for actual constrained generation:
  - [Outlines](https://github.com/dottxt-ai/outlines), Apache-2.0
  - [LM Format Enforcer](https://github.com/noamgat/lm-format-enforcer), MIT
- Runtime warning: these libraries generally require direct access to a compatible
  Hugging Face or vLLM generation loop. An Ollama adapter must be verified before
  claiming that either library powers the experiment.

### Self-RAG

- Paper: [Self-RAG: Learning to Retrieve, Generate, and Critique through Self-Reflection](https://arxiv.org/abs/2310.11511)
- Reference implementation: [AkariAsai/self-rag](https://github.com/AkariAsai/self-rag)
- License reported by GitHub: MIT
- Current status: **Reference only; not integrated**
- Runtime warning: Self-RAG uses trained checkpoints and special reflection tokens.
  It is not equivalent to adding a verification prompt to an arbitrary Ollama model.

## Detection and evaluation methods

### AlignScore

- Paper: [AlignScore: Evaluating Factuality with a Unified Textual Entailment Model](https://arxiv.org/abs/2305.07035)
- Reference implementation: [yuh-zha/AlignScore](https://github.com/yuh-zha/AlignScore)
- License reported by GitHub: MIT
- Current status: **Integrated as an optional adapter; disabled by default**
- Adapter file: [detectors/alignscore_detector.py](detectors/alignscore_detector.py)
- Configuration: `detectors.alignscore` in [config.yaml](config.yaml)
- Recommended use: evaluate whether an answer is entailed by the retrieved evidence,
  rather than treating lexical similarity as proof of factuality.

### SummaC

- Paper: [SummaC: Re-Visiting NLI-based Models for Inconsistency Detection in Summarization](https://arxiv.org/abs/2111.09525)
- Reference implementation: [tingofurro/summac](https://github.com/tingofurro/summac)
- License reported by GitHub: Apache-2.0
- Current status: **Integrated through a thin local adapter; disabled by default**
- Adapter file: [detectors/summac_detector.py](detectors/summac_detector.py)
- Configuration: `detectors.summac` in [config.yaml](config.yaml)
- Recommended use: an additional NLI-based entailment/contradiction detector with
  calibration on the chosen benchmark. The adapter uses the official SummaC-ZS
  path by default; SummaC-Conv remains selectable with `model_name: "vitc"` in
  compatible environments.
- Environment note: the tested Python 3.14/Torch runtime segfaulted inside the
  upstream model, so enable this only in a compatible Python 3.10-3.12 ML
  environment after testing the package independently.

### Current local detectors

- [detectors/token_detector.py](detectors/token_detector.py): BLEU, ROUGE-L, and token
  overlap heuristic. **Implemented locally; baseline metric, not ground truth.**
- [detectors/semantic_detector.py](detectors/semantic_detector.py): embedding cosine
  similarity heuristic. **Implemented locally; baseline metric, not an entailment model.**
- [detectors/llm_detector.py](detectors/llm_detector.py): prompted model judge.
  **Implemented locally; judge results require validation and agreement analysis.**
- [detectors/bert_detector.py](detectors/bert_detector.py): stochastic resampling and
  BERTScore agreement. **Implemented locally; consistency signal, not factuality proof.**

## Dataset and domain sources

- HaluEval paper: [arXiv:2305.11747](https://arxiv.org/abs/2305.11747)
- HaluEval dataset: [pminervini/HaluEval](https://huggingface.co/datasets/pminervini/HaluEval)
- AWS example motivating the original detector discussion: [Detect hallucinations for RAG-based systems](https://aws.amazon.com/blogs/machine-learning/detect-hallucinations-for-rag-based-systems/)

AWS documentation and official, version-pinned GitHub repositories may be used as
source evidence for a new domain dataset. They are evidence sources, not automatic
hallucination labels; every evaluation sample still needs a verified answer,
evidence span, source URL, and source version or commit.

## Reproducibility requirements

For every external method added to the benchmark, record:

- paper title, DOI or arXiv URL, and publication year;
- upstream repository URL, license, commit SHA, and dependency version;
- model checkpoint and tokenizer;
- local adapter files and every behavior-changing modification;
- prompt, decoding parameters, dataset version, and random seed;
- whether results come from the upstream runtime or an Ollama adapter.

The benchmark must report upstream reproductions and the proposed method as separate
conditions. Local adapters may connect components, but they must not silently change
the algorithm being claimed.
