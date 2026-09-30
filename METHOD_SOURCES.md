# Method sources and status

Rule of this project: every detector and every dataset runs from its
**official, citable source**, pinned to an exact commit, release or dataset
revision. The code here is only glue (adapters, orchestration, reports). The
machine-readable register is [`provenance/sources.yaml`](provenance/sources.yaml);
BibTeX for every paper is in [`CITATION.bib`](CITATION.bib).

Where no official code exists, the method is implemented from the paper and
labeled as a local implementation, in the code, in every CSV row
(`reproduction_status`), and in the reports.

## Detection (Stage A)

Every score column is oriented the same way: **higher = more likely
hallucinated**.

| Family | Scores | Official source | What we add |
|---|---|---|---|
| SelfCheckGPT | `selfcheckgpt_ngram`, `_bertscore`, `_nli`, `_prompt` | [potsawee/selfcheckgpt](https://github.com/potsawee/selfcheckgpt), MIT, pinned commit | adapter: [`detectors/selfcheckgpt_detector.py`](detectors/selfcheckgpt_detector.py) |
| UQLM consistency | `uqlm_semantic_negentropy`, `_noncontradiction`, `_entailment`, `_cosine_sim`, `_exact_match`, `_bert_score` | [cvs-health/uqlm](https://github.com/cvs-health/uqlm) v0.6.6, Apache-2.0 | adapter: [`detectors/uqlm_detector.py`](detectors/uqlm_detector.py) |
| UQLM LLM-as-a-judge | `uqlm_judge` | same UQLM release | same adapter |
| MiniCheck | `minicheck` | [Liyan06/MiniCheck](https://github.com/Liyan06/MiniCheck), Apache-2.0 | [`detectors/minicheck_detector.py`](detectors/minicheck_detector.py) |
| SummaC | `summac` | [tingofurro/summac](https://github.com/tingofurro/summac), Apache-2.0 | [`detectors/summac_detector.py`](detectors/summac_detector.py) |
| AlignScore | `alignscore` | [yuh-zha/AlignScore](https://github.com/yuh-zha/AlignScore), MIT | [`detectors/alignscore_detector.py`](detectors/alignscore_detector.py) |

### SelfCheckGPT (Manakul et al., EMNLP 2023)

- Every scorer the official package ships, on the **same** sampled answers:
  `SelfCheckNgram(n=1)`, `SelfCheckBERTScore(rescale_with_baseline=True)`,
  `SelfCheckNLI` (the paper's strongest non-LLM variant) and
  `SelfCheckAPIPrompt` (the official LLM-prompt variant: "Is the sentence
  supported by the context above? Answer Yes or No."). The prompt scorer's
  judge is a local Ollama model reached through Ollama's OpenAI-compatible
  endpoint. `SelfCheckMQAG` is not used (it needs its own question-generation
  models and is not one of the paper's recommended variants).
- Responses are split into sentences with spaCy `en_core_web_sm`, exactly as
  in the upstream README; the response score is the mean of the sentence
  scores. Upstream values are reported as returned (n-gram is an unbounded
  negative log-probability, not a probability).
- Samples: N answers from the generator to the grounded prompt, drawn once
  per model and question and shared by every sampling-based detector
  ([`detectors/sampling.py`](detectors/sampling.py)); archived in
  `selfcheckgpt_samples.jsonl`.
- **What this measures differs by stage.** In Stage B (reduction) the
  checked answer comes from the same model and the same prompt as the
  samples: the paper's self-consistency setup. In Stage A the checked
  answers are the datasets' fixed answers (written by HaluEval's
  generator, RAGTruth's six LLMs, HaluBench), not by our model, so there the
  sampling-based detectors measure agreement of a given answer with our
  model's own answers (cross-model consistency). Stage A therefore
  validates these detectors as reference-free checkers of a given answer,
  which is how they are used on the reduction outputs; it is not a
  reproduction of the paper's self-consistency evaluation.
- `SelfCheckNLI` needs `sentencepiece`, which the package does not declare
  (found by running it; added to `requirements.txt`).
- Upstream `SelfCheckBERTScore` drops sample sentences of 3 tokens or fewer
  and raises `IndexError` when none remain (short answers, e.g. HaluEval QA,
  DROP). Each scorer fails on its own; the failure is recorded per case and
  counted per dataset in the report.

### UQLM (Bouchard et al., 2025)

- `BlackBoxUQ(...).score(responses, sampled_responses)`: semantic negentropy
  (normalized semantic entropy, Farquhar et al., Nature 2024),
  non-contradiction and entailment probabilities (NLI), cosine similarity,
  exact match, BERTScore. The scored answer is the labeled answer; the
  sampled responses are the shared samples above.
- **`use_best=False` is required.** With UQLM's default (`True`) the answer
  is replaced by the "best" sample *before* the consistency scorers run, so
  they score a different answer than the labeled one. Verified on uqlm
  0.6.6: for a hallucinated answer the default reported non-contradiction
  0.67 / entailment 0.50 (numbers of a correct sample); with `use_best=False`
  0.003 / 0.0003.
- `exact_match` is UQLM's short-answer scorer; on multi-sentence answers it
  almost never matches. It is kept (official, cheap) and shows up near chance
  in the results.
- `bert_score` (unrescaled F1, typically 0.8–0.95) and `cosine_sim`
  (`0.5 + cos/2`) give risks that rarely reach 0.5, so their
  threshold-dependent metrics (accuracy, precision, recall, F1) are
  degenerate at the default threshold; AUROC/AUPRC are unaffected.
- `semantic_negentropy` uses UQLM's own clustering rule (answers grouped when
  the NLI model finds entailment in either direction), adapted from, not
  identical to, Farquhar et al.'s bidirectional-entailment clustering.
- `LLMJudge` with its default `true_false_uncertain` template (Chen & Mueller,
  2023). The judged "question" is the grounded prompt (context + question),
  so the judge checks the answer against the context. Judge: `judge.model`
  in the config (a different model family than every generator).
- UQLM returns confidence; every value is reported as `1 − confidence`.
- Generator-independent detectors (the UQLM judge, MiniCheck, SummaC,
  AlignScore) score each fixed answer once and later runs reuse those
  scores, so their run-to-run std is 0 by construction (the judge runs at
  temperature 0).

### MiniCheck (Tang et al., EMNLP 2024)

- Upstream `MiniCheck.score(docs, claims)` on each response sentence; the
  response risk is `1 − min(sentence support)` (local aggregation policy,
  declared here, not a number from the paper).
- Upstream calls `nltk.sent_tokenize`; the run downloads the NLTK `punkt`
  data first.

### SummaC (Laban et al., TACL 2022)

- Upstream `SummaCConv` (or `SummaCZS`) `.score([document], [claim])`; risk =
  `1 − consistency`. Pinned to the reviewed commit (the PyPI release is older).
- SummaC-Conv's trained weights: upstream `start_file="default"` downloads
  them with `wget` from the `master` branch; instead the run downloads the
  same file at the pinned commit and checks its SHA-256 (byte-identical to
  `master` at review time). SummaC-ZS scores (entailment − contradiction, in
  [−1, 1]) are mapped to [0, 1] linearly, not clipped.

### AlignScore (Zha et al., ACL 2023)

- Upstream `AlignScore(..., evaluation_mode="nli_sp").score(...)`; risk =
  `1 − alignment`. Checkpoint `AlignScore-base.ckpt` from the authors'
  Hugging Face repo at a pinned revision, SHA-256 verified.

### Not used, and why

- The AWS blog "Detect hallucinations for RAG-based systems" describes four
  methods but publishes no code. Its BERT stochastic checker is SelfCheckGPT's
  BERTScore variant and its LLM prompt detector corresponds to the LLM-judge
  scorers above (both run from official code here). Its token- and
  embedding-similarity detectors have no official implementation and are not
  used. (Earlier versions of this repository had local implementations of all
  four that compared answers with the reference answer instead of the context;
  they were removed.)
- The AWS workshop repository's detection labs are Bedrock/SageMaker
  notebooks, not an importable package; the methods they demonstrate
  (semantic similarity, non-contradiction, semantic entropy) are covered by
  UQLM above.

## Reduction (Stage B)

Every method answers the same questions with the same models; each answer is
compared with the model's own grounded baseline answer to the same question,
scored by every detector against the same samples.

| Condition | What it is | Source | Status label in the CSVs |
|---|---|---|---|
| `baseline` | grounded answer: context in the prompt (the RAG setting), default sampling | — | `reference_condition_grounded_answer` |
| `closed_book` | same question without the context; the gap to `baseline` is what retrieval-augmented generation adds (dataset passage = oracle retrieval) | RAG, Lewis et al. 2020 (concept, no code needed) | `rag_ablation_…` |
| `greedy` | grounded answer at temperature 0 | decoding setting, not a paper method | `decoding_setting_…` |
| `self_refine_adapted` | generate → feedback → refine loop | Self-Refine, Madaan et al. 2023; official code is task-specific, so this is a **local inspired adaptation** ([`reducers/self_refine.py`](reducers/self_refine.py)) | `local_inspired_baseline_NOT_an_upstream_reproduction` |
| `cove_adapted` | Chain-of-Verification, factored: plan verification questions, answer each separately without the draft, revise (self-verification) | Dhuliawala et al. 2023; **no official code exists**, implemented from the paper ([`reducers/cove.py`](reducers/cove.py)) | `local_implementation_of_published_method_no_official_code` |
| `uqlm_best_response` | pick the answer from the most probable meaning cluster among the baseline + its samples | UQLM `SemanticEntropy(use_best=True)`, **official** ([`reducers/uqlm_best_response.py`](reducers/uqlm_best_response.py)) | `official_uqlm_implementation` |

Notes on `uqlm_best_response`: within the winning cluster UQLM returns the
most repeated answer, otherwise the **longest** one (a length bias). Our
prompts are not passed to its NLI step (with them, every answer fell into
one cluster). The picked answer is scored **leave-one-out**: against the
baseline + the samples minus every exact copy of itself, so it is never
compared with itself (its evidence can be smaller than the other methods').

All other methods' answers are scored against the same samples as the
baseline, and every answer in Stage B comes from the same model and prompt
family as those samples.

Candidates that were not integrated: [Self-RAG](https://github.com/AkariAsai/self-rag)
needs its own trained model and reflection tokens; [RARR](https://github.com/anthonywchen/RARR)
declares no license and needs a search API; AWS Bedrock contextual grounding
is a paid cloud service. See [`docs/MITIGATION_METHODS.md`](docs/MITIGATION_METHODS.md).

## Data

| Dataset | Used as | Official source | License |
|---|---|---|---|
| HaluEval QA / dialogue / summarization | one correct + one hallucinated answer per question | [RUCAIBox/HaluEval](https://github.com/RUCAIBox/HaluEval), pinned commit | MIT |
| RAGTruth QA / Summary / Data2txt | official **test** split: real answers from 6 LLMs per question, human span labels (hallucinated = at least one span) | [ParticleMedia/RAGTruth](https://github.com/ParticleMedia/RAGTruth), pinned commit | MIT |
| HaluBench DROP / FinanceBench / covidQA / pubmedQA | PASS/FAIL-labeled answers (its HaluEval/RAGTruth parts are taken from their original sources instead). 272 DROP FAIL answers are Python list literals while no PASS answer is, so detectors could tell them apart by format; answers of that form are excluded | [PatronusAI/HaluBench](https://huggingface.co/datasets/PatronusAI/HaluBench), pinned revision | CC-BY-NC-2.0 (research use) |
| synthetic | 8 hand-written pairs; plumbing tests only, disabled in the full run | this repository | — |

Every file is downloaded by the run itself from its pinned source and
checked against a recorded SHA-256 ([`utils/resources.py`](utils/resources.py)).
Each run's `run_manifest.json` records the checksums it used.
