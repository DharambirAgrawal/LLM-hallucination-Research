# Reduction-method decision record

Five reduction methods are compared with the grounded baseline answer (see
[`METHOD_SOURCES.md`](../METHOD_SOURCES.md#reduction-stage-b) for the full
table). What each one is, exactly:

- `uqlm_best_response`: **official** UQLM implementation.
- `self_refine_adapted`: **local inspired adaptation** of Self-Refine (below).
- `cove_adapted`: **local implementation** of Chain-of-Verification; the
  paper released no code ([`reducers/cove.py`](../reducers/cove.py)).
- `closed_book`: an ablation (no context), measuring what RAG adds.
- `greedy`: a decoding setting (temperature 0), not a published method.

Every output row carries this in its `reproduction_status` column, so a
results table can never present a local implementation as a reproduction.

## Candidates from verified sources

| Method | Verified source | Integration decision |
|---|---|---|
| Chain-of-Verification | Paper only (Dhuliawala et al., 2023); no code released | **Integrated as a local implementation** of the paper's factored 4-step method. |
| UQLM best-response selection | `cvs-health/uqlm` v0.6.6, Apache-2.0 | **Integrated, official implementation** (`SemanticEntropy(use_best=True)`). |
| Self-Refine | `madaan/self-refine`, Apache-2.0, pinned in the source register | **Integrated as a local inspired baseline**, not an upstream reproduction — see below. The upstream code is task-specific; adapting it into a generic QA loop changes prompts and behavior. |
| Self-RAG | `AkariAsai/self-rag`, published implementation | Reference only. It requires its trained model/checkpoints and special reflection tokens; it cannot be plugged into an arbitrary API model. |
| RARR | `anthonywchen/RARR`, pinned in the source register | Do not vendor. No license is declared at the repository root, and the original workflow depends on search/API infrastructure. |
| AWS Bedrock contextual grounding | Official AWS service and sample repository | Valid provider baseline when AWS credentials and cost approval exist. Report it as a managed-service baseline, not repository code. |
| OpenAI Guardrails hallucination detection | Official OpenAI package | Detection baseline, not a reduction method. It should not appear in a reduction table. |

## Active integration: Self-Refine adaptation (local inspired baseline)

- Adapter: [`reducers/self_refine.py`](../reducers/self_refine.py)
- Orchestration: [`benchmark/reduction_runner.py`](../benchmark/reduction_runner.py)
- Provenance record: `self_refine` in [`provenance/sources.yaml`](../provenance/sources.yaml)
- Paper: [Madaan et al., 2023](https://arxiv.org/abs/2303.17651)

What it does: for each sample, a generator model answers once (baseline),
then the same model is asked for feedback against the supplied context and
revises its answer, repeating until it reports no remaining unsupported
claims or a configured iteration budget (`reduction.max_iterations`) is
spent. The already-configured, already-frozen SelfCheckGPT detector scores
both the baseline and the revised answer so the two are directly comparable.

Why it is not "Self-Refine" without qualification: the official repository's
prompts and harnesses are task-specific (math, code, dialogue, acronym
generation, ...) and include no grounded-QA/hallucination-reduction task.
Only the paper's generate → feedback → refine control loop is reused here;
every prompt is written locally for this harness and is not copied from the
upstream repository. Per `docs/REPRODUCIBILITY.md` Stage C, report this as a
**local inspired baseline**, a separate condition from any future upstream
reproduction, provider baseline, or the project's own proposed method.

Scope: every enabled detector scores every method's answer (MiniCheck,
SummaC and AlignScore from their own environments, see
`scripts/run_full.py`); the comparison is paired per question and model, but
it is not yet the bootstrap-CI, held-out, human-reviewed Stage B protocol
described in `docs/REPRODUCIBILITY.md`.

## Research sequence

1. Validate official detectors on held-out labeled responses.
2. Freeze detector versions, checkpoints, thresholds, and environments.
3. Add one upstream reduction method in its own supported environment.
4. Keep the upstream condition unchanged; label any prompt/API port as an
   adaptation.
5. Compare correctness, groundedness, abstention, latency, and cost using paired
   examples and confidence intervals.

This prevents a locally written prompt from being described as RAG,
Self-Refine, constrained decoding, or another published algorithm.
