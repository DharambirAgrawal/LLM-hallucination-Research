# Reduction-method decision record

One local-inspired baseline is active: a generic-QA adaptation of Self-Refine
(`reducers/self_refine.py`). It is **not** an upstream reproduction — see the
"Active integration" section below before reporting any result from it. No
other candidate is integrated. This distinction (local inspired baseline vs.
upstream reproduction vs. provider baseline) is intentional: the previous
context prompting, sampling, and self-critique files were local baselines and
were removed because they could be mistaken for verified implementations of
published methods.

## Candidates from verified sources

| Method | Verified source | Integration decision |
|---|---|---|
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

Current scope limits: only SelfCheckGPT can score the before/after pair
today (`benchmark/reduction_runner.py` rejects any other detector), and the
comparison is a paired smoke run — not the bootstrap-CI, held-out, human-
reviewed Stage B protocol described in `docs/REPRODUCIBILITY.md`.

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
