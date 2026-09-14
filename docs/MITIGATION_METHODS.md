# Reduction-method decision record

The active code currently contains **no hallucination-reduction algorithm**.
This is intentional: the previous context prompting, sampling, and self-critique
files were local baselines and cannot be claimed as verified implementations of
published methods.

## Candidates from verified sources

| Method | Verified source | Integration decision |
|---|---|---|
| Self-Refine | `madaan/self-refine`, Apache-2.0, pinned in the source register | Reference only. The upstream code is task-specific; adapting it into a generic QA loop changes prompts and behavior and must be reported as an adaptation. |
| Self-RAG | `AkariAsai/self-rag`, published implementation | Reference only. It requires its trained model/checkpoints and special reflection tokens; it cannot be plugged into an arbitrary API model. |
| RARR | `anthonywchen/RARR`, pinned in the source register | Do not vendor. No license is declared at the repository root, and the original workflow depends on search/API infrastructure. |
| AWS Bedrock contextual grounding | Official AWS service and sample repository | Valid provider baseline when AWS credentials and cost approval exist. Report it as a managed-service baseline, not repository code. |
| OpenAI Guardrails hallucination detection | Official OpenAI package | Detection baseline, not a reduction method. It should not appear in a reduction table. |

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
