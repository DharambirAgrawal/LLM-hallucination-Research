# Takeaways (Combined Average)

Generated: 2026-04-28T03:20:44
Runs: 3

This file is generated from `combined/summary.csv` and `combined/reductions.csv`.
All numbers are computed from the benchmark outputs (no fabricated results).

## Overall mean scores (lower = better)

| reducer | token_score | semantic_score | bert_score | llm_score | mean_latency_s |
| --- | --- | --- | --- | --- | --- |
| baseline | 0.6656 | 0.1112 | 0.6015 | 0.0267 | 1.4850 |
| constrained_decoding | 0.6989 | 0.1094 | 0.6631 | 0.0300 | 1.5270 |
| rag | 0.4502 | 0.0734 | 0.5189 | 0.0333 | 1.0740 |
| self_verification | 0.6808 | 0.1045 | 0.6294 | 0.0933 | 3.8480 |

## Overall mean reduction vs baseline (higher = better)

| reducer | token_score_reduction | semantic_score_reduction | bert_score_reduction | llm_score_reduction |
| --- | --- | --- | --- | --- |
| constrained_decoding | -0.0333 | 0.0018 | -0.0616 | -0.0033 |
| rag | 0.2154 | 0.0378 | 0.0826 | -0.0067 |
| self_verification | -0.0152 | 0.0067 | -0.0280 | -0.0667 |

## Best reducer per model (by mean score across detectors)

Mean score is the simple average of the available detector scores in this run folder.

| model | reducer | mean_score | baseline_mean_score | score_delta_vs_baseline | mean_latency_s | baseline_mean_latency_s | latency_delta_s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| llama3:latest | rag | 0.2690 | 0.3512 | 0.0822 | 1.0740 | 1.4850 | -0.4110 |

## Best reducer per model (by mean reduction vs baseline)

| model | reducer | mean_reduction |
| --- | --- | --- |
| llama3:latest | rag | 0.0823 |
