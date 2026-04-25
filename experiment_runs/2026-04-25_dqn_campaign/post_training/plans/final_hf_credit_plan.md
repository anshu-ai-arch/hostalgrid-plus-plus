# Final HF Credit Plan

## Current Best Model
- Preference-Distilled

## Goal
Use HF credits only on runs that can beat the current best post-training result.

## Run A: Stronger Offline Run
- Base: stronger instruct model than current 0.5B if budget allows
- Method: preference distillation
- Data: current preference dataset + grouped preference dataset
- Keep only if reward_mean beats current Preference-Distilled baseline

## Run B: Stronger Online Run
- Method: GRPO-style rollout training
- Use more rollout states
- Use more samples per state
- Use more update steps
- Keep only if it beats base and approaches offline result

## Run C: Held-Out Evaluation
- Compare:
  - Base
  - Preference-Distilled
  - Best Offline Final
  - Best Online Final
- Save final table, summary, and plot

## Decision Rule
- Current local winner remains Preference-Distilled until a new run clearly beats it.
