# Post-Training Execution Sheet

## Completed
- Preference-distillation run: complete
- Grouped preference-distillation run: complete
- GRPO-lite run: complete
- Four-way evaluation: complete

## Current Results
- Base reward_mean: 10.927852
- Preference-Distilled reward_mean: 13.008463
- Grouped Preference-Distilled reward_mean: 13.008463
- GRPO-lite reward_mean: -1.628150

## Interpretation
- Preference distillation is the strongest post-training result.
- Grouped preference distillation matches the same reward but does not exceed it.
- GRPO-lite proves the environment supports online-style post-training runs, but this setup underperforms.
- EnergyMind now clearly distinguishes stronger and weaker post-training methods.

## Recommendation
- Keep Preference-Distilled as the main post-training result.
- Treat Grouped Preference-Distilled as a matching variant, not a superior one.
- Treat GRPO-lite as an honest negative result that shows the benchmark is non-trivial.
- Use HF credits next only for larger, better-structured final runs.
