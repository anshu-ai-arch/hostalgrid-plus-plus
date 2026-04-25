# Post-Training Execution Sheet

## Completed
- Preference-distillation run: complete
- GRPO-lite run: complete
- Three-way evaluation: complete

## Current Results
- Base reward_mean: 10.927852
- Preference-Distilled reward_mean: 13.008463
- GRPO-lite reward_mean: -1.628150

## Interpretation
- Preference distillation is currently the strongest post-training result.
- GRPO-lite proves the environment supports online-style post-training runs, but this setup underperformed and needs tuning.
- The benchmark now shows that post-training methods can be meaningfully compared rather than automatically succeeding.

## Next Priority
1. Save and push these artifacts
2. Decide whether to tune GRPO-lite or attempt grouped preference distillation
3. Prepare final judge-facing comparison table and plot pack
