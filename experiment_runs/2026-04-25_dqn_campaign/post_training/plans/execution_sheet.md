# Post-Training Execution Sheet

## Status
- Preference-distillation run: complete
- First post-training evaluation: complete
- Base vs Preference-Distilled comparison: complete
- Grouped preference and GRPO-lite adapters: not yet trained

## First Result
- Base reward_mean: 10.927852
- Preference-Distilled reward_mean: 13.008463
- Gain: 2.080611
- hp_sat_mean preserved at 3.0
- valid_rate preserved at 1.0

## Interpretation
The first post-training run successfully improved reward over the base model without sacrificing validity or high-priority satisfaction.

## Next Priority
1. Train grouped preference-distilled model
2. Evaluate again
3. Train GRPO-lite model
4. Build final comparison pack
