# Post-Training Three-Way Summary

## Models Compared
- Base Qwen 0.5B
- Preference-Distilled
- GRPO-lite

## Results
- Base reward_mean: 10.927852
- Preference-Distilled reward_mean: 13.008463
- GRPO-lite reward_mean: -1.628150

## Stability
- Base hp_sat_mean: 3.0
- Preference-Distilled hp_sat_mean: 3.0
- GRPO-lite hp_sat_mean: 1.0

- Base valid_rate: 1.0
- Preference-Distilled valid_rate: 1.0
- GRPO-lite valid_rate: 1.0

## Interpretation
Preference distillation improved reward over the base model while preserving validity and high-priority satisfaction. GRPO-lite successfully ran in the environment but underperformed badly, showing that online post-training in this benchmark is non-trivial and requires stronger tuning.
