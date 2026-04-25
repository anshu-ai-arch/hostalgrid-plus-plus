# Tuned GRPO-lite Result Summary

## Evaluation Result
- Base Qwen 0.5B reward_mean: 10.927852
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
The tuned GRPO-lite run completed successfully but still failed to improve over the base model. This shows that online closed-loop post-training in EnergyMind is non-trivial and likely requires stronger rollout quality, better reward shaping, or larger compute.
