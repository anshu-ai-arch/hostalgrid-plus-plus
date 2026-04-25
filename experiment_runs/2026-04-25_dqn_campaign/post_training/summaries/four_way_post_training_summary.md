# Four-Way Post-Training Summary

## Models Compared
- Base Qwen 0.5B
- Preference-Distilled
- Grouped Preference-Distilled
- GRPO-lite

## Results
- Base reward_mean: 10.927852
- Preference-Distilled reward_mean: 13.008463
- Grouped Preference-Distilled reward_mean: 13.008463
- GRPO-lite reward_mean: -1.628150

## Stability
- Base hp_sat_mean: 3.0
- Preference-Distilled hp_sat_mean: 3.0
- Grouped Preference-Distilled hp_sat_mean: 3.0
- GRPO-lite hp_sat_mean: 1.0

- Base valid_rate: 1.0
- Preference-Distilled valid_rate: 1.0
- Grouped Preference-Distilled valid_rate: 1.0
- GRPO-lite valid_rate: 1.0

## Interpretation
Preference distillation is the strongest post-training result. Grouped preference distillation matches the same level but does not improve beyond it. GRPO-lite runs successfully but underperforms badly, showing that online post-training in EnergyMind remains challenging and non-trivial.
