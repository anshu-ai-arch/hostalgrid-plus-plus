# Hybrid Hard-Mode Summary

## Models Compared
- Base LLM + DQN Executor
- Post-trained LLM + DQN Executor

## Results
- Base reward_mean: 10.945822
- Post-trained reward_mean: 11.120332
- Reward gain: 0.174510

## Safety / Priority
- Base hp_sat_mean: 3.666667
- Post-trained hp_sat_mean: 4.000000
- Base valid_rate: 1.0
- Post-trained valid_rate: 1.0

## Interpretation
Post-training the high-level LLM planner improved the full hybrid planner-executor system on the hardest benchmark mode. The post-trained planner produced better hard-mode reward while also improving high-priority satisfaction and preserving full validity.
