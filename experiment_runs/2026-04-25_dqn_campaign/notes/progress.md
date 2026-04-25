# DQN Campaign Progress

## Goal
Run rigorous DQN training across easy, medium, and hard modes and save clean evidence for final presentation.

## Current Status
- Easy DQN 1000-episode run completed
- Medium DQN 1000-episode run completed
- Hard DQN 1000-episode run completed

## Long-Run Results

### Easy | DQN | 1000 episodes | seed 42
- Reward: early=27.332 -> late=42.526
- Improving: True
- Avg satisfied at ep1000 block: 8.89/10
- Avg complaints at ep1000 block: 0.00
- HP satisfaction at ep1000 block: 1.94

### Medium | DQN | 1000 episodes | seed 42
- Reward: early=43.584 -> late=55.117
- Improving: True
- Avg satisfied at ep1000 block: 8.88/10
- Avg complaints at ep1000 block: 8.98
- HP satisfaction at ep1000 block: 2.93

### Hard | DQN | 1000 episodes | seed 42
- Reward: early=14.636 -> late=28.571
- Improving: True
- Avg satisfied at ep1000 block: 7.38/10
- Avg complaints at ep1000 block: 40.67
- HP satisfaction at ep1000 block: 3.24

## Plot Artifacts
- plots/easy_dqn_ep1000_progress.png
- plots/medium_dqn_ep1000_progress.png
- plots/hard_dqn_ep1000_progress.png
- plots/dqn_medium_hard_ep300_progress.png

## Interpretation
- DQN shows clear long-run learning on all three modes.
- Medium is the strongest overall result.
- Easy is the cleanest stable regime.
- Hard remains the toughest benchmark, but it now also shows strong improvement.
