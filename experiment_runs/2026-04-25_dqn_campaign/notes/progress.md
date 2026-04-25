# DQN Campaign Progress

## Goal
Run rigorous DQN training across easy, medium, and hard modes and save clean evidence for final presentation.

## Current Status
- Medium DQN 300-episode run completed
- Hard DQN 300-episode run completed
- Medium DQN 1000-episode run completed
- Hard DQN 1000-episode run completed

## Results Log

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
- plots/dqn_medium_hard_ep300_progress.png
- plots/medium_dqn_ep1000_progress.png
- plots/hard_dqn_ep1000_progress.png

## Interpretation
- Medium DQN is the strongest result so far.
- Hard DQN also shows clear long-run learning.
- The benchmark now has a strong story across both medium and hard difficulty.

## Next Planned Runs
- Easy | DQN | 1000 episodes
- Final comparison table across easy, medium, hard
