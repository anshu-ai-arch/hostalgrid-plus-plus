# DQN Campaign Progress

## Goal
Run rigorous DQN training across easy, medium, and hard modes and save clean evidence for final presentation.

## Current Status
- Campaign folder created
- Medium DQN 300-episode run completed
- Hard DQN 300-episode run completed
- Medium DQN 1000-episode run completed
- Comparison plot generated

## Results Log

### Medium | DQN | 300 episodes | seed 42
- Reward: early=33.669 -> late=38.509
- Improving: True
- Avg satisfied at ep300 block: 7.89/10
- Avg complaints at ep300 block: 20.37
- HP satisfaction at ep300 block: 2.56

### Hard | DQN | 300 episodes | seed 42
- Reward: early=7.354 -> late=7.243
- Improving: False
- Avg satisfied at ep300 block: 6.92/10
- Avg complaints at ep300 block: 67.07
- HP satisfaction at ep300 block: 3.05

### Medium | DQN | 1000 episodes | seed 42
- Reward: early=43.584 -> late=55.117
- Improving: True
- Avg satisfied at ep1000 block: 8.88/10
- Avg complaints at ep1000 block: 8.98
- HP satisfaction at ep1000 block: 2.93
- Loss at ep1000 block: 0.2978
- Epsilon: 0.0200
- Buffer: 20000
- Steps: 48501

## Plot Artifacts
- plots/dqn_medium_hard_ep300_progress.png

## Interpretation
- Medium DQN is now a strong benchmark result.
- Hard DQN is still the main challenge.
- The next best rigorous step is hard-mode long training.

## Next Planned Runs
- Hard | DQN | 1000 episodes
- Easy | DQN | 1000 episodes
