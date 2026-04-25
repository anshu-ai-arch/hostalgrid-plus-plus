# Campaign Snapshot 1

## Completed Artifacts
- Medium DQN 300 summary saved
- Hard DQN 300 summary saved
- Medium vs Hard 300 comparison plot saved
- Medium DQN 1000 summary saved
- Medium DQN 1000 progress plot saved

## Best Current Result
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

## Saved Plot Artifacts
- plots/dqn_medium_hard_ep300_progress.png
- plots/medium_dqn_ep1000_progress.png

## Current Interpretation
- Medium-mode DQN is now a strong result.
- It shows clear reward improvement and complaint reduction across long training.
- Hard-mode 1000-episode training is the current next rigorous run.

## Current Status
- Medium 1000: completed and saved
- Hard 1000: running / pending final summary
