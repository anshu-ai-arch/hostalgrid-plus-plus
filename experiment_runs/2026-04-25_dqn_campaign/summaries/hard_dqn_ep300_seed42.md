# Hard DQN Summary

- Mode: hard
- Agent: centralized DQN
- Episodes: 300
- Seed: 42

## Training Output
- Reward early: 7.354
- Reward late: 7.243
- Improving: False
- Avg satisfied (last block): 6.92/10
- Avg complaints (last block): 67.07
- HP satisfaction (last block): 3.05
- Loss (last block): 0.1461
- Epsilon: 0.3005
- Buffer: 15000
- Steps: 13001

## Takeaway
Hard mode is running correctly, but the learning signal is not yet stable at 300 episodes. This suggests hard mode likely needs longer training or tuning for stronger final performance.
