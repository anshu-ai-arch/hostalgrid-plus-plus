# Hard DQN Summary

- Mode: hard
- Agent: centralized DQN
- Episodes: 1000
- Seed: 42

## Checkpoints
- Ep100: reward=14.636, sat=7.11/10, complaints=59.78, hp=3.50, loss=0.0214, eps=0.670
- Ep200: reward=16.424, sat=7.08/10, complaints=58.39, hp=3.35, loss=0.0819, eps=0.449
- Ep300: reward=17.599, sat=7.36/10, complaints=57.23, hp=3.36, loss=0.1319, eps=0.300
- Ep400: reward=20.998, sat=7.64/10, complaints=52.01, hp=3.44, loss=0.1728, eps=0.201
- Ep500: reward=22.496, sat=7.55/10, complaints=47.93, hp=3.39, loss=0.2095, eps=0.135
- Ep600: reward=25.515, sat=7.16/10, complaints=45.32, hp=3.03, loss=0.2457, eps=0.090
- Ep700: reward=27.428, sat=7.64/10, complaints=41.80, hp=3.33, loss=0.2741, eps=0.060
- Ep800: reward=28.943, sat=7.51/10, complaints=42.71, hp=3.26, loss=0.3076, eps=0.041
- Ep900: reward=28.863, sat=7.17/10, complaints=42.56, hp=3.01, loss=0.3166, eps=0.027
- Ep1000: reward=28.571, sat=7.38/10, complaints=40.67, hp=3.24, loss=0.3227, eps=0.020

## Final
- Reward early: 14.636
- Reward late: 28.571
- Improving: True
- Buffer: 25000
- Steps: 48001

## Takeaway
Hard-mode DQN now shows clear long-run learning. Reward nearly doubled from the early to late stage, while complaints dropped substantially across training.
