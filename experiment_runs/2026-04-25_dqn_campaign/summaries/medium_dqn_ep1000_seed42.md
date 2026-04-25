# Medium DQN Summary

- Mode: medium
- Agent: centralized DQN
- Episodes: 1000
- Seed: 42

## Checkpoints
- Ep100: reward=43.584, sat=7.86/10, complaints=18.92, hp=2.61, loss=0.0307, eps=0.670
- Ep200: reward=44.793, sat=8.20/10, complaints=16.86, hp=2.78, loss=0.1053, eps=0.449
- Ep300: reward=46.215, sat=8.35/10, complaints=15.53, hp=2.84, loss=0.1658, eps=0.300
- Ep400: reward=49.195, sat=8.29/10, complaints=11.78, hp=2.72, loss=0.2106, eps=0.201
- Ep500: reward=50.424, sat=8.55/10, complaints=11.38, hp=2.80, loss=0.2415, eps=0.135
- Ep600: reward=52.212, sat=9.03/10, complaints=9.71, hp=2.91, loss=0.2618, eps=0.090
- Ep700: reward=53.405, sat=8.79/10, complaints=11.30, hp=2.86, loss=0.2835, eps=0.060
- Ep800: reward=54.874, sat=9.10/10, complaints=9.12, hp=2.94, loss=0.2936, eps=0.041
- Ep900: reward=56.463, sat=9.19/10, complaints=8.25, hp=2.98, loss=0.2945, eps=0.027
- Ep1000: reward=55.117, sat=8.88/10, complaints=8.98, hp=2.93, loss=0.2978, eps=0.020

## Final
- Reward early: 43.584
- Reward late: 55.117
- Improving: True
- Buffer: 20000
- Steps: 48501

## Takeaway
Medium-mode DQN shows clear long-run learning. Reward improved substantially, complaints dropped, and satisfaction stayed high through 1000 episodes.
