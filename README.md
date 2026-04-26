---
title: EnergyMind
emoji: ⚡
colorFrom: blue
colorTo: green
sdk: docker
app_port: 7860
pinned: false
license: mit
short_description: OpenEnv-compatible multi-agent energy governance environment with RL baselines and LLM post-training.
---

# EnergyMind: Training AI Systems Under Real-World Constraints

EnergyMind is an OpenEnv-compatible multi-agent governance environment where AI systems learn to allocate scarce electricity across competing human actors under delayed consequences, fairness pressure, and hard resource limits.

**Themes:** Multi-Agent Interactions, Long-Horizon Planning  
**Core result:** reward-informed post-training improves decision quality in a constrained human-aware environment.

---

## Submission Links

- **Hugging Face Space:** [https://huggingface.co/spaces/anshu-123/hostalgrid-plus-plus]
- **GitHub Repository:** [https://github.com/anshu-ai-arch/hostalgrid-plus-plus]
- **Training Notebook (Colab):** [https://colab.research.google.com/drive/1-nJ155ZiovbS_b5jzGITs489I8Wxpl89#scrollTo=dOZqMOZDHru2]
- **Hugging Face Blog / Writeup:** [https://huggingface.co/spaces/anshu-123/hostalgrid-plus-plus/blob/main/BLOG.md](#)
- **youtube Video:** [https://youtu.be/VZy5kQ5VUN0?si=yNcitynDQ11geCJi](#)
- **Plots/Training Curves:** [https://github.com/anshu-ai-arch/hostalgrid-plus-plus/tree/main/experiment_runs/2026-04-25_dqn_campaign](#)

---

## TL;DR

- We built an OpenEnv-compatible environment for energy governance in a shared hostel with multiple human actors, delayed complaints, and hard power constraints.
- We found that direct low-level LLM control is brittle, so we redesigned the system into a hybrid architecture: **LLM planner + learned low-level executor + safe execution layer**.
- Reward-informed post-training measurably improved performance, and the repo includes reproducible scripts, evaluation artifacts, and a public Space.

---

## Problem

Most AI training environments optimize one clean objective.

Real systems do not.

EnergyMind models a shared hostel where electricity has to be allocated under:
- limited power budgets
- different room priorities
- changing occupancy
- delayed complaint buildup
- crisis conditions such as heatwaves
- fairness pressure across multiple people

This makes the task a governance problem, not just an optimization problem.

The agent is not simply trying to “save power.” It has to decide **who gets served, when, under what constraints, and at what long-term cost**.

---

## Why This Environment Matters

EnergyMind is designed to teach capabilities that matter in real-world AI systems:

- reasoning under scarcity
- balancing competing human needs
- handling delayed consequences
- maintaining fairness under pressure
- recovering from bad short-term choices

This makes it a strong fit for:

- **Theme #1: Multi-Agent Interactions**  
  because one system must manage multiple human actors with different needs and priorities

- **Theme #2: Long-Horizon Planning**  
  because bad early choices come back later as complaints, neglected urgent rooms, and budget stress

---

## Environment Design

EnergyMind is a 10-room hostel environment with:

- dynamic room occupancy
- room priorities
- complaint accumulation over time
- easy / medium / hard modes
- hard power-budget constraints
- measurable rewards and normalized scores

Every action changes the future state of the environment.

### OpenEnv interface

EnergyMind exposes the standard environment flow:

- `reset()`
- `step(action)`
- `state()`
- `score()`

The official benchmark score is normalized to **[0, 1]**.

---

## Observation, Action, Reward

## Observation

The agent observes:

- room occupancy
- room priority
- complaint level
- current appliance state
- simulated hour
- heatwave status
- total power used
- current power budget

The structured observation is defined in:
- `env/observation.py`

## Action

### Canonical environment action
The low-level controller uses:

- `room_actions[10]`
- one action per room
- each room action is an integer `0..7`

These correspond to appliance combinations such as:
- `all_off`
- `fan_only`
- `ac_only`
- `all_on`

### High-level planner action
For the LLM lane, we use a higher-level strategy space:

- `priority_safe`
- `complaint_rescue`
- `energy_saver`
- `comfort_all`
- `shutdown_empty`
- `do_nothing`

The LLM chooses a strategy, and a safe executor converts it into valid room-level actions.

<img width="1047" height="463" alt="image" src="https://github.com/user-attachments/assets/4d54188b-8955-444f-bc96-f0b9af4a57b3" />


## Reward

The raw environment reward is shaped for learning and captures:

- satisfaction of occupied rooms
- high-priority room service
- complaint reduction
- budget-aware behavior
- avoidance of waste

The official final evaluation score is normalized to **[0,1]** through the environment scoring layer.

---

## Key Insight

Our first design let the LLM directly produce low-level room actions.

That failed.

Even when outputs looked valid, direct appliance-level control was brittle, unstable, and unsafe. This exposed an important gap between:
- generating structured outputs
- making good decisions under constraints

So we redesigned the system into a hybrid architecture:

- **LLM for high-level planning**
- **learned low-level executor for actual control**
- **safe execution layer for reliability**
- **environment reward for accountability**

That turned out to be the right decomposition.

---

## System Architecture
<img width="1048" height="495" alt="image" src="https://github.com/user-attachments/assets/323b68fd-2d24-4bb1-94b7-a489afbb21b9" />

EnergyMind uses a hybrid control stack:

- **LLM planner** chooses a high-level governance strategy
- **Q-learning or centralized DQN executor** handles low-level room-level control
- **safe executor / correction layer** prevents unsafe or invalid control decisions
- **environment reward** provides learning signal and final evaluation

This gives us:

- interpretability at the planning level
- robust execution at the control level
- reproducible training evidence
- a clear path for closed-loop post-training

---

## Training Pipeline

EnergyMind supports both baseline RL training and LLM post-training.

### RL baselines

We trained and evaluated:

- heuristic controller
- tabular Q-learning
- centralized DQN

These baselines show that the environment has real learning signal and is not solved by trivial hard-coded behavior.

### LLM lane

For the LLM system, we built:

- a strategy planner prompt interface
- a safe strategy-to-action executor
- reward-derived preference generation
- offline preference distillation
- grouped rollout generation
- GRPO-lite online-style updates

### Main scripts

- `training/build_strategy_preferences.py`
- `training/train_preference_distill.py`
- `training/build_grpo_rollouts.py`
- `training/train_grpo_lite.py`
- `training/evaluate_post_training.py`

---

## Results

## Main before/after result

| Model | Reward | Complaints | HP Satisfaction | Valid Rate |
|---|---:|---:|---:|---:|
| Base Qwen 0.5B | 10.93 | 7.33 | 3.0 | 1.0 |
| Preference-Distilled | 13.01 | 0.0 | 3.0 | 1.0 |
| Grouped Preference-Distilled | 13.01 | 0.0 | 3.0 | 1.0 |
| GRPO-lite | 11.35 | 1.67 | 3.0 | 1.0 |

### What changed

The main improvement is not just formatting or output validity.

The trained model makes **better decisions in the environment**:
- higher total reward
- far fewer complaints
- preserved service for high-priority rooms
- stable valid output rate

### Stronger run

We also validated the hybrid system with a stronger post-training setup on **Qwen 1.5B** across **1000+ episodes**, which further strengthened confidence that the environment supports serious post-training rather than only small demo runs.

---

## Training Evidence

### Post Training curve for Base LLM+DQN & Trained LLM+DQN
![https://github.com/anshu-ai-arch/hostalgrid-plus-plus/blob/main/experiment_runs/2026-04-25_dqn_campaign/hybrid/hybrid_hard_reward_comparison.png]
### Loss Curve
![https://github.com/anshu-ai-arch/hostalgrid-plus-plus/blob/main/experiment_runs/2026-04-25_dqn_campaign/post_training/plots/hf_merged_prefdistill_training_loss.png]

### Baseline vs Trained Comparison
![https://github.com/anshu-ai-arch/hostalgrid-plus-plus/blob/main/experiment_runs/2026-04-25_dqn_campaign/post_training/plots/post_training_three_way_reward.png]

### Post Training reward comparison
![https://github.com/anshu-ai-arch/hostalgrid-plus-plus/blob/main/experiment_runs/2026-04-25_dqn_campaign/post_training/plots/prefdistill_vs_base_reward.png]

---

## Closed-Loop Post-Training Evidence

EnergyMind is not just an evaluation benchmark.

It is a training environment.

In our closed-loop setup:

1. the model acts inside the environment
2. the environment returns reward
3. reward generates learning signal
4. the model is updated
5. behavior improves on evaluation

We demonstrate this through:

- reward-informed preference post-training
- grouped environment rollouts
- GRPO-lite online-style updates
- before/after evaluation artifacts
- hybrid planner + executor improvements in actual environment behavior

This is the central reason EnergyMind is useful for LLM training research.

---

## Safeguards and Reward Hacking Prevention

We explicitly designed the system to reduce reward hacking and unsafe control behavior.

Key safeguards:

- the LLM does **not** directly control appliances
- high-level planner output is constrained to a bounded strategy set
- a safe executor converts strategy to valid room-level actions
- invalid or malformed outputs fall back safely
- reward measures multiple dimensions, not a single easily-gameable target
- budget behavior, complaints, and high-priority service are all tracked
- official scoring is normalized and auditable

This makes it much harder for the model to get high reward by exploiting superficial shortcuts.

---

## Reproducibility

 Install runtime dependencies

Install training dependencies
pip install -r requirements-train.txt
Run the environment app
uvicorn app:app --host 0.0.0.0 --port 7860
Run the main post-training pipeline
python training/build_strategy_preferences.py
python training/train_preference_distill.py
python training/build_grpo_rollouts.py
python training/train_grpo_lite.py
python training/evaluate_post_training.py
Run the notebook
Open and execute:

EnergyMind_OpenEnv_TRL_Pipeline.ipynb

pip install -r requirements.txt

