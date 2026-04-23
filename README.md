# EnergyMind: Training AI Systems Under Real-World Constraints

## Overview

EnergyMind (HostelGrid++) is an OpenEnv-compatible environment designed to study how AI systems behave when optimization is not enough.

Instead of maximizing a single objective, the system forces an agent to operate under real-world constraints: limited resources, competing user needs, delayed feedback, and long-term consequences.

The setup models a shared hostel where electricity must be allocated across multiple rooms with different priorities, occupancy patterns, and complaint dynamics.

This turns the problem into decision-making under constraints, not simple optimization.

## What We Built

We developed a multi-actor, long-horizon environment with:

- multiple rooms and occupants
- high-priority vs normal users
- limited power budget
- delayed complaint accumulation
- easy / medium / hard modes
- measurable rewards and episode outcomes

This is not a static simulator. Every decision affects future system state.

## The Core Problem

Traditional RL environments assume:

- single objective
- stable conditions
- immediate feedback
- no human variability

Real systems do not behave like this.

In EnergyMind:

- users have different priorities
- feedback is delayed and noisy
- actions have long-term effects
- improving one metric harms another

This creates a trade-off system, where policy quality matters more than raw optimization.

## Environment and Baselines

We first built the OpenEnv-compatible environment:

- `app.py`
- `env/openenv_api.py`

Then established baselines:

- heuristic policy
- centralized Deep Q-Network (DQN)
- evaluation comparisons

Result: the environment produces real learning signal, not trivial reward shaping.

## LLM Integration Journey

### Direct Control (Failure)

Initial approach:

- LLM generates low-level actions per room

Problems:

- invalid outputs
- unstable behavior
- collapse to trivial policies

Conclusion: LLMs are not reliable low-level controllers in constrained systems.

### Strategy-Based Control (Working Design)

We redesigned the system:

- LLM selects high-level strategies
- `priority_safe`
- `complaint_rescue`
- `energy_saver`
- safe executor converts strategies into valid actions

Implementation:

- `training/llm_strategy.py`
- `training/eval_llm.py`

Result:

- stable behavior
- interpretable decisions
- consistent execution

## System Architecture

Final design:

- LLM -> high-level strategy planner
- constrained executor / RL agent -> low-level actions
- environment -> structured reward feedback

This separation ensures:

- safety
- interpretability
- reliability

## Post-Training Pipeline

EnergyMind supports both offline and online-style LLM post-training over a verifiable OpenEnv-compatible environment.

### Training Stages

#### 1. Build reward-derived strategy preferences
```bash
python training/build_strategy_preferences.py

Generates:

data/llm_strategy_preferences.jsonl
Contains:

prompt
chosen
rejected
best_reward
worst_reward

2. Train the offline preference-distilled planner
python training/train_preference_distill.py
Trains a LoRA-based strategy planner using reward-derived preferences.

3. Build grouped online-style rollouts
python training/build_grpo_rollouts.py
Records:

sampled completion
parsed strategy
reward
complaints
high-priority satisfaction
normalized advantage
Output:

data/llm_strategy_grpo_rollouts.jsonl

4. Train the GRPO-lite planner
python training/train_grpo_lite.py
Applies reward-weighted updates over rollout data.

5. Evaluate all post-trained models
python training/evaluate_post_training.py
Outputs:

eval_results/post_training_four_way_comparison.csv
Results : 
Model	                        Reward	Complaints	HP Satisfaction	Valid Rate
Base Qwen 0.5B	              10.93	  7.33	       3.0	           1.0
Preference-Distilled	        13.01	  0.0	         3.0	           1.0
Grouped Preference-Distilled	13.01	  0.0	         3.0	           1.0
GRPO-lite	                    11.35	  1.67	       3.0	           1.0

What This Actually Proves

direct LLM control fails in constrained environments
LLMs work better as planners, not controllers
reward-informed offline post-training is highly effective
environment-driven training beats pure supervised fine-tuning
online RL (GRPO-style) is viable but not yet optimal here

Key Insight

There is a real gap between:

generating structured outputs
making correct decisions
EnergyMind exposes this gap and provides a way to train against it.

Why This Matters
Most AI systems today are trained in clean, single-objective settings.

Real systems involve:

scarcity
competing users
delayed consequences
EnergyMind moves training closer to that reality.

Current Status
The project includes:

full environment implementation
OpenEnv integration
RL baselines
LLM strategy planner
TRL + LoRA pipeline
reward-informed post-training
grouped preference training
GRPO-lite online loop
evaluation artifacts
Everything is reproducible from the repository.

Conclusion
EnergyMind is not a benchmark.

It is a governance environment for training AI systems that must operate under real-world constraints.

Links
GitHub: https://github.com/anshu-ai-arch/hostalgrid-plus-plus
Demo: https://huggingface.co/spaces/anshu-123/hostalgrid-plus-plus

