## EnergyMind: Training AI Systems Under Real-World Constraints
    
GitHub: https://github.com/anshu-ai-arch/hostalgrid-plus-plus

Hugging Face Space: https://huggingface.co/spaces/anshu-123/hostalgrid-plus-plus

## The Problem

What happens when an AI system must decide how to allocate limited resources across multiple users with conflicting needs?

Most AI systems are trained in simplified environments with:
- single objectives
- immediate feedback
- stable conditions

Real-world systems involve:
- scarcity
- competing priorities
- delayed consequences
- unavoidable trade-offs

EnergyMind is designed to train AI systems in this setting.

## Core Idea

EnergyMind is an OpenEnv-compatible environment where an agent must allocate electricity across a shared hostel.

At every step, the agent must balance:
- fairness vs efficiency
- high-priority vs normal users
- energy cost vs comfort
- short-term relief vs long-term stability

This is a constrained decision-making problem, not a single-objective optimization task.

## Environment Design

### Setup
- 10 rooms with dynamic occupancy
- High-priority and normal users
- Global energy budget constraint
- Time-evolving system state

### Dynamics
- Complaints accumulate over time
- Ignoring users creates delayed penalties
- Resource allocation affects future states

The environment requires long-horizon planning rather than reactive control.

## Key Challenge

Traditional RL vs EnergyMind:
- Single objective → Multiple conflicting objectives
- Immediate reward → Delayed feedback
- Static state → Evolving system
- No trade-offs → Continuous trade-offs

## Key Insight

### LLM as Controller (Failure)
Direct LLM control leads to:
- invalid actions
- unstable policies
- collapse to trivial behavior

### LLM as Planner (Working Approach)
Using LLMs for high-level strategy selection results in:
- stable behavior
- interpretable decisions
- consistent execution

## System Architecture

LLM (Strategy Planner)
        ↓
High-Level Strategy
        ↓
Safe Executor / RL Policy
        ↓
Environment (EnergyMind)
        ↓
Reward Feedback

### Strategy Space
- priority_safe
- complaint_rescue
- energy_saver

This separation ensures safety, interpretability, and reliability.

## Post-Training Pipeline

EnergyMind supports both offline and online-style LLM post-training over a verifiable OpenEnv-compatible environment.

### Training Stages

1. Build Reward-Derived Strategy Preferences
python training/build_strategy_preferences.py

Generates:
data/llm_strategy_preferences.jsonl

Contains:
- prompt
- chosen
- rejected
- best_reward
- worst_reward

2. Train the Offline Preference-Distilled Planner
python training/train_preference_distill.py

3. Build Grouped Online-Style Rollouts
python training/build_grpo_rollouts.py

Records:
- sampled completion
- parsed strategy
- reward
- complaints
- high-priority satisfaction
- normalized advantage

Output:
data/llm_strategy_grpo_rollouts.jsonl

4. Train the GRPO-lite Planner
python training/train_grpo_lite.py

5. Evaluate All Post-Trained Models
python training/evaluate_post_training.py

Outputs:
eval_results/post_training_four_way_comparison.csv

## Results

Model                        | Reward | Complaints | HP Satisfaction | Valid Rate
Base Qwen 0.5B               | 10.93  | 7.33       | 3.0             | 1.0
Preference-Distilled         | 13.01  | 0.0        | 3.0             | 1.0
Grouped Preference-Distilled | 13.01  | 0.0        | 3.0             | 1.0
GRPO-lite                    | 11.35  | 1.67       | 3.0             | 1.0

## What This Actually Proves

- Direct LLM control fails in constrained environments
- LLMs perform better as planners than controllers
- Reward-informed offline post-training is highly effective
- Environment-driven training outperforms pure supervised fine-tuning
- Online RL (GRPO-style) is viable but not yet optimal in this setting

## Key Insight

There is a fundamental gap between:
- generating structured outputs
- making correct decisions

EnergyMind exposes this gap and provides a framework to train against it.

## Why This Matters

Most AI systems today are trained in clean, single-objective settings.

Real-world systems involve:
- scarcity
- competing users
- delayed consequences

EnergyMind introduces these constraints explicitly and enables training for them.

## Current Status

The project includes:
- full environment implementation
- OpenEnv integration
- RL baselines
- LLM strategy planner
- TRL + LoRA pipeline
- reward-informed post-training
- grouped preference training
- GRPO-lite online loop
- evaluation artifacts

All components are reproducible from the repository.

## Reproducibility

The repository provides:
- OpenEnv-compatible environment
- training scripts
- evaluation pipeline
- reproducible experiments

## Conclusion

EnergyMind is not a benchmark.

It is a governance environment for training AI systems that must operate under real-world constraints.

It demonstrates that:
- optimization alone is insufficient
- structured environments improve decision quality
- separating planning from execution leads to more reliable AI systems
  
