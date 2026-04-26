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

EnergyMind is a human-aware energy-governance environment where an AI system must allocate scarce electricity across competing rooms under hard power limits, room priority, complaint buildup, and delayed consequences. The project began with a simple question: can an LLM directly control a constrained real-world system? Our answer was no. Direct low-level LLM control was brittle. The system became much stronger when we redesigned it into a hybrid architecture: the LLM plans, a learned executor acts, and the environment reward closes the loop.

This README is the shortest complete version of the project story: what problem we modeled, what failed, what we changed, what architecture we used, and what results we achieved.

## One-Sentence Summary

EnergyMind shows that in constrained, human-centered environments, LLMs work better as high-level planners than as direct low-level controllers, and reward-informed post-training improves both planner quality and full hybrid-system performance.

## Quick Links

- Hugging Face Space: [anshu-123/hostalgrid-plus-plus](https://huggingface.co/spaces/anshu-123/hostalgrid-plus-plus)
- GitHub Repository: [anshu-ai-arch/hostalgrid-plus-plus](https://github.com/anshu-ai-arch/hostalgrid-plus-plus)
- Main notebook: `EnergyMind_OpenEnv_TRL_Pipeline.ipynb`
- Post-training plots: [`experiment_runs/2026-04-25_dqn_campaign/post_training/plots`](https://github.com/anshu-ai-arch/hostalgrid-plus-plus/tree/main/experiment_runs/2026-04-25_dqn_campaign/post_training/plots)
- Hybrid evaluation artifacts: [`experiment_runs/2026-04-25_dqn_campaign/hybrid`](https://github.com/anshu-ai-arch/hostalgrid-plus-plus/tree/main/experiment_runs/2026-04-25_dqn_campaign/hybrid)
- Evaluation tables: [`eval_results`](https://github.com/anshu-ai-arch/hostalgrid-plus-plus/tree/main/eval_results)

## Why This Problem Matters

Most training environments optimize one clean objective. Real systems do not.

EnergyMind models a shared hostel where electricity must be allocated under:

- scarce shared power
- multiple human actors with different needs
- room priority and occupancy changes
- delayed complaint buildup
- fairness pressure
- crisis conditions such as heatwaves and outages

This turns energy control into a governance problem, not just an optimization problem. The agent is not simply trying to save power. It must decide who gets served, when, under what constraints, and at what future cost.

## Core Contributions

This project contributes three things together:

1. A constrained human-aware environment for training and evaluating control policies under delayed social consequences.
2. A hybrid planner-executor architecture where the LLM suggests high-level strategy and a learned controller chooses final low-level actions.
3. Closed-loop post-training evidence showing that better planner training improves actual environment behavior.

## Project Story

### Phase 1: Let the LLM directly control the environment

Our first design asked the LLM to directly produce low-level room actions.

That looked attractive because it was simple:

- observe state
- generate action
- step the environment

But it failed in practice. Even when outputs were syntactically valid, direct low-level control was brittle under hard constraints. The model could generate plausible structured output without actually making robust decisions about budget, complaints, urgency, or safe allocation.

This was the turning point of the project.

### Phase 2: Change the role of the LLM

Instead of making the LLM directly control appliances, we changed its role:

- from low-level executor
- to high-level planner

That redesign is the key idea in EnergyMind.

The LLM now proposes a high-level strategy such as:

- `priority_safe`
- `complaint_rescue`
- `energy_saver`
- `comfort_all`
- `shutdown_empty`
- `do_nothing`

Then a learned executor uses the actual state and learned value estimates to choose room-level actions. The executor is not forced to blindly copy the planner. It can benefit from the planner when the suggestion helps, while still staying grounded in the low-level control problem.

## Final Architecture

```mermaid
flowchart LR
    A["Environment state<br/>occupancy, room priority, complaints,<br/>power budget, hour, heatwave"] --> B["LLM planner<br/>proposes high-level strategy"]
    A --> C["Q-learning / DQN executor<br/>scores low-level room actions"]
    B --> D["Strategy guidance signal<br/>soft preference, not hard override
