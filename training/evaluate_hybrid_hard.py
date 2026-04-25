from __future__ import annotations

import json
import os
import sys
import re
import numpy as np
import torch
import pandas as pd
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from env.hostelgrid_env import HostelGridEnv
from agent.dqn_agent import DQNAgent, NUM_ROOMS, ACTION_COSTS
from training.llm_strategy import make_strategy_prompt, execute_strategy, parse_strategy

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
POST_TRAINED_REPO = "anshu-123/energymind-prefdistill-merged-final"
DQN_CKPT = os.path.join(REPO_ROOT, "experiment_runs", "2026-04-25_dqn_campaign", "hybrid", "checkpoints", "dqn_hard_ep1000_seed42.npz")

SEEDS = [100, 101, 102]
STEPS = 10


def load_dqn(checkpoint_path: str) -> DQNAgent:
    agent = DQNAgent()
    data = np.load(checkpoint_path)

    agent.policy_net.W1 = data["W1"]
    agent.policy_net.b1 = data["b1"]
    agent.policy_net.W2 = data["W2"]
    agent.policy_net.b2 = data["b2"]
    agent.policy_net.W3 = data["W3"]
    agent.policy_net.b3 = data["b3"]

    agent.target_net.copy_from(agent.policy_net)
    agent.epsilon = float(data["epsilon"][0])
    agent.steps = int(data["steps"][0])
    agent.episodes = int(data["episodes"][0])
    return agent


def load_planner(adapter_repo: str | None = None):
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    if adapter_repo:
        model = PeftModel.from_pretrained(base, adapter_repo)
    else:
        model = base

    model.eval()
    return model, tokenizer


def planner_strategy(model, tokenizer, env):
    prompt = make_strategy_prompt(env)
    messages = [{"role": "user", "content": prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=40,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    strategy, valid = parse_strategy(decoded)
    return strategy, valid, decoded


def hybrid_actions(env, dqn: DQNAgent, strategy: str, bonus: float = 0.35):
    planner_pref = execute_strategy(env, strategy)

    states, budget = dqn._obs_to_room_states(env.state())
    flat_state = states.reshape(1, -1)
    q_vals = dqn._to_q_tensor(dqn.policy_net.predict(flat_state))[0]

    actions = [0] * NUM_ROOMS
    remaining = float(budget)

    order = sorted(
        range(NUM_ROOMS),
        key=lambda i: dqn._urgency(states[i], q_vals[i]),
        reverse=True
    )

    for i in order:
        cands = dqn._candidate_actions(states[i])
        pref = int(planner_pref[i])

        ranked = sorted(
            cands,
            key=lambda a: float(q_vals[i, a] + (bonus if int(a) == pref else 0.0)),
            reverse=True,
        )

        chosen = 0
        for a in ranked:
            if ACTION_COSTS[a] <= remaining + 1e-6:
                chosen = int(a)
                break

        actions[i] = chosen
        remaining -= float(ACTION_COSTS[chosen])

    return actions


def eval_hybrid(planner_model, planner_tokenizer, dqn: DQNAgent, label: str):
    rewards = []
    complaints = []
    hp_sats = []
    valid_rates = []

    for seed in SEEDS:
        env = HostelGridEnv(mode="hard", seed=seed)
        env.reset()
        total_reward = 0.0
        valid_count = 0
        step_count = 0
        last_info = None

        for _ in range(STEPS):
            strategy, valid, _ = planner_strategy(planner_model, planner_tokenizer, env)
            actions = hybrid_actions(env, dqn, strategy)
            _, reward, done, info = env.step(actions)

            total_reward += float(reward.total)
            valid_count += int(valid)
            step_count += 1
            last_info = info

            if done:
                break

        rewards.append(total_reward)
        complaints.append(last_info["complaints"])
        hp_sats.append(last_info["hp_satisfied"])
        valid_rates.append(valid_count / max(1, step_count))

    return {
        "model": label,
        "reward_mean": float(np.mean(rewards)),
        "complaints_mean": float(np.mean(complaints)),
        "hp_sat_mean": float(np.mean(hp_sats)),
        "valid_rate": float(np.mean(valid_rates)),
    }


def main():
    dqn = load_dqn(DQN_CKPT)

    base_model, base_tok = load_planner(None)
    pref_model, pref_tok = load_planner(POST_TRAINED_REPO)

    rows = [
        eval_hybrid(base_model, base_tok, dqn, "Base LLM + DQN Executor"),
        eval_hybrid(pref_model, pref_tok, dqn, "Post-trained LLM + DQN Executor"),
    ]

    df = pd.DataFrame(rows)
    out_path = os.path.join(REPO_ROOT, "eval_results", "hybrid_hard_comparison.csv")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    df.to_csv(out_path, index=False)

    print(df)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
