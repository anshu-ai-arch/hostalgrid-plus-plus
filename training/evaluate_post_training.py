from __future__ import annotations

import json
import os
import re
import sys

import numpy as np
import pandas as pd
import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = os.path.dirname(os.path.dirname(__file__))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from env.hostelgrid_env import HostelGridEnv
from env.action import Action
from training.llm_strategy import execute_strategy, make_strategy_prompt

STRATEGIES = [
    "priority_safe",
    "complaint_rescue",
    "energy_saver",
    "comfort_all",
    "shutdown_empty",
    "do_nothing",
]

BASE_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
PREF_LORA_DIR = "/content/drive/MyDrive/hostelgrid-work/outputs/energymind-prefdistill-qwen05b"
GROUP_PREF_LORA_DIR = "/content/drive/MyDrive/hostelgrid-work/outputs/energymind-group-prefdistill-qwen05b"
GRPO_LORA_DIR = "/content/drive/MyDrive/hostelgrid-work/outputs/energymind-grpo-lite-qwen05b"


def parse_strategy(text: str):
    try:
        text = text.replace("```json", "").replace("```", "").strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return "priority_safe", False

        data = json.loads(match.group(0))
        strategy = data.get("strategy", "")
        if strategy not in STRATEGIES:
            return "priority_safe", False

        return strategy, True
    except Exception:
        return "priority_safe", False


def generate_strategy(model, tokenizer, env, max_new_tokens=40):
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
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )

    strategy, valid = parse_strategy(decoded)
    return strategy, valid, decoded


def eval_strategy_model(model, tokenizer, mode="medium", seeds=(100, 101, 102), steps=10):
    rewards = []
    complaints = []
    hp_sats = []
    valids = []

    for seed in seeds:
        env = HostelGridEnv(mode=mode, seed=seed)
        env.reset()
        total_reward = 0.0
        valid_count = 0
        step_count = 0
        last_info = None

        for _ in range(steps):
            strategy, valid, _ = generate_strategy(model, tokenizer, env)
            actions = execute_strategy(env, strategy)
            _, reward, done, info = env.step(Action.from_list(actions))
            total_reward += reward.total
            valid_count += int(valid)
            step_count += 1
            last_info = info
            if done:
                break

        rewards.append(total_reward)
        complaints.append(last_info["complaints"])
        hp_sats.append(last_info["hp_satisfied"])
        valids.append(valid_count / max(1, step_count))

    return {
        "reward_mean": float(np.mean(rewards)),
        "complaints_mean": float(np.mean(complaints)),
        "hp_sat_mean": float(np.mean(hp_sats)),
        "valid_rate": float(np.mean(valids)),
    }


def load_base_and_adapter(adapter_dir: str | None = None):
    base = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    if adapter_dir is None:
        base.eval()
        return base

    model = PeftModel.from_pretrained(base, adapter_dir)
    model.eval()
    return model


def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = load_base_and_adapter(None)
    pref_model = load_base_and_adapter(PREF_LORA_DIR)
    group_pref_model = load_base_and_adapter(GROUP_PREF_LORA_DIR)
    grpo_model = load_base_and_adapter(GRPO_LORA_DIR)

    base_res = eval_strategy_model(base_model, tokenizer)
    pref_res = eval_strategy_model(pref_model, tokenizer)
    group_pref_res = eval_strategy_model(group_pref_model, tokenizer)
    grpo_res = eval_strategy_model(grpo_model, tokenizer)

    df = pd.DataFrame([
        {"model": "Base Qwen 0.5B", **base_res},
        {"model": "Preference-Distilled", **pref_res},
        {"model": "Grouped Preference-Distilled", **group_pref_res},
        {"model": "GRPO-lite", **grpo_res},
    ])

    out_dir = os.path.join(REPO_ROOT, "eval_results")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "post_training_four_way_comparison.csv")
    df.to_csv(out_path, index=False)

    print(df)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
