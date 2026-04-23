from __future__ import annotations

import json
import os
import re
import sys

import numpy as np
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


def parse_strategy(text: str):
    try:
        text = text.replace("```json", "").replace("```", "").strip()
        match = re.search(r"\{.*\}", text, re.DOTALL)
        if not match:
            return None, False
        data = json.loads(match.group(0))
        strategy = data.get("strategy", "")
        if strategy not in STRATEGIES:
            return None, False
        return strategy, True
    except Exception:
        return None, False


def sample_strategy_text(model, tokenizer, env, temperature=1.0, max_new_tokens=40):
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
            do_sample=True,
            temperature=temperature,
            top_p=0.95,
            pad_token_id=tokenizer.eos_token_id,
        )

    decoded = tokenizer.decode(
        output[0][inputs["input_ids"].shape[1]:],
        skip_special_tokens=True,
    )
    strategy, valid = parse_strategy(decoded)
    return prompt, strategy, valid, decoded


def score_generated_strategy(mode, seed, strategy):
    env = HostelGridEnv(mode=mode, seed=seed)
    env.reset()

    if strategy not in STRATEGIES:
        strategy = "do_nothing"

    actions = execute_strategy(env, strategy)
    _, reward, done, info = env.step(Action.from_list(actions))

    return {
        "reward": float(reward.total),
        "complaints": info["complaints"],
        "hp_satisfied": info["hp_satisfied"],
        "power_used": info["power_used"],
    }


def build_grpo_rollouts(model, tokenizer, mode="medium", seeds=range(100, 115), samples_per_state=8):
    rows = []

    for seed in seeds:
        group = []
        env = HostelGridEnv(mode=mode, seed=seed)
        env.reset()

        for sample_id in range(samples_per_state):
            prompt, strategy, valid, raw = sample_strategy_text(model, tokenizer, env)

            scored = score_generated_strategy(
                mode=mode,
                seed=seed,
                strategy=strategy if valid else "do_nothing",
            )

            group.append({
                "prompt": prompt,
                "completion": raw,
                "strategy": strategy if valid else "do_nothing",
                "valid": valid,
                "raw": raw,
                **scored,
            })

        rewards = np.array([x["reward"] for x in group], dtype=np.float32)
        mean_r = float(rewards.mean())
        std_r = float(rewards.std()) + 1e-6

        for row in group:
            advantage = (row["reward"] - mean_r) / std_r
            row["advantage"] = float(np.clip(advantage, -2.0, 2.0))
            row["mode"] = mode
            row["seed"] = seed
            rows.append(row)

    return rows


def main():
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL,
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )
    model = PeftModel.from_pretrained(base_model, PREF_LORA_DIR)
    model.eval()

    out_path = os.path.join(REPO_ROOT, "data", "llm_strategy_grpo_rollouts.jsonl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    rows = build_grpo_rollouts(model, tokenizer)

    with open(out_path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")

    print("Saved:", out_path)
    print("Rows:", len(rows))
    if rows:
        print(rows[0])


if __name__ == "__main__":
    main()
