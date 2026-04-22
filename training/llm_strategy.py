from __future__ import annotations

import json
import re
from typing import Dict, List, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


STRATEGIES = [
    "priority_safe",
    "complaint_rescue",
    "energy_saver",
    "comfort_all",
    "shutdown_empty",
    "do_nothing",
]


def current_actions(env) -> List[int]:
    actions = []
    for r in env.hostel.rooms:
        action = int(r.ac * 4 + r.fan * 2 + r.light)
        actions.append(action)
    return actions


def execute_strategy(env, strategy: str) -> List[int]:
    strategy = strategy if strategy in STRATEGIES else "priority_safe"

    if strategy == "do_nothing":
        return current_actions(env)

    if strategy == "comfort_all":
        return [7 if r.occupancy == 1 else 0 for r in env.hostel.rooms]

    if strategy == "energy_saver":
        return [6 if r.occupancy == 1 else 0 for r in env.hostel.rooms]

    if strategy == "shutdown_empty":
        actions = current_actions(env)
        for i, r in enumerate(env.hostel.rooms):
            if r.occupancy == 0:
                actions[i] = 0
        return actions

    actions = [0] * 10
    used = 0.0
    budget = env.hostel.power_budget

    if strategy == "complaint_rescue":
        order = sorted(
            range(10),
            key=lambda i: (
                -env.hostel.rooms[i].complaint,
                -env.hostel.rooms[i].priority,
                -env.hostel.rooms[i].occupancy,
            ),
        )
    else:
        order = sorted(
            range(10),
            key=lambda i: (
                -env.hostel.rooms[i].priority,
                -env.hostel.rooms[i].occupancy,
                -env.hostel.rooms[i].complaint,
            ),
        )

    for i in order:
        room = env.hostel.rooms[i]
        if room.occupancy == 0:
            continue

        if used + 990 <= budget:
            actions[i] = 7
            used += 990
        elif used + 90 <= budget:
            actions[i] = 6
            used += 90

    return actions


def make_strategy_prompt(env) -> str:
    lines = []
    lines.append("You are EnergyMind, an LLM planner for hostel electricity control.")
    lines.append("Choose exactly one strategy.")
    lines.append(
        "Valid strategies: priority_safe, complaint_rescue, energy_saver, comfort_all, shutdown_empty, do_nothing."
    )
    lines.append('Return only JSON like: {"strategy":"priority_safe"}')
    lines.append(f"mode={env.hostel.mode}, hour={env.hostel.hour}, heatwave={env.hostel.heatwave}")
    lines.append(
        f"power_used={env.hostel.total_power()}, power_budget={env.hostel.power_budget}, power_ratio={env.hostel.power_ratio():.3f}"
    )

    for r in env.hostel.rooms:
        lines.append(
            f"room {r.room_id}: occupied={r.occupancy}, priority={r.priority}, "
            f"complaint={r.complaint}, ac={r.ac}, fan={r.fan}, light={r.light}"
        )

    return "\n".join(lines)


def parse_strategy(text: str) -> Tuple[str, bool]:
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


class LLMStrategyPlanner:
    def __init__(self, model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto",
        )
        self.model.eval()

    def generate_strategy(self, env, max_new_tokens: int = 40):
        prompt = make_strategy_prompt(env)
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)

        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.eos_token_id,
            )

        decoded = self.tokenizer.decode(
            output[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )

        strategy, valid = parse_strategy(decoded)
        return strategy, valid, decoded

    def policy(self, env) -> Dict[str, object]:
        strategy, valid, raw = self.generate_strategy(env)
        actions = execute_strategy(env, strategy)
        return {
            "strategy": strategy,
            "valid": valid,
            "raw": raw,
            "actions": actions,
        }
