# inference.py
# HostelGrid++ — OpenEnv baseline inference script
# Uses OpenAI client with [START][STEP][END] structured logging

import os
import random
import time
from openai import OpenAI
from env.openenv_api import HostelGridOpenEnv, Action

# ── API Configuration ─────────────────────────────────────────
API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME",   "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN",     "")

client = OpenAI(
    api_key  = HF_TOKEN if HF_TOKEN else os.environ.get("OPENAI_API_KEY", "sk-placeholder"),
    base_url = API_BASE_URL,
)

TASKS = ["task_easy", "task_medium", "task_hard"]


def get_action_from_llm(observation: dict, task_id: str, step: int) -> int:
    """Smarter prompt — gives LLM context about what to optimize."""

    # Task-specific guidance
    task_hints = {
        "task_easy":   "Priority: NEVER let committed rooms run out of power. Honor all approved requests first.",
        "task_medium": "Priority: Detect demand spikes and use action 4 to defer heavy loads immediately.",
        "task_hard":   "Priority: Keep system trust high. Balance comfort and cost. Avoid complaint spikes.",
    }
    hint = task_hints.get(task_id, "")

    # Time context
    hour = observation['time_of_day']
    if 9 <= hour <= 12 or 18 <= hour <= 22:
        time_context = "PEAK HOURS — electricity is expensive (Rs 8.5/kWh). Prefer saving energy."
    elif 0 <= hour <= 5:
        time_context = "NIGHT HOURS — electricity is cheap (Rs 4.0/kWh). Safe to use more."
    else:
        time_context = "NORMAL HOURS — moderate cost (Rs 6.0/kWh)."

    prompt = f"""You are EnergyMind, an AI managing energy for a 20-room hostel.

CURRENT STATE (Hour {step}/24):
- Power usage    : {observation['power_usage']:.2f} kW
- Avg temperature: {observation['avg_temperature']:.1f}°C  
- Occupancy      : {observation['avg_occupancy']:.1%} of rooms occupied
- Complaints     : {observation['complaint_level']} active complaints
- Carbon rate    : {observation['carbon_rate']:.2f} gCO2/kWh
- Cost so far    : Rs {observation['current_cost']:.2f}
- {time_context}

TASK: {task_id}
STRATEGY: {hint}

ACTIONS:
0: increase_ac     → comfort up, cost up, complaints drop
1: decrease_ac     → save energy, complaints may rise
2: lights_off_empty → silent energy save (best if occupancy low)
3: lights_on       → restore lights
4: defer_heavy_load → big energy save, shift loads to off-peak
5: do_nothing      → hold current state

DECISION RULES:
- complaints > 3 → choose 0 (increase comfort)
- peak hours + power > 8kW → choose 4 (defer load)
- occupancy < 0.4 → choose 2 (lights off empty rooms)
- power > 10kW at night → choose 1 (decrease AC)

Respond with ONLY a single digit 0-5."""

    try:
        response = client.chat.completions.create(
            model       = MODEL_NAME,
            messages    = [{"role": "user", "content": prompt}],
            max_tokens  = 5,
            temperature = 0.0,
        )
        text   = response.choices[0].message.content.strip()
        action = int(text[0])
        if action not in range(6):
            action = _rule_based_fallback(observation)
    except Exception:
        action = _rule_based_fallback(observation)

    return action


def _rule_based_fallback(observation: dict) -> int:
    """Smart fallback when LLM fails — uses rules."""
    hour        = int(observation['time_of_day'])
    power       = observation['power_usage']
    complaints  = observation['complaint_level']
    occupancy   = observation['avg_occupancy']

    if complaints > 3:
        return 0   # increase AC
    if (9 <= hour <= 12 or 18 <= hour <= 22) and power > 8:
        return 4   # defer heavy load during peak
    if occupancy < 0.4:
        return 2   # lights off empty rooms
    if power > 10 and hour < 6:
        return 1   # decrease AC at night
    return 5       # do nothing

def run_task(task_id: str) -> float:
    """Run one full episode for a task and return score 0.0-1.0."""

    env = HostelGridOpenEnv(task_id=task_id, num_rooms=20)
    obs = env.reset()

    # ── [START] log ───────────────────────────────────────────
    print(f"[START] task={task_id}", flush=True)

    done  = False
    step  = 0
    score = 0.0

    while not done:
        step += 1

        # Get action from LLM
        action_id = get_action_from_llm(obs.model_dump(), task_id, step)
        action    = Action(action_id=action_id)

        # Step environment
        next_obs, reward, done, info = env.step(action)

        # ── [STEP] log ────────────────────────────────────────
        print(f"[STEP] step={step} reward={reward.value:.4f}", flush=True)

        obs = next_obs

    score = env.score()

    # ── [END] log ─────────────────────────────────────────────
    print(f"[END] task={task_id} score={score:.4f} steps={step}", flush=True)

    return score


def main():
    """Run all 3 tasks and report final scores."""

    scores = {}
    for task_id in TASKS:
        print(f"\n{'='*50}", flush=True)
        print(f"Running {task_id}...", flush=True)
        print(f"{'='*50}", flush=True)
        score = run_task(task_id)
        scores[task_id] = score
        print(f"\n✅ {task_id} Score: {score:.4f}", flush=True)
        time.sleep(1)

    avg = sum(scores.values()) / len(scores)

    print(f"\n{'='*50}", flush=True)
    print(f"📊 Final Scores:", flush=True)
    for t, s in scores.items():
        print(f"   {t}: {s:.4f}", flush=True)
    print(f"   Average: {avg:.4f}", flush=True)
    print(f"{'='*50}", flush=True)


if __name__ == "__main__":
    main()