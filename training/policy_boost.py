"""
training/policy_boost.py

LLM-guided teacher policy + safe action wrapper.
LLM suggests room-wise actions, but fallback rule teacher remains available.
"""

import os
import json
from openai import OpenAI
from env.action import ACTION_MAP

AC_WATTS    = 900.0
FAN_WATTS   = 75.0
LIGHT_WATTS = 15.0

API_BASE_URL = os.environ.get("API_BASE_URL", "https://api.openai.com/v1")
MODEL_NAME   = os.environ.get("MODEL_NAME", "gpt-4o-mini")
HF_TOKEN     = os.environ.get("HF_TOKEN", "")

client = OpenAI(
    api_key=HF_TOKEN if HF_TOKEN else os.environ.get("OPENAI_API_KEY", "sk-placeholder"),
    base_url=API_BASE_URL,
)


def action_power(action_id: int) -> float:
    ac, fan, light = ACTION_MAP[int(action_id)]
    return ac * AC_WATTS + fan * FAN_WATTS + light * LIGHT_WATTS


def service_count(action_id: int) -> int:
    return sum(ACTION_MAP[int(action_id)])


def teacher_prob(ep: int,
                 total_eps: int,
                 start: float = 0.80,
                 end: float = 0.05,
                 frac: float = 0.55) -> float:
    cutoff = max(1, int(total_eps * frac))
    if ep >= cutoff:
        return end
    alpha = ep / cutoff
    return start * (1.0 - alpha) + end * alpha


def _candidate_actions(room, heatwave: bool):
    complaint = getattr(room, "complaint", 0)

    if room.occupancy == 0:
        return [0]

    if room.priority >= 3:
        if heatwave or complaint >= 2:
            return [7, 6, 4, 5, 2, 3, 0]
        return [6, 7, 2, 3, 4, 5, 0]

    if room.priority == 2:
        if complaint >= 2:
            return [6, 7, 2, 3, 4, 5, 0]
        return [6, 2, 3, 7, 0]

    if complaint >= 2 or getattr(room, "consecutive_ignored", 0) >= 2:
        return [6, 2, 3, 7, 0]

    return [2, 3, 6, 0]


def rule_teacher_actions(env) -> list:
    rooms = env.hostel.rooms
    actions = [0] * len(rooms)
    budget = env.hostel.power_budget
    used = 0.0
    heatwave = bool(getattr(env.hostel, "heatwave", False))

    order = sorted(
        range(len(rooms)),
        key=lambda i: (
            -rooms[i].priority,
            -rooms[i].occupancy,
            -getattr(rooms[i], "complaint", 0),
            -getattr(rooms[i], "consecutive_ignored", 0),
        )
    )

    for i in order:
        room = rooms[i]
        for cand in _candidate_actions(room, heatwave):
            if used + action_power(cand) <= budget:
                actions[i] = cand
                used += action_power(cand)
                break

    return actions


def build_env_snapshot(env) -> dict:
    rooms = env.hostel.rooms
    return {
        "mode": env.mode,
        "step": env.current_step,
        "hour": env.hostel.hour,
        "heatwave": bool(getattr(env.hostel, "heatwave", False)),
        "power_budget": env.hostel.power_budget,
        "power_used": env.hostel.total_power(),
        "rooms": [
            {
                "room_id": r.room_id,
                "priority": r.priority,
                "occupancy": r.occupancy,
                "complaint": r.complaint,
                "consecutive_ignored": getattr(r, "consecutive_ignored", 0),
                "current_action": [r.ac, r.fan, r.light],
            }
            for r in rooms
        ],
    }


def llm_teacher_actions(env) -> list:
    snapshot = build_env_snapshot(env)

    prompt = f"""
You are an energy-governance planner for a 10-room hostel.

You must choose one action 0-7 for each room.

Action meanings:
0=all_off
1=ac_only
2=fan_only
3=light_only
4=ac_fan
5=ac_light
6=fan_light
7=all_on

Rules:
- Respect scarcity and complaints.
- Occupied high-priority rooms should be protected first.
- Empty rooms should usually be 0.
- In heatwave, stronger cooling for important occupied rooms is preferred.
- Budget matters. Avoid giving all rooms expensive actions.

Return ONLY valid JSON like:
{{"actions":[0,6,7,2,0,3,6,0,2,0]}}

State:
{json.dumps(snapshot)}
""".strip()

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=120,
            temperature=0.0,
        )
        text = response.choices[0].message.content.strip()
        start = text.find("{")
        end = text.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("No JSON object found")
        payload = json.loads(text[start:end+1])
        actions = payload.get("actions", [])
        if not isinstance(actions, list) or len(actions) != len(env.hostel.rooms):
            raise ValueError("Bad action list length")
        actions = [int(a) for a in actions]
        if any(a < 0 or a > 7 for a in actions):
            raise ValueError("Invalid action id")
        return actions
    except Exception:
        return rule_teacher_actions(env)


def teacher_actions(env) -> list:
    """
    Main teacher entry point used by training.
    First try LLM teacher, then fallback to strong rule teacher.
    """
    return llm_teacher_actions(env)


def _downgrade_action(action_id: int) -> int:
    chain = {
        7: 6,
        6: 2,
        5: 3,
        4: 2,
        3: 0,
        2: 0,
        1: 0,
        0: 0,
    }
    return chain.get(int(action_id), 0)


def safe_actions(env, actions: list) -> list:
    rooms = env.hostel.rooms
    actions = list(actions)
    budget = env.hostel.power_budget
    heatwave = bool(getattr(env.hostel, "heatwave", False))

    for i, room in enumerate(rooms):
        if room.occupancy == 0:
            actions[i] = 0

    for i, room in enumerate(rooms):
        if room.occupancy != 1:
            continue

        complaint = getattr(room, "complaint", 0)
        cur_service = service_count(actions[i])

        if room.priority >= 3 and complaint >= 2:
            target = 7 if heatwave else 6
            if cur_service < service_count(target):
                actions[i] = target
        elif room.priority >= 2 and complaint >= 1:
            if cur_service < 2:
                actions[i] = 6
        elif cur_service == 0:
            actions[i] = 2 if complaint > 0 else 3

    def total_cost() -> float:
        return sum(action_power(a) for a in actions)

    while total_cost() > budget:
        changed = False

        order = sorted(
            range(len(rooms)),
            key=lambda i: (
                rooms[i].occupancy,
                rooms[i].priority,
                getattr(rooms[i], "complaint", 0),
                -action_power(actions[i]),
            )
        )

        for i in order:
            old = actions[i]
            new = _downgrade_action(old)
            if new != old:
                actions[i] = new
                changed = True
                if total_cost() <= budget:
                    return actions

        if not changed:
            break

    return actions
