"""
training/policy_boost.py

Teacher policy + safe action wrapper.
This is the highest-value change for closing the gap to the heuristic.
"""

from env.action import ACTION_MAP

AC_WATTS    = 900.0
FAN_WATTS   = 75.0
LIGHT_WATTS = 15.0


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


def teacher_actions(env) -> list:
    """
    Stronger teacher than the old heuristic:
    - serve occupied HP and complaining rooms first
    - prefer fan+light unless heatwave / high complaint / HP needs stronger service
    """
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


def _downgrade_action(action_id: int) -> int:
    # Remove AC first, then reduce to minimal service, then off.
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
    """
    Post-process agent actions:
    - empty rooms OFF
    - protect occupied HP/complaining rooms from being fully ignored
    - if over budget, cut low-priority rooms first
    """
    rooms = env.hostel.rooms
    actions = list(actions)
    budget = env.hostel.power_budget
    heatwave = bool(getattr(env.hostel, "heatwave", False))

    # 1. Never waste on empty rooms
    for i, room in enumerate(rooms):
        if room.occupancy == 0:
            actions[i] = 0

    # 2. Protect urgent occupied rooms
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

    # 3. If over budget, degrade LP rooms first
    while total_cost() > budget:
        changed = False

        order = sorted(
            range(len(rooms)),
            key=lambda i: (
                rooms[i].occupancy,                  # empty / less important first
                rooms[i].priority,                   # low priority first
                getattr(rooms[i], "complaint", 0),  # low complaint first
                -action_power(actions[i]),          # remove expensive actions first
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
