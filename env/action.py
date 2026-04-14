# env/action.py
# Next-level action space — more granular, correct strategy categories

import numpy as np

# ── Full Action Space (10 actions) ────────────────────────────
# Expanded from 6 to 10 for finer control
ACTIONS = {
    # ── Conservative (preserve comfort, protect commitments) ──
    0: "increase_ac_full",      # +2C cooling, comfort up, cost up
    1: "increase_ac_partial",   # +1C cooling, moderate cost
    2: "restore_lights",        # restore lights in all occupied rooms
    3: "do_nothing",            # hold current state ← CONSERVATIVE (fixed)

    # ── Balanced (efficiency without sacrifice) ───────────────
    4: "decrease_ac_partial",   # -1C cooling, mild energy save
    5: "lights_off_empty",      # lights off in unoccupied rooms only
    6: "defer_light_load",      # shift small appliances to off-peak

    # ── Aggressive (strong optimization, some risk) ───────────
    7: "decrease_ac_full",      # -2C cooling, strong save, complaints likely
    8: "defer_heavy_load",      # shift heavy appliances (washing, EV, heater)
    9: "emergency_curtail",     # cut all non-critical power (crisis only)
}

# ── Strategy categories (fixed: do_nothing = conservative) ───
ACTION_STRATEGY = {
    0: "conservative",
    1: "conservative",
    2: "conservative",
    3: "conservative",   # ← FIXED (was aggressive before)
    4: "balanced",
    5: "balanced",
    6: "balanced",
    7: "aggressive",
    8: "aggressive",
    9: "aggressive",
}

# ── Power delta per action (kW change to total load) ─────────
ACTION_POWER_DELTA = {
    0:  +1.5,   # increase_ac_full
    1:  +0.8,   # increase_ac_partial
    2:  +0.2,   # restore_lights
    3:   0.0,   # do_nothing
    4:  -0.5,   # decrease_ac_partial
    5:  -0.3,   # lights_off_empty (depends on empty rooms)
    6:  -0.8,   # defer_light_load
    7:  -1.2,   # decrease_ac_full
    8:  -2.5,   # defer_heavy_load
    9:  -4.0,   # emergency_curtail
}

# ── Comfort impact per action ─────────────────────────────────
# Positive = comfort increases, Negative = complaints may rise
ACTION_COMFORT_DELTA = {
    0:  +2,   # strong comfort gain
    1:  +1,   # mild comfort gain
    2:  +1,   # lights restore = comfort gain
    3:   0,   # no change
    4:  -1,   # mild complaint risk
    5:   0,   # empty rooms = no impact
    6:  -1,   # mild disruption
    7:  -2,   # strong complaint risk
    8:  -2,   # heavy load off = disruption
    9:  -4,   # emergency = severe complaints
}

# ── Minimum power floor per action (kW) ──────────────────────
# Agent CANNOT reduce below this — prevents complete shutdown bug
ACTION_MIN_POWER = {
    0:  2.0,
    1:  2.0,
    2:  1.0,
    3:  1.0,
    4:  1.0,
    5:  0.5,
    6:  0.5,
    7:  0.5,
    8:  0.3,
    9:  0.3,   # emergency still keeps 0.3kW (life safety)
}

# ── Temperature delta per action (C) ─────────────────────────
ACTION_TEMP_DELTA = {
    0:  -2.0,   # more cooling
    1:  -1.0,
    2:   0.0,
    3:   0.0,
    4:  +0.5,
    5:   0.0,
    6:   0.0,
    7:  +1.5,   # less cooling = warmer
    8:   0.0,
    9:  +2.0,   # curtail = very warm
}


# ── Helper functions ──────────────────────────────────────────

def get_action_name(action_id: int) -> str:
    return ACTIONS.get(action_id, "unknown")

def get_action_strategy(action_id: int) -> str:
    return ACTION_STRATEGY.get(action_id, "balanced")

def get_power_delta(action_id: int) -> float:
    return ACTION_POWER_DELTA.get(action_id, 0.0)

def get_comfort_delta(action_id: int) -> int:
    return ACTION_COMFORT_DELTA.get(action_id, 0)

def get_temp_delta(action_id: int) -> float:
    return ACTION_TEMP_DELTA.get(action_id, 0.0)

def get_min_power(action_id: int) -> float:
    return ACTION_MIN_POWER.get(action_id, 0.5)

def get_action_count() -> int:
    return len(ACTIONS)

def is_conservative(action_id: int) -> bool:
    return ACTION_STRATEGY.get(action_id) == "conservative"

def is_aggressive(action_id: int) -> bool:
    return ACTION_STRATEGY.get(action_id) == "aggressive"

def get_conservative_actions() -> list:
    return [a for a, s in ACTION_STRATEGY.items() if s == "conservative"]

def get_aggressive_actions() -> list:
    return [a for a, s in ACTION_STRATEGY.items() if s == "aggressive"]

def get_balanced_actions() -> list:
    return [a for a, s in ACTION_STRATEGY.items() if s == "balanced"]