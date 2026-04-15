# env/action.py
# Fixed: trimmed to 6 actions to match agent action_count=6 in all tasks.
# Original had 10 actions but agents were trained with action_count=6,
# meaning actions 6-9 were never reachable. The 6 kept actions cover
# the full conservative→balanced range needed by all three tasks.
# hostelgrid_env.step() asserts action < get_action_count(), so this
# makes the assert consistent with what agents actually produce.

import numpy as np

ACTIONS = {
    0: "increase_ac_full",      # conservative: +2C cooling, cost up
    1: "increase_ac_partial",   # conservative: +1C cooling, moderate cost
    2: "restore_lights",        # conservative: restore all occupied lights
    3: "do_nothing",            # conservative: hold state
    4: "decrease_ac_partial",   # balanced: -1C, mild save, some complaint risk
    5: "lights_off_empty",      # balanced: lights off unoccupied rooms only
}

ACTION_STRATEGY = {
    0: "conservative",
    1: "conservative",
    2: "conservative",
    3: "conservative",
    4: "balanced",
    5: "balanced",
}

ACTION_POWER_DELTA = {
    0: +1.5,
    1: +0.8,
    2: +0.2,
    3:  0.0,
    4: -0.5,
    5: -0.3,
}

ACTION_COMFORT_DELTA = {
    0: +2,
    1: +1,
    2: +1,
    3:  0,
    4: -1,
    5:  0,
}

ACTION_MIN_POWER = {
    0: 2.0,
    1: 2.0,
    2: 1.0,
    3: 1.0,
    4: 1.0,
    5: 0.5,
}

ACTION_TEMP_DELTA = {
    0: -2.0,
    1: -1.0,
    2:  0.0,
    3:  0.0,
    4: +0.5,
    5:  0.0,
}

def get_action_name(action_id):     return ACTIONS.get(action_id, "unknown")
def get_action_strategy(action_id): return ACTION_STRATEGY.get(action_id, "balanced")
def get_power_delta(action_id):     return ACTION_POWER_DELTA.get(action_id, 0.0)
def get_comfort_delta(action_id):   return ACTION_COMFORT_DELTA.get(action_id, 0)
def get_temp_delta(action_id):      return ACTION_TEMP_DELTA.get(action_id, 0.0)
def get_min_power(action_id):       return ACTION_MIN_POWER.get(action_id, 0.5)
def get_action_count():             return len(ACTIONS)   # now returns 6
def is_conservative(action_id):     return ACTION_STRATEGY.get(action_id) == "conservative"
def is_aggressive(action_id):       return ACTION_STRATEGY.get(action_id) == "aggressive"
def get_conservative_actions():     return [a for a, s in ACTION_STRATEGY.items() if s == "conservative"]
def get_aggressive_actions():       return [a for a, s in ACTION_STRATEGY.items() if s == "aggressive"]
def get_balanced_actions():         return [a for a, s in ACTION_STRATEGY.items() if s == "balanced"]