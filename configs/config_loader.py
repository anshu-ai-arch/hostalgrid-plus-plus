"""
configs/config_loader.py

Loads hostelgrid_config.yaml and provides dot-access to every value.

Usage anywhere in the project:
    from configs.config_loader import get_config
    cfg = get_config()

    cfg.env.num_rooms          → 10
    cfg.q_learning.gamma       → 0.9
    cfg.dqn.batch_size         → 64
    cfg.tasks.easy.n_episodes  → 500
    cfg.grading.thresholds.A   → 85
"""

import os
import yaml
from types import SimpleNamespace


CONFIG_PATH = os.path.join(os.path.dirname(__file__),
                            "hostelgrid_config.yaml")


def _to_ns(obj):
    """Recursively convert dict → SimpleNamespace for dot-access."""
    if isinstance(obj, dict):
        return SimpleNamespace(**{k: _to_ns(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_ns(i) for i in obj]
    return obj


def get_config(path: str = CONFIG_PATH) -> SimpleNamespace:
    """
    Load and return full config as dot-accessible namespace.

    Examples:
        cfg = get_config()
        cfg.env.num_rooms          → 10
        cfg.dqn.learning_rate      → 0.001
        cfg.tasks.hard.description → "Complaints + peak hours..."
        cfg.grading.thresholds.A   → 85
    """
    with open(path, "r") as f:
        raw = yaml.safe_load(f)
    return _to_ns(raw)


def get_task_config(task_name: str) -> SimpleNamespace:
    """
    Shortcut to get config for one task.

    Args:
        task_name : "easy" | "medium" | "hard"

    Returns:
        SimpleNamespace with task-specific settings
    """
    cfg  = get_config()
    task = getattr(cfg.tasks, task_name)
    return task