"""
utils/visualizer.py

Generates 4-panel training dashboard comparing Q-Learning vs DQN.

Usage:
    from utils.visualizer import plot_dashboard
    plot_dashboard(q_hist, dqn_hist, eval_results, mode="medium")
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

BG_OUTER = "#0f1117"
BG_INNER = "#1a1a2e"
GRID_COL = "#2a2a3e"
TC       = "#e0e0e0"
C_Q      = "#ffd166"   # yellow — Q-Learning
C_DQN    = "#00d4aa"   # teal   — DQN
C_RAND   = "#ff6b6b"   # red    — random


def _style(ax, title: str):
    ax.set_facecolor(BG_INNER)
    ax.set_title(title, color=TC, fontsize=10,
                 fontweight="bold", pad=7)
    ax.tick_params(colors=TC, labelsize=8)
    for sp in ax.spines.values():
        sp.set_color(GRID_COL)
    ax.xaxis.label.set_color(TC)
    ax.yaxis.label.set_color(TC)
    ax.grid(color=GRID_COL, linewidth=0.5)


def _smooth(arr, w=30):
    if len(arr) < w:
        return np.array(arr)
    return np.convolve(arr, np.ones(w)/w, mode="valid")


def plot_dashboard(q_hist:   dict,
                   dqn_hist: dict,
                   eval_results: dict,
                   mode: str = "medium",
                   save_dir: str = "plots") -> str:
    """
    4-panel dashboard:
        1. Reward curves (Q vs DQN with rolling mean)
        2. Rooms satisfied over training
        3. Complaints over training
        4. Final evaluation box-plot

    HOW TO READ:
        Panel 1 — teal line should end higher than yellow = DQN wins
        Panel 2 — both should trend toward 10 (all rooms satisfied)
        Panel 3 — both should trend toward 0 (no complaints)
        Panel 4 — teal box above yellow box = DQN more consistent

    Args:
        q_hist       : history dict from train_q()
        dqn_hist     : history dict from train_dqn()
        eval_results : dict from evaluate_all()
        mode         : task mode label for title
        save_dir     : directory to save PNG

    Returns:
        path to saved PNG
    """
    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, f"dashboard_{mode}.png")

    fig = plt.figure(figsize=(16, 10))
    fig.patch.set_facecolor(BG_OUTER)
    gs  = gridspec.GridSpec(2, 2, figure=fig,
                            hspace=0.45, wspace=0.35)

    q_rew   = q_hist["rewards"]
    dqn_rew = dqn_hist["rewards"]
    q_sat   = q_hist["satisfied"]
    dqn_sat = dqn_hist["satisfied"]
    q_com   = q_hist["complaints"]
    dqn_com = dqn_hist["complaints"]

    # ── Panel 1: Reward curves ─────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    _style(ax1, "1  Reward vs Episode")
    ax1.plot(q_rew,   color=C_Q,   alpha=0.15, lw=0.6)
    ax1.plot(dqn_rew, color=C_DQN, alpha=0.15, lw=0.6)
    ax1.plot(_smooth(q_rew),   color=C_Q,   lw=2,
             label="Q-Learning")
    ax1.plot(_smooth(dqn_rew), color=C_DQN, lw=2,
             label="DQN (shared)")

    if "random" in eval_results:
        r_mean = eval_results["random"]["mean"]
        ax1.axhline(r_mean, color=C_RAND, ls="--",
                    lw=1.2, label=f"Random ({r_mean:.2f})")

    ax1.set_xlabel("Episode")
    ax1.set_ylabel("Total Reward")
    ax1.legend(fontsize=7, facecolor=BG_INNER,
               labelcolor=TC, framealpha=0.7)

    # HOW TO READ annotation
    ax1.text(0.02, 0.05,
             "Teal above yellow = DQN wins",
             transform=ax1.transAxes, color=TC,
             fontsize=7, alpha=0.7)

    # ── Panel 2: Rooms satisfied ───────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    _style(ax2, "2  Rooms Satisfied (of 10)")
    ax2.plot(_smooth(q_sat),   color=C_Q,   lw=2,
             label="Q-Learning")
    ax2.plot(_smooth(dqn_sat), color=C_DQN, lw=2,
             label="DQN (shared)")
    ax2.axhline(10, color=TC, ls=":", lw=0.8, alpha=0.5,
                label="Perfect (10)")
    ax2.set_ylim(0, 11)
    ax2.set_xlabel("Episode")
    ax2.set_ylabel("Rooms Satisfied")
    ax2.legend(fontsize=7, facecolor=BG_INNER,
               labelcolor=TC, framealpha=0.7)

    # ── Panel 3: Complaints ────────────────────────────────
    ax3 = fig.add_subplot(gs[1, 0])
    _style(ax3, "3  Complaints per Episode")
    ax3.plot(_smooth(q_com),   color=C_Q,   lw=2,
             label="Q-Learning")
    ax3.plot(_smooth(dqn_com), color=C_DQN, lw=2,
             label="DQN (shared)")
    ax3.axhline(0, color=TC, ls=":", lw=0.8, alpha=0.5,
                label="Target (0)")
    ax3.set_xlabel("Episode")
    ax3.set_ylabel("Total Complaints")
    ax3.legend(fontsize=7, facecolor=BG_INNER,
               labelcolor=TC, framealpha=0.7)

    # HOW TO READ annotation
    ax3.text(0.02, 0.92,
             "Lower = fewer dissatisfied rooms",
             transform=ax3.transAxes, color=TC,
             fontsize=7, alpha=0.7)

    # ── Panel 4: Eval box-plot ─────────────────────────────
    ax4 = fig.add_subplot(gs[1, 1])
    _style(ax4, "4  Final Evaluation (100 episodes)")

    data   = []
    labels = []
    colors = []

    if "random" in eval_results:
        data.append(eval_results["random"].get(
            "rewards", [eval_results["random"]["mean"]]*100))
        labels.append("Random")
        colors.append(C_RAND)

    if "q_agent" in eval_results:
        data.append(eval_results["q_agent"]["rewards"])
        labels.append("Q-Learning")
        colors.append(C_Q)

    if "dqn_agent" in eval_results:
        data.append(eval_results["dqn_agent"]["rewards"])
        labels.append("DQN (shared)")
        colors.append(C_DQN)

    if data:
        bp = ax4.boxplot(data, tick_labels=labels,
                         patch_artist=True,
                         medianprops=dict(color="white", lw=2),
                         whiskerprops=dict(color=TC),
                         capprops=dict(color=TC),
                         flierprops=dict(markerfacecolor=TC,
                                         marker="o",
                                         markersize=3,
                                         alpha=0.4))
        for patch, col in zip(bp["boxes"], colors):
            patch.set_facecolor(col)
            patch.set_alpha(0.75)

    ax4.set_ylabel("Total Reward")
    ax4.text(0.02, 0.05,
             "Higher box = better policy",
             transform=ax4.transAxes, color=TC,
             fontsize=7, alpha=0.7)

    fig.suptitle(
        f"HostelGrid++  |  Mode: {mode.upper()}"
        f"  |  Q-Learning vs DQN (Shared Policy)",
        color=TC, fontsize=13,
        fontweight="bold", y=0.99
    )

    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=BG_OUTER)
    plt.close(fig)
    print(f"  [Visualizer] Dashboard saved → {out}")
    return out


def plot_curriculum(all_hist: dict, save_dir: str = "plots") -> str:
    """
    3-column reward comparison across easy/medium/hard.

    all_hist: {
        "easy":   {"q": q_hist, "dqn": dqn_hist},
        "medium": {...},
        "hard":   {...},
    }
    """
    os.makedirs(save_dir, exist_ok=True)
    out = os.path.join(save_dir, "curriculum.png")

    fig = plt.figure(figsize=(18, 5))
    fig.patch.set_facecolor(BG_OUTER)
    gs  = gridspec.GridSpec(1, 3, figure=fig,
                            hspace=0.3, wspace=0.35)

    for col, mode in enumerate(("easy", "medium", "hard")):
        ax  = fig.add_subplot(gs[0, col])
        _style(ax, f"{mode.upper()} — Reward")
        q_r   = all_hist[mode]["q"]["rewards"]
        dqn_r = all_hist[mode]["dqn"]["rewards"]
        ax.plot(_smooth(q_r),   color=C_Q,   lw=2,
                label="Q-Learning")
        ax.plot(_smooth(dqn_r), color=C_DQN, lw=2,
                label="DQN (shared)")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Total Reward")
        ax.legend(fontsize=7, facecolor=BG_INNER,
                  labelcolor=TC, framealpha=0.7)

    fig.suptitle(
        "HostelGrid++  |  Curriculum: Easy → Medium → Hard",
        color=TC, fontsize=13,
        fontweight="bold", y=1.02
    )
    plt.savefig(out, dpi=150, bbox_inches="tight",
                facecolor=BG_OUTER)
    plt.close(fig)
    print(f"  [Visualizer] Curriculum plot saved → {out}")
    return out