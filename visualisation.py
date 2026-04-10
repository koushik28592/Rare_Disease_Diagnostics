"""
=============================================================================
visualisation.py
Rare Disease Diagnosis using MAML + RL²
=============================================================================
Generates all project figures:
  1. Training curves (loss, accuracy) for MAML
  2. RL² reward and accuracy progression
  3. Per-step accuracy curve (RL² memory effect)
  4. Comparison bar chart (Standard AI vs MAML vs RL² vs Hybrid)
  5. Per-disease accuracy heatmap
  6. Symptom co-occurrence heatmap (EDA)
  7. System architecture diagram (text-based)
=============================================================================
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import pandas as pd
import seaborn as sns
import os

# ── Style ────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "figure.facecolor": "#0D1B2A",
    "axes.facecolor":   "#0D1B2A",
    "axes.edgecolor":   "#C9A84C",
    "axes.labelcolor":  "#FFFFFF",
    "xtick.color":      "#FFFFFF",
    "ytick.color":      "#FFFFFF",
    "text.color":       "#FFFFFF",
    "grid.color":       "#1E3A5F",
    "grid.linestyle":   "--",
    "grid.alpha":       0.4,
    "font.family":      "DejaVu Sans",
    "font.size":        10,
    "legend.facecolor": "#0D1B2A",
    "legend.edgecolor": "#C9A84C",
    "figure.dpi":       120,
})

GOLD   = "#C9A84C"
BLUE   = "#4A90D9"
GREEN  = "#5DBB63"
RED    = "#E05C5C"
PURPLE = "#9B59B6"
ORANGE = "#E67E22"

SAVE_DIR = "plots"
os.makedirs(SAVE_DIR, exist_ok=True)


def _save(fig, name):
    path = os.path.join(SAVE_DIR, name)
    fig.savefig(path, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    print(f"  [Plot] Saved → {path}")
    return path


# ─────────────────────────────────────────────────────────────────────────────
# 1. EDA: Dataset Overview
# ─────────────────────────────────────────────────────────────────────────────
def plot_dataset_overview(df, symptom_list):
    """Shows disease distribution and top symptoms."""
    fig = plt.figure(figsize=(18, 7))
    gs = GridSpec(1, 2, figure=fig, wspace=0.35)
    fig.suptitle("Dataset Overview — Disease & Symptom Distribution",
                 fontsize=14, color=GOLD, fontweight="bold", y=1.02)

    # Disease record count (all equal = 120)
    ax1 = fig.add_subplot(gs[0])
    diseases = df["Disease"].str.strip().value_counts()
    colors = [BLUE if i < 35 else RED for i in range(len(diseases))]
    bars = ax1.barh(diseases.index, diseases.values, color=colors, edgecolor="#0D1B2A", linewidth=0.5)
    ax1.set_xlabel("Number of Records", color="#FFFFFF")
    ax1.set_title("Records per Disease\n(Red = held-out as Rare)", fontsize=10, color=GOLD)
    ax1.axvline(120, color=GOLD, linestyle="--", alpha=0.7, label="120 records")
    ax1.legend(fontsize=8)
    ax1.tick_params(labelsize=6)

    # Top 20 symptoms
    ax2 = fig.add_subplot(gs[1])
    sym_cols = [c for c in df.columns if c.startswith("Symptom_")]
    all_syms = []
    for col in sym_cols:
        all_syms.extend(df[col].str.strip().tolist())
    from collections import Counter
    sym_counts = Counter([s for s in all_syms if s])
    top20 = pd.Series(dict(sym_counts.most_common(20)))
    ax2.barh(top20.index, top20.values, color=GREEN, edgecolor="#0D1B2A")
    ax2.set_xlabel("Frequency", color="#FFFFFF")
    ax2.set_title("Top 20 Most Common Symptoms", fontsize=10, color=GOLD)
    ax2.tick_params(labelsize=8)

    return _save(fig, "01_dataset_overview.png")


# ─────────────────────────────────────────────────────────────────────────────
# 2. MAML Training Curves
# ─────────────────────────────────────────────────────────────────────────────
def plot_maml_training(train_history, val_history):
    """MAML outer-loop loss and accuracy over meta-iterations."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle("MAML Meta-Training Progress", fontsize=14,
                 color=GOLD, fontweight="bold")

    # Smoothing helper
    def smooth(vals, w=10):
        return np.convolve(vals, np.ones(w) / w, mode="valid")

    iters = range(len(train_history["loss"]))

    # Loss
    ax = axes[0]
    raw = train_history["loss"]
    ax.plot(iters, raw, color=BLUE, alpha=0.3, linewidth=0.8)
    ax.plot(range(len(smooth(raw))), smooth(raw), color=BLUE, linewidth=2, label="Train Loss")
    ax.set_xlabel("Meta-Iteration")
    ax.set_ylabel("NLL Loss")
    ax.set_title("Outer-Loop Query Loss", color=GOLD)
    ax.legend()
    ax.grid(True)

    # Train Accuracy
    ax = axes[1]
    raw = [a * 100 for a in train_history["acc"]]
    ax.plot(iters, raw, color=GREEN, alpha=0.3, linewidth=0.8)
    ax.plot(range(len(smooth(raw))), smooth(raw), color=GREEN, linewidth=2, label="Train Acc")
    ax.set_xlabel("Meta-Iteration")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Meta-Train Query Accuracy", color=GOLD)
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True)

    # Val k-shot accuracy
    ax = axes[2]
    if val_history.get("iteration"):
        for key, color, label in [
            ("acc_5shot",  RED,    "5-shot"),
            ("acc_10shot", GOLD,   "10-shot"),
            ("acc_15shot", PURPLE, "15-shot"),
        ]:
            if key in val_history:
                ax.plot(val_history["iteration"], val_history[key],
                        color=color, linewidth=2, marker="o", markersize=4,
                        label=label)
    ax.set_xlabel("Meta-Iteration")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Rare Disease k-Shot Accuracy\n(held-out diseases)", color=GOLD)
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True)

    plt.tight_layout()
    return _save(fig, "02_maml_training.png")


# ─────────────────────────────────────────────────────────────────────────────
# 3. RL² Training Curves
# ─────────────────────────────────────────────────────────────────────────────
def plot_rl2_training(train_history, val_history):
    """RL² episode reward, accuracy, and PPO losses."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("RL² Training Progress (PPO)", fontsize=14,
                 color=GOLD, fontweight="bold")

    def smooth(vals, w=20):
        if len(vals) < w:
            return vals
        return np.convolve(vals, np.ones(w) / w, mode="valid")

    ep = range(len(train_history["episode_reward"]))

    # Episode Reward
    ax = axes[0, 0]
    raw = train_history["episode_reward"]
    ax.plot(ep, raw, color=BLUE, alpha=0.2, linewidth=0.5)
    sm = smooth(raw)
    ax.plot(range(len(sm)), sm, color=BLUE, linewidth=2, label="Smoothed Reward")
    ax.axhline(0, color=GOLD, linestyle="--", alpha=0.5)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Total Episode Reward")
    ax.set_title("Episode Reward", color=GOLD)
    ax.legend()
    ax.grid(True)

    # Episode Accuracy
    ax = axes[0, 1]
    raw = train_history["episode_accuracy"]
    ax.plot(ep, raw, color=GREEN, alpha=0.2, linewidth=0.5)
    sm = smooth(raw)
    ax.plot(range(len(sm)), sm, color=GREEN, linewidth=2, label="Smoothed Acc")
    n_way = 5
    ax.axhline(100 / n_way, color=RED, linestyle="--",
               label=f"Random ({100/n_way:.0f}%)", alpha=0.7)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Diagnostic Accuracy per Episode", color=GOLD)
    ax.set_ylim(0, 105)
    ax.legend()
    ax.grid(True)

    # Policy Loss
    ax = axes[1, 0]
    raw = train_history["policy_loss"]
    sm = smooth(raw)
    ax.plot(range(len(sm)), sm, color=RED, linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Policy Loss")
    ax.set_title("PPO Policy Loss", color=GOLD)
    ax.grid(True)

    # Entropy
    ax = axes[1, 1]
    raw = train_history["entropy"]
    sm = smooth(raw)
    ax.plot(range(len(sm)), sm, color=ORANGE, linewidth=2, label="Entropy")
    ax2 = ax.twinx()
    raw_v = train_history["value_loss"]
    sm_v = smooth(raw_v)
    ax2.plot(range(len(sm_v)), sm_v, color=PURPLE, linewidth=2,
             linestyle="--", label="Value Loss")
    ax2.set_ylabel("Value Loss", color=PURPLE)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Entropy", color=ORANGE)
    ax.set_title("Entropy & Value Loss", color=GOLD)
    ax.grid(True)

    plt.tight_layout()
    return _save(fig, "03_rl2_training.png")


# ─────────────────────────────────────────────────────────────────────────────
# 4. RL² Per-Step Accuracy (Memory Effect)
# ─────────────────────────────────────────────────────────────────────────────
def plot_rl2_memory_effect(step_accuracy_dict, n_way=5):
    """
    Shows how RL² accuracy improves within an episode as the GRU
    accumulates more patient evidence — the core memory effect.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    fig.suptitle("RL² GRU Memory Effect — Accuracy vs Patient Encounters",
                 fontsize=13, color=GOLD, fontweight="bold")

    steps = sorted(step_accuracy_dict.keys())
    accs = [step_accuracy_dict[t] for t in steps]

    ax.plot(steps, accs, color=GOLD, linewidth=2.5, marker="o", markersize=4)
    ax.fill_between(steps, accs, alpha=0.15, color=GOLD)
    ax.axhline(100 / n_way, color=RED, linestyle="--",
               label=f"Random baseline ({100/n_way:.0f}%)", alpha=0.7)
    ax.axhline(np.max(accs), color=GREEN, linestyle=":", alpha=0.6,
               label=f"Peak accuracy ({np.max(accs):.1f}%)")

    # Shade early vs late regions
    mid = len(steps) // 2
    ax.axvspan(steps[0], steps[mid], alpha=0.08, color=BLUE, label="Early phase")
    ax.axvspan(steps[mid], steps[-1], alpha=0.08, color=GREEN, label="Late phase")

    ax.set_xlabel("Patient Encounter Number (within episode)")
    ax.set_ylabel("Diagnostic Accuracy (%)")
    ax.set_title("Accuracy improves as GRU accumulates clinical history →",
                 fontsize=9, color="#AAAAAA")
    ax.legend()
    ax.grid(True)
    ax.set_ylim(0, 105)

    plt.tight_layout()
    return _save(fig, "04_rl2_memory_effect.png")


# ─────────────────────────────────────────────────────────────────────────────
# 5. Comparison Chart
# ─────────────────────────────────────────────────────────────────────────────
def plot_comparison(baseline_results, maml_results, rl2_acc,
                    hybrid_acc=None, k_shots=(5, 10, 15)):
    """Bar chart comparing all four approaches at multiple k-shot settings."""
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.suptitle("Performance Comparison: Standard AI vs MAML vs RL² vs Hybrid",
                 fontsize=13, color=GOLD, fontweight="bold")

    k_values = sorted(k_shots)
    x = np.arange(len(k_values))
    width = 0.2

    methods = {
        "Standard AI":   (baseline_results, BLUE),
        "MAML Only":     (maml_results,     GREEN),
        "RL² Only":      ({k: (rl2_acc, 0) for k in k_values}, RED),
    }
    if hybrid_acc is not None:
        methods["MAML + RL²"] = ({k: (hybrid_acc, 0) for k in k_values}, GOLD)

    for i, (label, (results, color)) in enumerate(methods.items()):
        means = [results.get(k, (0, 0))[0] for k in k_values]
        stds  = [results.get(k, (0, 0))[1] for k in k_values]
        offset = (i - len(methods) / 2 + 0.5) * width
        bars = ax.bar(x + offset, means, width, label=label,
                      color=color, edgecolor="#0D1B2A", alpha=0.85,
                      yerr=stds, capsize=4, error_kw={"ecolor": "white"})
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 1,
                    f"{mean:.1f}%", ha="center", va="bottom",
                    fontsize=7, color="white")

    ax.set_xticks(x)
    ax.set_xticklabels([f"{k}-Shot" for k in k_values])
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 115)
    ax.legend(loc="upper left")
    ax.grid(True, axis="y")

    plt.tight_layout()
    return _save(fig, "05_comparison_chart.png")


# ─────────────────────────────────────────────────────────────────────────────
# 6. Per-Disease MAML Accuracy
# ─────────────────────────────────────────────────────────────────────────────
def plot_per_disease_accuracy(per_disease_results):
    """Horizontal bar chart showing MAML accuracy per rare disease."""
    if not per_disease_results:
        return None

    fig, ax = plt.subplots(figsize=(10, max(4, len(per_disease_results) * 0.6)))
    fig.suptitle("Per-Disease MAML Accuracy (10-shot, Rare Diseases)",
                 fontsize=13, color=GOLD, fontweight="bold")

    diseases = list(per_disease_results.keys())
    means = [per_disease_results[d][0] for d in diseases]
    stds  = [per_disease_results[d][1] for d in diseases]

    # Sort by accuracy
    sorted_idx = np.argsort(means)
    diseases = [diseases[i] for i in sorted_idx]
    means = [means[i] for i in sorted_idx]
    stds  = [stds[i] for i in sorted_idx]

    colors = [GREEN if m >= 70 else ORANGE if m >= 50 else RED for m in means]
    bars = ax.barh(diseases, means, xerr=stds, color=colors,
                   edgecolor="#0D1B2A", capsize=4,
                   error_kw={"ecolor": "white"})

    ax.axvline(50, color=GOLD, linestyle="--", alpha=0.6, label="50% baseline")
    for bar, mean in zip(bars, means):
        ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                f"{mean:.1f}%", va="center", fontsize=8)

    ax.set_xlabel("Accuracy (%)")
    ax.set_xlim(0, 115)
    ax.legend()
    ax.grid(True, axis="x")

    plt.tight_layout()
    return _save(fig, "06_per_disease_accuracy.png")


# ─────────────────────────────────────────────────────────────────────────────
# 7. Summary Dashboard
# ─────────────────────────────────────────────────────────────────────────────
def plot_summary_dashboard(maml_results, rl2_results, train_history_maml,
                            train_history_rl2):
    """A single combined dashboard figure for the report."""
    fig = plt.figure(figsize=(20, 12))
    gs = GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)
    fig.suptitle("Rare Disease Diagnosis — MAML + RL² Summary Dashboard",
                 fontsize=15, color=GOLD, fontweight="bold", y=0.98)

    def smooth(vals, w=15):
        if len(vals) < w:
            return vals
        return np.convolve(vals, np.ones(w) / w, mode="valid")

    # ── MAML loss ─────────────────────────────────────────────────────────
    ax1 = fig.add_subplot(gs[0, 0])
    raw = train_history_maml.get("loss", [])
    sm = smooth(raw)
    ax1.plot(range(len(sm)), sm, color=BLUE, linewidth=2)
    ax1.set_title("MAML — Query Loss", color=GOLD)
    ax1.set_xlabel("Meta-Iteration")
    ax1.set_ylabel("NLL Loss")
    ax1.grid(True)

    # ── MAML accuracy ─────────────────────────────────────────────────────
    ax2 = fig.add_subplot(gs[0, 1])
    raw = [a * 100 for a in train_history_maml.get("acc", [])]
    sm = smooth(raw)
    ax2.plot(range(len(sm)), sm, color=GREEN, linewidth=2)
    ax2.set_title("MAML — Train Accuracy", color=GOLD)
    ax2.set_xlabel("Meta-Iteration")
    ax2.set_ylabel("Accuracy (%)")
    ax2.set_ylim(0, 105)
    ax2.grid(True)

    # ── k-shot bar chart ───────────────────────────────────────────────────
    ax3 = fig.add_subplot(gs[0, 2])
    ks = sorted(maml_results.keys())
    ms = [maml_results[k][0] for k in ks]
    ss = [maml_results[k][1] for k in ks]
    bars = ax3.bar([f"{k}-shot" for k in ks], ms, yerr=ss,
                   color=[GOLD, GREEN, BLUE][:len(ks)],
                   edgecolor="#0D1B2A", capsize=6)
    for b, m in zip(bars, ms):
        ax3.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                 f"{m:.1f}%", ha="center", fontsize=9, color="white")
    ax3.set_title("MAML — k-Shot Accuracy (Rare Diseases)", color=GOLD)
    ax3.set_ylabel("Accuracy (%)")
    ax3.set_ylim(0, 115)
    ax3.grid(True, axis="y")

    # ── RL² reward ────────────────────────────────────────────────────────
    ax4 = fig.add_subplot(gs[1, 0])
    raw = train_history_rl2.get("episode_reward", [])
    sm = smooth(raw, w=30)
    ax4.plot(range(len(sm)), sm, color=ORANGE, linewidth=2)
    ax4.axhline(0, color=RED, linestyle="--", alpha=0.5)
    ax4.set_title("RL² — Episode Reward", color=GOLD)
    ax4.set_xlabel("Episode")
    ax4.set_ylabel("Total Reward")
    ax4.grid(True)

    # ── RL² accuracy ──────────────────────────────────────────────────────
    ax5 = fig.add_subplot(gs[1, 1])
    raw = train_history_rl2.get("episode_accuracy", [])
    sm = smooth(raw, w=30)
    ax5.plot(range(len(sm)), sm, color=PURPLE, linewidth=2)
    ax5.axhline(20, color=RED, linestyle="--", alpha=0.5, label="Random (20%)")
    ax5.set_title("RL² — Diagnostic Accuracy", color=GOLD)
    ax5.set_xlabel("Episode")
    ax5.set_ylabel("Accuracy (%)")
    ax5.set_ylim(0, 105)
    ax5.legend(fontsize=8)
    ax5.grid(True)

    # ── Memory effect ──────────────────────────────────────────────────────
    ax6 = fig.add_subplot(gs[1, 2])
    if rl2_results.get("step_accuracy"):
        steps = sorted(rl2_results["step_accuracy"].keys())
        accs  = [rl2_results["step_accuracy"][t] for t in steps]
        ax6.plot(steps, accs, color=GOLD, linewidth=2.5, marker="o", markersize=3)
        ax6.fill_between(steps, accs, alpha=0.15, color=GOLD)
        ax6.axhline(20, color=RED, linestyle="--", alpha=0.6)
    ax6.set_title("RL² — GRU Memory Effect\n(accuracy vs patient #)", color=GOLD)
    ax6.set_xlabel("Patient Encounter #")
    ax6.set_ylabel("Accuracy (%)")
    ax6.set_ylim(0, 105)
    ax6.grid(True)

    return _save(fig, "00_summary_dashboard.png")
