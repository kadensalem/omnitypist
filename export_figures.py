"""export_figures.py — Generate publication-ready figures for OmniTypist results.

Run from crtypist/ with the deeptyping conda env active:

    conda activate deeptyping
    python export_figures.py

Outputs (in figures/plots/):
    fig_mode.pdf         Experiment 1 — typing mode comparison (6-panel)
    fig_vocab_size.pdf   Experiment 2 — vocabulary size ablation (2-panel)
    fig_difficulty.pdf   Experiment 3 — difficulty assignment ablation (2-panel)
    fig_individual.pdf   Experiment 4 — individual differences (3-panel)
"""

import csv
import os
import statistics

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------

plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        8,
    "axes.titlesize":   8,
    "axes.labelsize":   8,
    "xtick.labelsize":  7,
    "ytick.labelsize":  7,
    "legend.fontsize":  7,
    "axes.spines.top":  False,
    "axes.spines.right": False,
    "axes.linewidth":   0.6,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
    "figure.dpi":       150,
    "savefig.dpi":      300,
    "savefig.bbox":     "tight",
    "savefig.pad_inches": 0.02,
})

COLORS = ["#4878CF", "#6ACC65", "#D65F5F", "#B47CC7"]
SCATTER_ALPHA = 0.35
SCATTER_SIZE  = 8
CAPSIZE       = 3
BAR_WIDTH     = 0.6
OUT_DIR       = "figures/plots"


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------

def load(path):
    rows = []
    with open(path) as f:
        for r in csv.DictReader(f):
            rows.append({k: (float(v) if k not in ("condition", "target_text", "typed_text") else v)
                         for k, v in r.items()})
    return rows


def by_condition(rows):
    out = {}
    for r in rows:
        out.setdefault(r["condition"], []).append(r)
    return out


def stats(rows, key):
    vals = [r[key] for r in rows]
    m = sum(vals) / len(vals)
    s = statistics.stdev(vals) if len(vals) > 1 else 0.0
    return m, s, vals


# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------

def bar_group(ax, conditions, data, metric, ylabel, title,
              colors=None, pct=False, order=None):
    """Single grouped bar chart on ax."""
    if order is None:
        order = conditions
    colors = colors or COLORS[:len(order)]
    means, errs, all_vals = [], [], []
    for cond in order:
        m, s, v = stats(data[cond], metric)
        if pct:
            m, s, v = m * 100, s * 100, [x * 100 for x in v]
        means.append(m)
        errs.append(s)
        all_vals.append(v)

    x = np.arange(len(order))
    bars = ax.bar(x, means, BAR_WIDTH, yerr=errs, capsize=CAPSIZE,
                  color=colors, edgecolor="white", linewidth=0.4,
                  error_kw=dict(elinewidth=0.8, ecolor="#555555"))

    # individual data points
    for i, vals in enumerate(all_vals):
        jitter = np.random.default_rng(i).uniform(-0.15, 0.15, len(vals))
        ax.scatter(i + jitter, vals, s=SCATTER_SIZE, color="black",
                   alpha=SCATTER_ALPHA, zorder=3, linewidths=0)

    ax.set_xticks(x)
    ax.set_xticklabels([c.replace("-", "\n") for c in order], fontsize=7)
    ax.set_ylabel(ylabel, fontsize=7)
    ax.set_title(title, fontsize=8, pad=3)
    ax.set_xlim(-0.6, len(order) - 0.4)
    return bars


# ---------------------------------------------------------------------------
# Figure 1 — Typing mode comparison
# ---------------------------------------------------------------------------

def fig_mode():
    rows = load("results/mode.csv")
    data = by_condition(rows)
    order  = ["single-finger", "two-thumb", "chording"]
    labels = ["Single-finger", "Two-thumb", "Chording"]
    colors = COLORS[:3]

    fig, axes = plt.subplots(2, 3, figsize=(6.5, 3.6))
    axes = axes.flatten()

    specs = [
        ("WPM",           "WPM",                   "Words per Minute",             False),
        ("IKI",           "IKI (ms)",               "Inter-keystroke Interval (ms)",False),
        ("char_error_rate","CER (%)",               "Character Error Rate (%)",      True),
        ("num_backspaces","Backspaces",             "Backspaces per Episode",        False),
        ("gaze_shift",    "Gaze shifts",            "Gaze Shifts per Episode",       False),
        ("gaze_kbd_ratio","Gaze on keyboard (%)",   "Time Gazing at Keyboard (%)",   True),
    ]

    for ax, (metric, ylabel, title, pct) in zip(axes, specs):
        bar_group(ax, order, data, metric, ylabel, title, colors=colors, pct=pct, order=order)

    # shared legend
    patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(3)]
    fig.legend(handles=patches, loc="lower center", ncol=3,
               frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.02))

    fig.suptitle("Experiment 1: Typing Mode Comparison", fontsize=9, y=1.01)
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    path = os.path.join(OUT_DIR, "fig_mode.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Figure 2 — Vocabulary size
# ---------------------------------------------------------------------------

def fig_vocab_size():
    rows = load("results/vocab_size.csv")
    data = by_condition(rows)
    order  = ["vocab-2", "vocab-5", "vocab-10"]
    colors = COLORS[:3]

    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.4))

    bar_group(axes[0], order, data, "WPM",          "WPM",
              "Words per Minute", colors=colors, order=order)
    bar_group(axes[1], order, data, "fire_rate",    "Fire rate (%)",
              "Chord Fire Rate (%)", colors=colors, pct=True, order=order)

    # Coverage scatter: one point per condition
    coverages = [statistics.mean(r["chord_coverage"] for r in data[c]) * 100
                 for c in order]
    wpms      = [statistics.mean(r["WPM"] for r in data[c]) for c in order]
    axes[2].scatter(coverages, wpms, color=colors, s=60, zorder=3)
    for i, c in enumerate(order):
        axes[2].annotate(c.replace("vocab-", "k="),
                         (coverages[i], wpms[i]),
                         textcoords="offset points", xytext=(4, 2), fontsize=6)
    axes[2].set_xlabel("Chord coverage (%)", fontsize=7)
    axes[2].set_ylabel("WPM", fontsize=7)
    axes[2].set_title("Coverage vs. Speed", fontsize=8, pad=3)
    axes[2].spines["top"].set_visible(False)
    axes[2].spines["right"].set_visible(False)

    fig.suptitle("Experiment 2: Chord Vocabulary Size", fontsize=9)
    fig.tight_layout()
    path = os.path.join(OUT_DIR, "fig_vocab_size.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Figure 3 — Difficulty assignment
# ---------------------------------------------------------------------------

def fig_difficulty():
    rows = load("results/difficulty.csv")
    data = by_condition(rows)
    order  = ["uniform-easy", "uniform-medium", "graduated", "uniform-hard"]
    colors = COLORS[:4]

    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.4))

    bar_group(axes[0], order, data, "WPM",            "WPM",
              "Words per Minute", colors=colors, order=order)
    bar_group(axes[1], order, data, "char_error_rate", "CER (%)",
              "Character Error Rate (%)", colors=colors, pct=True, order=order)
    bar_group(axes[2], order, data, "fire_rate",       "Fire rate (%)",
              "Chord Fire Rate (%)", colors=colors, pct=True, order=order)

    patches = [mpatches.Patch(color=colors[i], label=order[i].replace("-", " ").title())
               for i in range(4)]
    fig.legend(handles=patches, loc="lower center", ncol=4,
               frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, -0.05))

    fig.suptitle("Experiment 3: Chord Difficulty Assignment", fontsize=9)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    path = os.path.join(OUT_DIR, "fig_difficulty.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Figure 4 — Individual differences
# ---------------------------------------------------------------------------

def fig_individual():
    rows = load("results/individual_diff.csv")
    data = by_condition(rows)
    order  = ["high-memory", "fitted-memory", "low-memory"]
    labels = ["High memory", "Fitted (pop. avg.)", "Low memory"]
    colors = COLORS[:3]

    fig, axes = plt.subplots(1, 3, figsize=(6.5, 2.4))

    bar_group(axes[0], order, data, "WPM",          "WPM",
              "Words per Minute", colors=colors, order=order)
    bar_group(axes[1], order, data, "gaze_shift",   "Gaze shifts",
              "Gaze Shifts per Episode", colors=colors, order=order)
    bar_group(axes[2], order, data, "gaze_kbd_ratio", "Gaze on keyboard (%)",
              "Time Gazing at Keyboard (%)", colors=colors, pct=True, order=order)

    patches = [mpatches.Patch(color=colors[i], label=labels[i]) for i in range(3)]
    fig.legend(handles=patches, loc="lower center", ncol=3,
               frameon=False, fontsize=7, bbox_to_anchor=(0.5, -0.04))

    fig.suptitle("Experiment 4: Individual Differences in Working Memory", fontsize=9)
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    path = os.path.join(OUT_DIR, "fig_individual.pdf")
    fig.savefig(path)
    plt.close(fig)
    print(f"  saved {path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Exporting figures...")
    np.random.seed(0)
    fig_mode()
    fig_vocab_size()
    fig_difficulty()
    fig_individual()
    print("Done.")
