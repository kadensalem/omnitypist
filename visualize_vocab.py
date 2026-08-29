"""visualize_vocab.py — Chord vocabulary coverage comparison (3 rows × 3 cols).

Each row is a different sentence.
Columns: Low coverage | Medium coverage | High coverage

Run with:
    conda activate deeptyping
    cd crtypist/
    python visualize_vocab.py
"""

import os
import sys
import difflib
import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(__file__))
from typing_env.internal_env import InternalEnv
from setting import KEYS, PLACES, CHARS
from parameters import parameters as DEFAULT_PARAMS
from data.chord_vocab import CHORD_VOCAB, CHORD_DIFFICULTY
from stable_baselines3 import PPO

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

OUT_PATH       = "figures/plots/fig_vocab.pdf"
CROP_Y_TOP     = 25   # just trim the status bar; shows text field + keyboard
KBD_IMG_FOLDER = "kbd1k/keyboard_dataset/"
KBD_LABEL_FILE = "kbd1k/keyboard_label.csv"
FINGER_PATH    = "outputs/finger_agent.pt"
VISION_PATH    = "outputs/vision_agent.pt"
CKPT_CHORD     = "outputs/supervisor_agent_two_thumb_chord.pt"

MAX_RETRIES  = 15
MIN_ACCURACY = 0.90

# ---------------------------------------------------------------------------
# Row definitions
# Each row: one sentence + three vocab conditions (low / med / high).
# Sentence chord hits noted in comments.
# ---------------------------------------------------------------------------

ROWS = [
    # ---- Row 1 -------------------------------------------------------
    # Sentence chord hits: "the", "and", "for", "with", "all", "have"  (6)
    {
        "sentence": "we have the best for all and with care",
        "low": {
            "n_matches": 1,
            "vocab": {
                ("h", "t"): "the",    # ✓
                ("b", "u"): "but",    # ✗
                ("c", "o"): "com",    # ✗
                ("i", "t"): "tion",   # ✗
                ("e", "n"): "ent",    # ✗
            },
            "diff": {
                ("h", "t"): 0.1, ("b", "u"): 0.4,
                ("c", "o"): 0.6, ("i", "t"): 0.6, ("e", "n"): 0.7,
            },
        },
        "med": {
            "n_matches": 3,
            "vocab": {
                ("h", "t"): "the",    # ✓
                ("a", "n"): "and",    # ✓
                ("f", "o"): "for",    # ✓
                ("c", "o"): "com",    # ✗
                ("i", "t"): "tion",   # ✗
                ("e", "n"): "ent",    # ✗
            },
            "diff": {
                ("h", "t"): 0.1, ("a", "n"): 0.1, ("f", "o"): 0.3,
                ("c", "o"): 0.6, ("i", "t"): 0.6, ("e", "n"): 0.7,
            },
        },
        "high": {
            "n_matches": 6,
            "vocab": {
                ("h", "t"): "the",    # ✓
                ("a", "n"): "and",    # ✓
                ("f", "o"): "for",    # ✓
                ("i", "w"): "with",   # ✓
                ("a", "l"): "all",    # ✓
                ("h", "v"): "have",   # ✓
            },
            "diff": {
                ("h", "t"): 0.1, ("a", "n"): 0.1, ("f", "o"): 0.3,
                ("i", "w"): 0.3, ("a", "l"): 0.4, ("h", "v"): 0.8,
            },
        },
    },

    # ---- Row 2 -------------------------------------------------------
    # Sentence: weather/nature
    # Chord hits: "but", "the", "and", "with", "all", "for"  (6)
    {
        "sentence": "but the rain and wind with all their force for change",
        "low": {
            "n_matches": 1,
            "vocab": {
                ("b", "u"): "but",    # ✓
                ("c", "o"): "com",    # ✗
                ("i", "t"): "tion",   # ✗
                ("e", "n"): "ent",    # ✗
                ("h", "v"): "have",   # ✗
            },
            "diff": {
                ("b", "u"): 0.4, ("c", "o"): 0.6,
                ("i", "t"): 0.6, ("e", "n"): 0.7, ("h", "v"): 0.8,
            },
        },
        "med": {
            "n_matches": 3,
            "vocab": {
                ("b", "u"): "but",    # ✓
                ("h", "t"): "the",    # ✓
                ("a", "n"): "and",    # ✓
                ("c", "o"): "com",    # ✗
                ("i", "t"): "tion",   # ✗
                ("e", "n"): "ent",    # ✗
            },
            "diff": {
                ("b", "u"): 0.4, ("h", "t"): 0.1, ("a", "n"): 0.1,
                ("c", "o"): 0.6, ("i", "t"): 0.6, ("e", "n"): 0.7,
            },
        },
        "high": {
            "n_matches": 6,
            "vocab": {
                ("b", "u"): "but",    # ✓
                ("h", "t"): "the",    # ✓
                ("a", "n"): "and",    # ✓
                ("i", "w"): "with",   # ✓
                ("a", "l"): "all",    # ✓
                ("f", "o"): "for",    # ✓
            },
            "diff": {
                ("b", "u"): 0.4, ("h", "t"): 0.1, ("a", "n"): 0.1,
                ("i", "w"): 0.3, ("a", "l"): 0.4, ("f", "o"): 0.3,
            },
        },
    },

    # ---- Row 3 -------------------------------------------------------
    # Sentence: learning/growth
    # Chord hits: "the" (x2), "and", "with", "all", "for"  (5 unique)
    {
        "sentence": "the knowledge and skill with all the effort for growth",
        "low": {
            "n_matches": 1,
            "vocab": {
                ("h", "t"): "the",    # ✓
                ("c", "o"): "com",    # ✗
                ("i", "t"): "tion",   # ✗
                ("e", "n"): "ent",    # ✗
                ("h", "v"): "have",   # ✗
            },
            "diff": {
                ("h", "t"): 0.1, ("c", "o"): 0.6,
                ("i", "t"): 0.6, ("e", "n"): 0.7, ("h", "v"): 0.8,
            },
        },
        "med": {
            "n_matches": 3,
            "vocab": {
                ("h", "t"): "the",    # ✓
                ("a", "n"): "and",    # ✓
                ("i", "w"): "with",   # ✓
                ("c", "o"): "com",    # ✗
                ("i", "t"): "tion",   # ✗
                ("e", "n"): "ent",    # ✗
            },
            "diff": {
                ("h", "t"): 0.1, ("a", "n"): 0.1, ("i", "w"): 0.3,
                ("c", "o"): 0.6, ("i", "t"): 0.6, ("e", "n"): 0.7,
            },
        },
        "high": {
            "n_matches": 5,
            "vocab": {
                ("h", "t"): "the",    # ✓ (fires twice)
                ("a", "n"): "and",    # ✓
                ("i", "w"): "with",   # ✓
                ("a", "l"): "all",    # ✓
                ("f", "o"): "for",    # ✓
            },
            "diff": {
                ("h", "t"): 0.1, ("a", "n"): 0.1, ("i", "w"): 0.3,
                ("a", "l"): 0.4, ("f", "o"): 0.3,
            },
        },
    },
]

COL_LABELS = ["Low coverage", "Medium coverage", "High coverage"]
COL_KEYS   = ["low", "med", "high"]

# ---------------------------------------------------------------------------
# Vocab patching
# ---------------------------------------------------------------------------

def patch_vocab(new_vocab, new_diff):
    orig_vocab = dict(CHORD_VOCAB)
    orig_diff  = dict(CHORD_DIFFICULTY)
    CHORD_VOCAB.clear();      CHORD_VOCAB.update(new_vocab)
    CHORD_DIFFICULTY.clear(); CHORD_DIFFICULTY.update(new_diff)
    return orig_vocab, orig_diff


def restore_vocab(orig_vocab, orig_diff):
    CHORD_VOCAB.clear();      CHORD_VOCAB.update(orig_vocab)
    CHORD_DIFFICULTY.clear(); CHORD_DIFFICULTY.update(orig_diff)


# ---------------------------------------------------------------------------
# Recording wrapper
# ---------------------------------------------------------------------------

class RecordingEnv:
    def __init__(self, env):
        self.env  = env
        self.taps = []
        self.gaze_fixations = []
        self._t   = 0

    def reset(self, **kwargs):
        self._t = 0
        self.taps = []
        self.gaze_fixations = []
        return self.env.reset(**kwargs)

    def step(self, action):
        prev_typed      = self.env.typed_text
        prev_vis_in_act = self.env.vision_in_action
        prev_vis_goal   = self.env.vision_goal

        obs, reward, done, info = self.env.step(action)
        self._t += 1
        new_typed = self.env.typed_text

        if (not prev_vis_in_act
                and self.env.vision_in_action
                and self.env.vision_goal != prev_vis_goal):
            self.gaze_fixations.append({
                "t": self._t,
                "x": float(self.env.gaze[0]),
                "y": float(self.env.gaze[1]),
                "is_proofread": self.env.vision_goal == "input_box",
            })

        delta = len(new_typed) - len(prev_typed)
        if delta > 1:
            for hand, attr in [("left", "left_finger"), ("right", "right_finger")]:
                pos = getattr(self.env, attr)
                self.taps.append({"t": self._t, "hand": hand,
                                   "x": float(pos[0]), "y": float(pos[1]),
                                   "is_chord": True, "is_backspace": False})
        elif delta == 1:
            hand = getattr(self.env, "active_finger", None) or "right"
            pos  = self.env.left_finger if hand == "left" else self.env.right_finger
            self.taps.append({"t": self._t, "hand": hand,
                               "x": float(pos[0]), "y": float(pos[1]),
                               "is_chord": False, "is_backspace": False})
        elif delta < 0:
            hand = getattr(self.env, "active_finger", None) or "right"
            pos  = self.env.left_finger if hand == "left" else self.env.right_finger
            self.taps.append({"t": self._t, "hand": hand,
                               "x": float(pos[0]), "y": float(pos[1]),
                               "is_chord": False, "is_backspace": True})

        return obs, reward, done, info

    def __getattr__(self, name):
        return getattr(self.env, name)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def run_episode(env, supervisor, sentence, vocab_cfg):
    orig_v, orig_d = patch_vocab(vocab_cfg["vocab"], vocab_cfg["diff"])
    best_result = None
    best_ratio  = 0.0
    try:
        for attempt in range(MAX_RETRIES):
            rec = RecordingEnv(env)
            env.img_index = -1
            obs = rec.reset(
                target_text=sentence,
                parameters=list(DEFAULT_PARAMS),
                reset_kbd=True,
            )
            done = False
            while not done:
                action, _ = supervisor.predict(obs, deterministic=True)
                obs, _, done, _ = rec.step(action)

            typed        = env.typed_text
            screenshot   = env.screenshot.copy()   # capture after episode (shows typed text)
            chords_fired = env.chords_fired
            ratio        = difflib.SequenceMatcher(None, typed, sentence).ratio()

            if ratio > best_ratio:
                best_ratio  = ratio
                best_result = (rec, screenshot, typed, chords_fired)

            if ratio >= MIN_ACCURACY:
                break
            if attempt < MAX_RETRIES - 1:
                print(f"    retry {attempt+1} (acc={ratio:.2f}  '{typed[:25]}')")
    finally:
        restore_vocab(orig_v, orig_d)

    return best_result


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def draw_panel(ax, rec, screenshot, chord_label=None, crop_y=CROP_Y_TOP):
    img = screenshot.crop((0, crop_y, screenshot.width, screenshot.height))
    W, H = img.width, img.height

    ax.imshow(np.array(img.convert("RGB")), origin="upper", aspect="auto",
              extent=[0, W, H, 0])
    ax.set_xlim(0, W)
    ax.set_ylim(H, 0)
    ax.set_xticks([])
    ax.set_yticks([])

    if chord_label:
        ax.text(0.97, 0.97, chord_label,
                transform=ax.transAxes, fontsize=6.5,
                ha="right", va="top", color="#222222",
                bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.8, ec="none"))

    taps  = rec.taps
    gazes = rec.gaze_fixations

    def shift_y(y):
        return y - crop_y

    T = max((e["t"] for e in taps + gazes), default=1) + 1

    def tcol(t, cmap, lo=0.3):
        return cmap(lo + (1.0 - lo) * t / T)

    blues = cm.get_cmap("Blues")
    reds  = cm.get_cmap("Reds")

    # Gaze fixations (key + proofread treated as one)
    gaze_sorted = sorted(gazes, key=lambda g: g["t"])
    xs_g = [g["x"] for g in gaze_sorted]
    ys_g = [shift_y(g["y"]) for g in gaze_sorted]
    ts_g = [g["t"] for g in gaze_sorted]
    if xs_g:
        if len(xs_g) > 1:
            for i in range(len(xs_g) - 1):
                ax.plot([xs_g[i], xs_g[i+1]], [ys_g[i], ys_g[i+1]],
                        color="#cc7700", lw=0.7, alpha=0.4, zorder=2)
        ax.scatter(xs_g, ys_g, c=ts_g, cmap="YlOrBr", vmin=0, vmax=T,
                   s=22, marker="D", zorder=3, linewidths=0)

    # Taps
    for hand, cmap in [("left", blues), ("right", reds)]:
        seq = [e for e in taps if e["hand"] == hand
               and not e["is_chord"] and not e["is_backspace"]]
        xs = [e["x"]          for e in seq]
        ys = [shift_y(e["y"]) for e in seq]
        ts = [e["t"]          for e in seq]
        if not xs:
            continue
        for i in range(len(xs) - 1):
            mid = tcol((seq[i]["t"] + seq[i+1]["t"]) / 2, cmap)
            ax.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]],
                    color=mid, lw=1.0, alpha=0.55, zorder=2)
        ax.scatter(xs, ys, c=ts, cmap=cmap.name, vmin=0, vmax=T,
                   s=55, marker="o", edgecolors="white",
                   linewidths=0.4, zorder=4)

    # Backspaces
    back = [e for e in taps if e["is_backspace"]]
    if back:
        ax.scatter([e["x"] for e in back],
                   [shift_y(e["y"]) for e in back],
                   c="#888888", s=28, marker="x",
                   linewidths=1.1, zorder=4, alpha=0.65)

    # Chord rings
    chord_times = sorted(set(e["t"] for e in taps if e["is_chord"]))
    for ct in chord_times:
        pair = [e for e in taps if e["is_chord"] and e["t"] == ct]
        if len(pair) == 2:
            ax.plot([pair[0]["x"],          pair[1]["x"]],
                    [shift_y(pair[0]["y"]), shift_y(pair[1]["y"])],
                    color="gold", lw=1.5, alpha=0.85, zorder=5, ls="--")
    chord_taps = [e for e in taps if e["is_chord"]]
    if chord_taps:
        ax.scatter([e["x"]          for e in chord_taps],
                   [shift_y(e["y"]) for e in chord_taps],
                   s=210, facecolors="none", edgecolors="gold",
                   linewidths=1.8, zorder=6)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs("figures/plots", exist_ok=True)

    print("Loading supervisor...")
    supervisor = PPO.load(CKPT_CHORD, device="cpu")

    print("Building env...")
    env = InternalEnv(
        img_folder=KBD_IMG_FOLDER,
        position_file=KBD_LABEL_FILE,
        text_path="./data/sentences.txt",
        vision_path=VISION_PATH,
        finger_path=FINGER_PATH,
        chars=CHARS, places=PLACES, keys=KEYS,
        two_thumb=True, chords=True, chords_active=True,
    )

    # Run all 9 episodes
    episode_results = []  # (row_idx, col_key, rec, ss, typed, cf, n_matches)
    for row_idx, row in enumerate(ROWS):
        sentence = row["sentence"]
        print(f"\nRow {row_idx+1}: \"{sentence}\"")
        for col_key in COL_KEYS:
            cfg = row[col_key]
            rec, ss, typed, cf = run_episode(env, supervisor, sentence, cfg)
            print(f"  {col_key:<4}: '{typed}'  taps={len(rec.taps)}  chords_fired={cf}")

            episode_results.append((row_idx, col_key, rec, ss, typed, cf, cfg["n_matches"]))

    # ---- Figure layout ----
    N_ROWS    = len(ROWS)
    N_COLS    = 3
    crop_h    = 455 - CROP_Y_TOP
    aspect    = crop_h / 256.0
    panel_w   = 2.2
    panel_h   = panel_w * aspect
    lbl_h_in  = 0.38
    hdr_h_in  = 0.30
    fig_w     = N_COLS * panel_w + 0.1
    fig_h     = N_ROWS * (panel_h + lbl_h_in) + hdr_h_in + 0.4

    fig = plt.figure(figsize=(fig_w, fig_h))

    height_ratios = [hdr_h_in / panel_h]
    for _ in range(N_ROWS):
        height_ratios += [lbl_h_in / panel_h, 1.0]

    gs = fig.add_gridspec(
        1 + N_ROWS * 2, N_COLS,
        height_ratios=height_ratios,
        hspace=0.0, wspace=0.04,
        left=0.01, right=0.99,
        top=0.99, bottom=0.07,
    )

    # Column headers in row 0
    for col_idx, label in enumerate(COL_LABELS):
        ax_hdr = fig.add_subplot(gs[0, col_idx])
        ax_hdr.set_axis_off()
        ax_hdr.text(0.5, 0.3, label,
                    ha="center", va="center", fontsize=8.5,
                    fontweight="bold", transform=ax_hdr.transAxes)

    axes_data = {}

    for row_idx, row in enumerate(ROWS):
        gs_row = 1 + row_idx * 2

        ax_lbl = fig.add_subplot(gs[gs_row, :])
        ax_lbl.set_axis_off()
        ax_lbl.text(0.5, 0.5, f'"{row["sentence"]}"',
                    ha="center", va="center", fontsize=7.5,
                    fontweight="bold", transform=ax_lbl.transAxes)

        for col_idx in range(N_COLS):
            ax = fig.add_subplot(gs[gs_row + 1, col_idx])
            axes_data[(row_idx, col_idx)] = ax

    # Draw panels
    col_key_to_idx = {k: i for i, k in enumerate(COL_KEYS)}
    for row_idx, col_key, rec, ss, typed, cf, n_matches in episode_results:
        col_idx = col_key_to_idx[col_key]
        ax = axes_data[(row_idx, col_idx)]
        chord_label = f"{cf} chord{'s' if cf != 1 else ''} fired  ({n_matches} matches)"
        draw_panel(ax, rec, ss, chord_label=chord_label)

    # Legend
    legend_elems = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#4a90d9",
               markersize=7, label="Left thumb  (light→dark = early→late)"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#d94a4a",
               markersize=7, label="Right thumb"),
        Line2D([0],[0], marker="D", color="w", markerfacecolor="#cc8800",
               markersize=7, label="Gaze"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="none",
               markeredgecolor="gold", markeredgewidth=1.8,
               markersize=11, label="Chord tap"),
        Line2D([0],[0], marker="x", color="#888888",
               markersize=7, label="Backspace", lw=0),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=3,
               frameon=False, fontsize=7, bbox_to_anchor=(0.5, 0.0))

    fig.savefig(OUT_PATH, bbox_inches="tight", dpi=200)
    png_path = OUT_PATH.replace(".pdf", ".png")
    fig.savefig(png_path, bbox_inches="tight", dpi=200)
    print(f"\nSaved → {OUT_PATH}")
    print(f"Saved → {png_path}")
