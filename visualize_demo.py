"""visualize_demo.py — 3×4 tap and gaze trajectory figure for OmniTypist.

Columns : Single-finger | Two-thumb | Chording
Rows    : 4 sentences, sentence label spanning all columns above each row.

Run with:
    conda activate deeptyping
    cd crtypist/
    python visualize_demo.py
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
from stable_baselines3 import PPO

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

SENTENCES = [
    "she walks by the river every morning",
    "the team worked hard to build the bridge",
    "he bought some books at the local shop",
    "the children played games in the park",
]

OUT_PATH       = "figures/plots/fig_demo.pdf"
KBD_IMG_FOLDER = "kbd1k/keyboard_dataset/"
KBD_LABEL_FILE = "kbd1k/keyboard_label.csv"
FINGER_PATH    = "outputs/finger_agent.pt"
VISION_PATH    = "outputs/vision_agent.pt"
CKPT_SINGLE    = "outputs/supervisor_agent.pt"
CKPT_TWO_THUMB = "outputs/supervisor_agent_two_thumb.pt"
CKPT_CHORD     = "outputs/supervisor_agent_two_thumb_chord.pt"

CROP_Y_TOP = 25   # just trim the status bar; shows text field + keyboard

MAX_RETRIES  = 15
MIN_ACCURACY = 0.90   # difflib similarity threshold to accept an episode

# ---------------------------------------------------------------------------
# Recording wrapper
# ---------------------------------------------------------------------------

class RecordingEnv:
    def __init__(self, env, single_finger=False):
        self.env = env
        self.single_finger = single_finger
        self.taps = []
        self.gaze_fixations = []
        self._t = 0

    def reset(self, **kwargs):
        self._t = 0
        self.taps = []
        self.gaze_fixations = []
        return self.env.reset(**kwargs)

    def step(self, action):
        prev_typed         = self.env.typed_text
        prev_vis_in_act    = self.env.vision_in_action
        prev_vis_goal      = self.env.vision_goal

        obs, reward, done, info = self.env.step(action)
        self._t += 1
        new_typed = self.env.typed_text

        if (not prev_vis_in_act
                and self.env.vision_in_action
                and self.env.vision_goal != prev_vis_goal):
            self.gaze_fixations.append({
                "t":           self._t,
                "x":           float(self.env.gaze[0]),
                "y":           float(self.env.gaze[1]),
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
            hand, pos = self._active_pos()
            self.taps.append({"t": self._t, "hand": hand,
                               "x": float(pos[0]), "y": float(pos[1]),
                               "is_chord": False, "is_backspace": False})
        elif delta < 0:
            hand, pos = self._active_pos()
            self.taps.append({"t": self._t, "hand": hand,
                               "x": float(pos[0]), "y": float(pos[1]),
                               "is_chord": False, "is_backspace": True})

        return obs, reward, done, info

    def _active_pos(self):
        if self.single_finger:
            return "single", self.env.finger
        hand = getattr(self.env, "active_finger", None) or "right"
        pos  = self.env.left_finger if hand == "left" else self.env.right_finger
        return hand, pos

    def __getattr__(self, name):
        return getattr(self.env, name)


# ---------------------------------------------------------------------------
# Episode runner
# ---------------------------------------------------------------------------

def build_env(two_thumb, chords):
    return InternalEnv(
        img_folder=KBD_IMG_FOLDER,
        position_file=KBD_LABEL_FILE,
        text_path="./data/sentences.txt",
        vision_path=VISION_PATH,
        finger_path=FINGER_PATH,
        chars=CHARS, places=PLACES, keys=KEYS,
        two_thumb=two_thumb, chords=chords, chords_active=chords,
    )


def run_episode(env, supervisor, sentence, single_finger=False):
    """Run episode, retrying until >= MIN_ACCURACY. Returns best attempt."""
    best_result = None
    best_ratio  = 0.0

    for attempt in range(MAX_RETRIES):
        rec = RecordingEnv(env, single_finger=single_finger)
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
        chords_fired = getattr(env, "chords_fired", 0)
        ratio        = difflib.SequenceMatcher(None, typed, sentence).ratio()

        if ratio > best_ratio:
            best_ratio  = ratio
            best_result = (rec, screenshot, typed, chords_fired)

        if ratio >= MIN_ACCURACY:
            break
        if attempt < MAX_RETRIES - 1:
            print(f"    retry {attempt+1} (acc={ratio:.2f}  '{typed[:25]}')")

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
                transform=ax.transAxes, fontsize=6,
                ha="right", va="top", color="#222222",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.8, ec="none"))

    taps  = rec.taps
    gazes = rec.gaze_fixations

    def shift_y(y):
        return y - crop_y

    T = max((e["t"] for e in taps + gazes), default=1) + 1

    def tcol(t, cmap, lo=0.3):
        return cmap(lo + (1.0 - lo) * t / T)

    blues  = cm.get_cmap("Blues")
    reds   = cm.get_cmap("Reds")
    purps  = cm.get_cmap("Purples")

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

    # Finger taps
    has_single = any(e["hand"] == "single" for e in taps)

    def draw_hand(hand, cmap):
        seq = [e for e in taps if e["hand"] == hand
               and not e["is_chord"] and not e["is_backspace"]]
        xs  = [e["x"]           for e in seq]
        ys  = [shift_y(e["y"])  for e in seq]
        ts  = [e["t"]           for e in seq]
        if not xs:
            return
        for i in range(len(xs) - 1):
            mid = tcol((seq[i]["t"] + seq[i+1]["t"]) / 2, cmap)
            ax.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]],
                    color=mid, lw=1.0, alpha=0.55, zorder=2)
        ax.scatter(xs, ys, c=ts, cmap=cmap.name, vmin=0, vmax=T,
                   s=50, marker="o", edgecolors="white",
                   linewidths=0.4, zorder=4)

    if has_single:
        draw_hand("single", purps)
    else:
        draw_hand("left",  blues)
        draw_hand("right", reds)

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
                    color="gold", lw=1.4, alpha=0.8, zorder=5, ls="--")
    chord_taps = [e for e in taps if e["is_chord"]]
    if chord_taps:
        ax.scatter([e["x"]          for e in chord_taps],
                   [shift_y(e["y"]) for e in chord_taps],
                   s=200, facecolors="none", edgecolors="gold",
                   linewidths=1.6, zorder=6)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    os.makedirs("figures/plots", exist_ok=True)

    print("Loading supervisors...")
    sup_single    = PPO.load(CKPT_SINGLE,    device="cpu")
    sup_two_thumb = PPO.load(CKPT_TWO_THUMB, device="cpu")
    sup_chord     = PPO.load(CKPT_CHORD,     device="cpu")

    print("Building envs...")
    env_sf = build_env(two_thumb=False, chords=False)
    env_tt = build_env(two_thumb=True,  chords=False)
    env_ch = build_env(two_thumb=True,  chords=True)

    # Run all 12 episodes
    results = []   # (sentence, mode, rec, screenshot, typed, chords_fired)
    for sentence in SENTENCES:
        print(f"\nSentence: \"{sentence}\"")
        for mode, env, sup, sf in [
            ("single",    env_sf, sup_single,    True),
            ("two-thumb", env_tt, sup_two_thumb, False),
            ("chording",  env_ch, sup_chord,     False),
        ]:
            rec, ss, typed, cf = run_episode(env, sup, sentence, single_finger=sf)
            print(f"  {mode:<10}: '{typed}'  taps={len(rec.taps)}"
                  + (f"  chords_fired={cf}" if mode == "chording" else ""))
            results.append((sentence, mode, rec, ss, typed, cf))

    # ---- Figure layout ----
    N_ROWS   = len(SENTENCES)
    N_COLS   = 3
    COL_LABELS = ["Single-finger", "Two-thumb", "Chording"]

    crop_h    = 455 - CROP_Y_TOP
    aspect    = crop_h / 256.0
    panel_w   = 1.9
    panel_h   = panel_w * aspect
    lbl_h_in  = 0.38          # height of each sentence-label row in inches
    hdr_h_in  = 0.30          # height of the column-header row
    fig_w     = N_COLS * panel_w + 0.1
    fig_h     = N_ROWS * (panel_h + lbl_h_in) + hdr_h_in + 0.45

    fig = plt.figure(figsize=(fig_w, fig_h))

    # GridSpec: one header row, then alternating label/data rows
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

    axes_data = {}   # (row, col) -> ax

    for row_idx, sentence in enumerate(SENTENCES):
        gs_row = 1 + row_idx * 2   # offset by 1 for header row

        # Sentence label row — spans all columns
        ax_lbl = fig.add_subplot(gs[gs_row, :])
        ax_lbl.set_axis_off()
        ax_lbl.text(0.5, 0.5, f'"{sentence}"',
                    ha="center", va="center", fontsize=7.5,
                    fontweight="bold", transform=ax_lbl.transAxes)

        # Data panels
        for col_idx in range(N_COLS):
            ax = fig.add_subplot(gs[gs_row + 1, col_idx])
            axes_data[(row_idx, col_idx)] = ax

    # Draw panels
    for i, (sentence, mode, rec, ss, typed, cf) in enumerate(results):
        row_idx = i // N_COLS
        col_idx = i  % N_COLS
        ax = axes_data[(row_idx, col_idx)]

        chord_label = (f"{cf} chord{'s' if cf != 1 else ''} fired"
                       if mode == "chording" else None)
        draw_panel(ax, rec, ss, chord_label=chord_label)

    # Legend
    legend_elems = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#4a90d9",
               markersize=7, label="Left thumb  (light→dark = early→late)"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#d94a4a",
               markersize=7, label="Right thumb"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#8a60c8",
               markersize=7, label="Single finger"),
        Line2D([0],[0], marker="D", color="w", markerfacecolor="#cc8800",
               markersize=7, label="Gaze"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="none",
               markeredgecolor="gold", markeredgewidth=1.6,
               markersize=10, label="Chord tap"),
        Line2D([0],[0], marker="x", color="#888888",
               markersize=7, label="Backspace", lw=0),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=4,
               frameon=False, fontsize=6.5, bbox_to_anchor=(0.5, 0.0))

    fig.savefig(OUT_PATH, bbox_inches="tight", dpi=200)
    png_path = OUT_PATH.replace(".pdf", ".png")
    fig.savefig(png_path, bbox_inches="tight", dpi=200)
    print(f"\nSaved → {OUT_PATH}")
    print(f"Saved → {png_path}")
