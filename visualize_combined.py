"""visualize_combined.py — 3 rows × 5 cols combined figure.

Columns : Single-finger | Two-thumb | Chord (low) | Chord (med) | Chord (high)
Rows    : 3 sentences, sentence label spanning all columns above each row.

Only episodes with >= MIN_ACCURACY character similarity are shown; up to
MAX_RETRIES attempts per cell, falling back to the best attempt if the
threshold is never met.  Final screenshot (showing typed text) is used.

Run with:
    conda activate deeptyping
    cd crtypist/
    python visualize_combined.py
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

OUT_PATH       = "figures/plots/fig_combined.pdf"
KBD_IMG_FOLDER = "kbd1k/keyboard_dataset/"
KBD_LABEL_FILE = "kbd1k/keyboard_label.csv"
FINGER_PATH    = "outputs/finger_agent.pt"
VISION_PATH    = "outputs/vision_agent.pt"
CKPT_SINGLE    = "outputs/supervisor_agent.pt"
CKPT_TWO_THUMB = "outputs/supervisor_agent_two_thumb.pt"
CKPT_CHORD     = "outputs/supervisor_agent_two_thumb_chord.pt"

CROP_Y_TOP   = 0
TEXT_PAD_PX  = 32
MAX_RETRIES  = 15
MIN_ACCURACY = 0.90

# ---------------------------------------------------------------------------
# Sentences + per-row chord vocab conditions (same as visualize_vocab.py)
# ---------------------------------------------------------------------------

ROWS = [
    {
        "sentence": "we have the best for all and with care",
        "low": {
            "n_matches": 1,
            "vocab": {("h", "t"): "the", ("b", "u"): "but", ("c", "o"): "com",
                      ("i", "t"): "tion", ("e", "n"): "ent"},
            "diff":  {("h", "t"): 0.1, ("b", "u"): 0.4, ("c", "o"): 0.6,
                      ("i", "t"): 0.6, ("e", "n"): 0.7},
        },
        "med": {
            "n_matches": 3,
            "vocab": {("h", "t"): "the", ("a", "n"): "and", ("f", "o"): "for",
                      ("c", "o"): "com", ("i", "t"): "tion", ("e", "n"): "ent"},
            "diff":  {("h", "t"): 0.1, ("a", "n"): 0.1, ("f", "o"): 0.3,
                      ("c", "o"): 0.6, ("i", "t"): 0.6, ("e", "n"): 0.7},
        },
        "high": {
            "n_matches": 6,
            "vocab": {("h", "t"): "the", ("a", "n"): "and", ("f", "o"): "for",
                      ("i", "w"): "with", ("a", "l"): "all", ("h", "v"): "have"},
            "diff":  {("h", "t"): 0.1, ("a", "n"): 0.1, ("f", "o"): 0.3,
                      ("i", "w"): 0.3, ("a", "l"): 0.4, ("h", "v"): 0.8},
        },
    },
    {
        "sentence": "but the rain and wind with all their force for change",
        "low": {
            "n_matches": 1,
            "vocab": {("b", "u"): "but", ("c", "o"): "com", ("i", "t"): "tion",
                      ("e", "n"): "ent", ("h", "v"): "have"},
            "diff":  {("b", "u"): 0.4, ("c", "o"): 0.6, ("i", "t"): 0.6,
                      ("e", "n"): 0.7, ("h", "v"): 0.8},
        },
        "med": {
            "n_matches": 3,
            "vocab": {("b", "u"): "but", ("h", "t"): "the", ("a", "n"): "and",
                      ("c", "o"): "com", ("i", "t"): "tion", ("e", "n"): "ent"},
            "diff":  {("b", "u"): 0.4, ("h", "t"): 0.1, ("a", "n"): 0.1,
                      ("c", "o"): 0.6, ("i", "t"): 0.6, ("e", "n"): 0.7},
        },
        "high": {
            "n_matches": 6,
            "vocab": {("b", "u"): "but", ("h", "t"): "the", ("a", "n"): "and",
                      ("i", "w"): "with", ("a", "l"): "all", ("f", "o"): "for"},
            "diff":  {("b", "u"): 0.4, ("h", "t"): 0.1, ("a", "n"): 0.1,
                      ("i", "w"): 0.3, ("a", "l"): 0.4, ("f", "o"): 0.3},
        },
    },
    {
        "sentence": "hello how are you today",
        "low": {
            "n_matches": 1,
            "vocab": {("h", "l"): "hello", ("c", "o"): "com", ("i", "t"): "tion",
                      ("e", "n"): "ent", ("h", "v"): "have"},
            "diff":  {("h", "l"): 0.1, ("c", "o"): 0.6, ("i", "t"): 0.6,
                      ("e", "n"): 0.7, ("h", "v"): 0.8},
        },
        "med": {
            "n_matches": 3,
            "vocab": {("h", "l"): "hello", ("w", "h"): "how", ("a", "r"): "are",
                      ("c", "o"): "com", ("i", "t"): "tion", ("e", "n"): "ent"},
            "diff":  {("h", "l"): 0.1, ("w", "h"): 0.1, ("a", "r"): 0.3,
                      ("c", "o"): 0.6, ("i", "t"): 0.6, ("e", "n"): 0.7},
        },
        "high": {
            "n_matches": 5,
            "vocab": {("h", "l"): "hello", ("w", "h"): "how", ("a", "r"): "are",
                      ("y", "o"): "you", ("t", "o"): "today"},
            "diff":  {("h", "l"): 0.1, ("w", "h"): 0.1, ("a", "r"): 0.3,
                      ("y", "o"): 0.4, ("t", "o"): 0.3},
        },
    },
]

# (display label, mode, chord col_key or None)
COLUMNS = [
    ("Single-finger", "single",    None),
    ("Two-thumb",     "two-thumb", None),
    ("Chord: Low",    "chording",  "low"),
    ("Chord: Med",    "chording",  "med"),
    ("Chord: High",   "chording",  "high"),
]

# ---------------------------------------------------------------------------
# Vocab patching
# ---------------------------------------------------------------------------

def patch_vocab(new_vocab, new_diff):
    orig_v = dict(CHORD_VOCAB)
    orig_d = dict(CHORD_DIFFICULTY)
    CHORD_VOCAB.clear();      CHORD_VOCAB.update(new_vocab)
    CHORD_DIFFICULTY.clear(); CHORD_DIFFICULTY.update(new_diff)
    return orig_v, orig_d


def restore_vocab(orig_v, orig_d):
    CHORD_VOCAB.clear();      CHORD_VOCAB.update(orig_v)
    CHORD_DIFFICULTY.clear(); CHORD_DIFFICULTY.update(orig_d)


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

def run_episode(env, supervisor, sentence, single_finger=False, vocab_cfg=None):
    """Run episode, retrying until >= MIN_ACCURACY. Returns best attempt."""
    if vocab_cfg is not None:
        orig_v, orig_d = patch_vocab(vocab_cfg["vocab"], vocab_cfg["diff"])

    best_result = None
    best_ratio  = 0.0

    try:
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
            screenshot   = env.screenshot.copy()   # after episode — shows typed text
            chords_fired = getattr(env, "chords_fired", 0)
            ratio        = difflib.SequenceMatcher(None, typed, sentence).ratio()

            if ratio > best_ratio:
                best_ratio  = ratio
                best_result = (rec, screenshot, typed, chords_fired)

            if ratio >= MIN_ACCURACY:
                break
            if attempt < MAX_RETRIES - 1:
                print(f"    retry {attempt+1} (acc={ratio:.2f}  '{typed[:25]}')")
    finally:
        if vocab_cfg is not None:
            restore_vocab(orig_v, orig_d)

    return best_result


# ---------------------------------------------------------------------------
# Drawing
# ---------------------------------------------------------------------------

def _draw_produced_text(ax, text, highlight_words=None,
                        text_color="#222222", highlight_color="#8fc3e8",
                        highlight_alpha=0.5, y_top=0.0, line_gap_px=10):
    if not text:
        return

    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    if renderer is None:
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()

    words = text.split(" ")
    x = 0.0
    line = 0
    fontsize = 6
    alpha = 0.75
    space = " "

    def norm(w):
        return w.lower().strip(".,;:!?\"'()[]{}")

    for i, word in enumerate(words):
        token = word if i == 0 else space + word
        is_highlight = highlight_words and (norm(word) in highlight_words)
        bbox = None
        if is_highlight:
            bbox = dict(boxstyle="round,pad=0.12",
                        fc=highlight_color, ec="none", alpha=highlight_alpha)
        t = ax.text(x, y_top + line * line_gap_px, token,
                    transform=ax.transData, ha="left", va="top",
                    fontsize=fontsize, color=text_color,
                    alpha=alpha, bbox=bbox, clip_on=False, zorder=10)
        bb = t.get_window_extent(renderer=renderer)
        inv = ax.transData.inverted()
        dx = inv.transform((bb.width, 0))[0] - inv.transform((0, 0))[0]
        x_next = x + dx
        if x_next >= ax.get_xlim()[1] and line == 0:
            t.remove()
            line = 1
            x = 0.0
            t = ax.text(x, y_top + line * line_gap_px, token,
                        transform=ax.transData, ha="left", va="top",
                        fontsize=fontsize, color=text_color,
                        alpha=alpha, bbox=bbox, clip_on=False, zorder=10)
            bb = t.get_window_extent(renderer=renderer)
            dx = inv.transform((bb.width, 0))[0] - inv.transform((0, 0))[0]
            x = dx
        else:
            x = x_next
        if x >= ax.get_xlim()[1] or line > 1:
            break


def draw_panel(ax, rec, screenshot, label=None, crop_y=CROP_Y_TOP,
               produced_text=None, highlight_words=None):
    img = screenshot.crop((0, crop_y, screenshot.width, screenshot.height))
    W, H = img.width, img.height
    pad = TEXT_PAD_PX

    ax.imshow(np.array(img.convert("RGB")), origin="upper", aspect="auto",
              extent=[0, W, H + pad, pad])
    ax.set_xlim(0, W)
    ax.set_ylim(H + pad, 0)
    ax.set_xticks([])
    ax.set_yticks([])

    text_y_top = pad * 0.25
    text_gap = pad * 0.5
    _draw_produced_text(ax, produced_text, highlight_words=highlight_words,
                        y_top=text_y_top, line_gap_px=text_gap)

    if label:
        ax.text(0.97, 0.93, label,
                transform=ax.transAxes, fontsize=5,
                ha="right", va="top", color="#222222",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.85, ec="none"))

    taps  = rec.taps
    gazes = rec.gaze_fixations

    def shift_y(y):
        return y - crop_y + pad

    T = max((e["t"] for e in taps + gazes), default=1) + 1

    def tcol(t, cmap, lo=0.3):
        return cmap(lo + (1.0 - lo) * t / T)

    blues = cm.get_cmap("Blues")
    reds  = cm.get_cmap("Reds")
    purps = cm.get_cmap("Purples")

    # Gaze fixations (unified — key + proofread treated as one)
    gaze_sorted = sorted(gazes, key=lambda g: g["t"])
    xs_g = [g["x"] for g in gaze_sorted]
    ys_g = [shift_y(g["y"]) for g in gaze_sorted]
    ts_g = [g["t"] for g in gaze_sorted]
    if xs_g:
        if len(xs_g) > 1:
            for i in range(len(xs_g) - 1):
                ax.plot([xs_g[i], xs_g[i+1]], [ys_g[i], ys_g[i+1]],
                        color="#cc7700", lw=0.6, alpha=0.4, zorder=2)
        ax.scatter(xs_g, ys_g, c=ts_g, cmap="YlOrBr", vmin=0, vmax=T,
                   s=16, marker="D", zorder=3, linewidths=0)

    # Finger taps
    has_single = any(e["hand"] == "single" for e in taps)

    def draw_hand(hand, cmap):
        seq = [e for e in taps if e["hand"] == hand
               and not e["is_chord"] and not e["is_backspace"]]
        xs = [e["x"]          for e in seq]
        ys = [shift_y(e["y"]) for e in seq]
        ts = [e["t"]          for e in seq]
        if not xs:
            return
        for i in range(len(xs) - 1):
            mid = tcol((seq[i]["t"] + seq[i+1]["t"]) / 2, cmap)
            ax.plot([xs[i], xs[i+1]], [ys[i], ys[i+1]],
                    color=mid, lw=0.8, alpha=0.55, zorder=2)
        ax.scatter(xs, ys, c=ts, cmap=cmap.name, vmin=0, vmax=T,
                   s=35, marker="o", edgecolors="white",
                   linewidths=0.35, zorder=4)

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
                   c="#888888", s=20, marker="x",
                   linewidths=0.9, zorder=4, alpha=0.65)

    # Chord rings
    chord_times = sorted(set(e["t"] for e in taps if e["is_chord"]))
    for ct in chord_times:
        pair = [e for e in taps if e["is_chord"] and e["t"] == ct]
        if len(pair) == 2:
            ax.plot([pair[0]["x"],          pair[1]["x"]],
                    [shift_y(pair[0]["y"]), shift_y(pair[1]["y"])],
                    color="gold", lw=1.2, alpha=0.85, zorder=5, ls="--")
    chord_taps = [e for e in taps if e["is_chord"]]
    if chord_taps:
        ax.scatter([e["x"]          for e in chord_taps],
                   [shift_y(e["y"]) for e in chord_taps],
                   s=150, facecolors="none", edgecolors="gold",
                   linewidths=1.4, zorder=6)


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
    env_sf = InternalEnv(
        img_folder=KBD_IMG_FOLDER, position_file=KBD_LABEL_FILE,
        text_path="./data/sentences.txt", vision_path=VISION_PATH,
        finger_path=FINGER_PATH, chars=CHARS, places=PLACES, keys=KEYS,
        two_thumb=False, chords=False,
    )
    env_tt = InternalEnv(
        img_folder=KBD_IMG_FOLDER, position_file=KBD_LABEL_FILE,
        text_path="./data/sentences.txt", vision_path=VISION_PATH,
        finger_path=FINGER_PATH, chars=CHARS, places=PLACES, keys=KEYS,
        two_thumb=True, chords=False,
    )
    env_ch = InternalEnv(
        img_folder=KBD_IMG_FOLDER, position_file=KBD_LABEL_FILE,
        text_path="./data/sentences.txt", vision_path=VISION_PATH,
        finger_path=FINGER_PATH, chars=CHARS, places=PLACES, keys=KEYS,
        two_thumb=True, chords=True, chords_active=True,
    )

    # Run all 15 episodes
    results = []  # (row_idx, col_idx, rec, ss, typed, chords_fired)
    for row_idx, row in enumerate(ROWS):
        sentence = row["sentence"]
        print(f"\nRow {row_idx+1}: \"{sentence}\"")
        for col_idx, (col_label, mode, col_key) in enumerate(COLUMNS):
            if mode == "single":
                env, sup, sf, vocab_cfg = env_sf, sup_single, True, None
            elif mode == "two-thumb":
                env, sup, sf, vocab_cfg = env_tt, sup_two_thumb, False, None
            else:
                env, sup, sf, vocab_cfg = env_ch, sup_chord, False, row[col_key]

            rec, ss, typed, cf = run_episode(env, sup, sentence,
                                             single_finger=sf, vocab_cfg=vocab_cfg)
            ratio = difflib.SequenceMatcher(None, typed, sentence).ratio()
            chord_info = f"  chords_fired={cf}" if mode == "chording" else ""
            print(f"  {col_label:<14}: acc={ratio:.2f}  '{typed[:30]}'"
                  f"  taps={len(rec.taps)}{chord_info}")
            results.append((row_idx, col_idx, rec, ss, typed, cf))

    # ---- Figure layout ----
    N_ROWS  = len(ROWS)
    N_COLS  = len(COLUMNS)

    crop_h   = 455 - CROP_Y_TOP
    aspect   = crop_h / 256.0
    panel_w  = 1.45
    panel_h  = panel_w * aspect
    lbl_h_in = 0.35
    hdr_h_in = 0.26
    fig_w    = N_COLS * panel_w + 0.1
    fig_h    = N_ROWS * (panel_h + lbl_h_in) + hdr_h_in + 0.42

    fig = plt.figure(figsize=(fig_w, fig_h))

    height_ratios = [hdr_h_in / panel_h]
    for _ in range(N_ROWS):
        height_ratios += [lbl_h_in / panel_h, 1.0]

    gs = fig.add_gridspec(
        1 + N_ROWS * 2, N_COLS,
        height_ratios=height_ratios,
        hspace=0.0, wspace=0.03,
        left=0.01, right=0.99,
        top=0.99, bottom=0.07,
    )

    # Column headers
    for col_idx, (col_label, _, _) in enumerate(COLUMNS):
        ax_hdr = fig.add_subplot(gs[0, col_idx])
        ax_hdr.set_axis_off()
        ax_hdr.text(0.5, 0.3, col_label,
                    ha="center", va="center", fontsize=7,
                    fontweight="bold", transform=ax_hdr.transAxes)

    axes_data = {}
    for row_idx, row in enumerate(ROWS):
        gs_row = 1 + row_idx * 2
        ax_lbl = fig.add_subplot(gs[gs_row, :])
        ax_lbl.set_axis_off()
        ax_lbl.text(0.5, 0.5, f'"{row["sentence"]}"',
                    ha="center", va="center", fontsize=7,
                    fontweight="bold", transform=ax_lbl.transAxes)
        for col_idx in range(N_COLS):
            axes_data[(row_idx, col_idx)] = fig.add_subplot(gs[gs_row + 1, col_idx])

    # Draw panels
    for row_idx, col_idx, rec, ss, typed, cf in results:
        ax   = axes_data[(row_idx, col_idx)]
        mode = COLUMNS[col_idx][1]
        label = (f"{cf} chord{'s' if cf != 1 else ''}" if mode == "chording" else None)
        highlight_words = None
        if mode == "chording":
            col_key = COLUMNS[col_idx][2]
            vocab = ROWS[row_idx][col_key]["vocab"]
            highlight_words = {v.lower() for v in vocab.values()}
        draw_panel(ax, rec, ss, label=label,
                   produced_text=typed, highlight_words=highlight_words)

    # Legend
    legend_elems = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#4a90d9",
               markersize=6, label="Left thumb  (light→dark = early→late)"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#d94a4a",
               markersize=6, label="Right thumb"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#8a60c8",
               markersize=6, label="Single finger"),
        Line2D([0],[0], marker="D", color="w", markerfacecolor="#cc8800",
               markersize=6, label="Gaze"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="none",
               markeredgecolor="gold", markeredgewidth=1.4,
               markersize=9, label="Chord tap"),
        Line2D([0],[0], marker="x", color="#888888",
               markersize=6, label="Backspace", lw=0),
    ]
    fig.legend(handles=legend_elems, loc="lower center", ncol=6,
               frameon=False, fontsize=6, bbox_to_anchor=(0.5, 0.0))
    fig.text(0.5, 0.035,
             "Light-blue highlight indicates produced word matches the active chord vocabulary.",
             ha="center", va="bottom", fontsize=6, color="#333333")

    fig.savefig(OUT_PATH, bbox_inches="tight", dpi=200)
    png_path = OUT_PATH.replace(".pdf", ".png")
    fig.savefig(png_path, bbox_inches="tight", dpi=200)
    print(f"\nSaved → {OUT_PATH}")
    print(f"Saved → {png_path}")
