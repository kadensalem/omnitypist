"""Transfer weights from supervisor_agent_two_thumb.pt to the chord env.

The two-thumb and chord policy networks share the same hidden layer shapes
but differ in two places:

  Input layer (policy_net.0 and value_net.0):
    Two-thumb: (64, 11)  — 11 obs features
    Chord:     (64, 16)  — 16 obs features (5 new chord dims interleaved)

    gym.spaces.Dict sorts keys alphabetically (gym 0.21), so the feature
    columns are NOT simply appended at the end. The actual column layout is:

      Two-thumb (11 cols):
        P(0-2), certainty(3), correctness(4),
        left_finger_in_action(5-6), right_finger_in_action(7-8),
        vision_in_action(9-10)

      Chord (16 cols):
        P(0-2), backspace_ready(3-4)[NEW], certainty(5),
        chord_available(6-7)[NEW], chord_recall_strength(8)[NEW],
        correctness(9), left_finger_in_action(10-11),
        right_finger_in_action(12-13), vision_in_action(14-15)

    Transfer: each shared key is mapped individually to its correct destination
    column. New chord-only columns are zero-initialised.

    This script computes the column mapping dynamically from the actual
    observation space key ordering so it stays correct if keys are added later.

  Action output (action_net):
    Two-thumb: (14, 64)  — MultiDiscrete([2, 2, 10])
    Chord:     (15, 64)  — MultiDiscrete([2, 3, 10])
    SB3 lays out rows contiguously per action dimension:
      rows 0-1:   action[0] (vision goal, 2 choices)  — copy
      rows 2-3:   action[1] type + backspace           — copy
      row  4:     action[1] chord (NEW)                — zero weight, negative bias
      rows 5-14:  action[2] speeds (10 choices)        — copy from rows 4-13

  All hidden layers and value_net output: identical shape — copy directly.

Usage:
    python transfer_weights.py
    # then run phase 2:
    # python main.py --train --supervisor-agent --two-thumb --chords --chords-active
    #     --continue-training --total-timesteps 5000000
"""

import torch
import numpy as np
from gym import spaces
from gym.spaces.utils import flatdim
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from typing_env.internal_env import InternalEnv
from setting import KEYS, PLACES, CHARS

TWO_THUMB_PATH = "outputs/supervisor_agent_two_thumb.pt"
CHORD_PATH     = "outputs/supervisor_agent_two_thumb_chord.pt"

# Initial bias for the chord action output. Negative value makes the model
# strongly prefer type/backspace over chord at the start of phase 2, so it
# doesn't destabilise the already-learned typing policy.
CHORD_ACTION_INIT_BIAS = -0.5


def _key_offsets(obs_space: spaces.Dict) -> dict:
    """Return {key: (start_col, end_col)} for each key in obs_space.

    Discrete(n) occupies n columns because SB3 one-hot encodes Discrete
    observations before passing them through the MLP extractor.
    """
    offsets = {}
    col = 0
    for key, subspace in obs_space.spaces.items():
        if isinstance(subspace, spaces.Discrete):
            dim = subspace.n  # SB3 one-hot encodes → n columns
        else:
            dim = flatdim(subspace)
        offsets[key] = (col, col + dim)
        col += dim
    return offsets


def build_chord_env():
    return InternalEnv(
        img_folder="kbd1k/keyboard_dataset/",
        position_file="kbd1k/keyboard_label.csv",
        text_path="./data/sentences.txt",
        vision_path="outputs/vision_agent.pt",
        finger_path="outputs/finger_agent.pt",
        chars=CHARS, places=PLACES, keys=KEYS,
        two_thumb=True, chords=True, chords_active=False,
    )


def transfer():
    print("Loading two-thumb checkpoint...")
    two_thumb_env = InternalEnv(
        img_folder="kbd1k/keyboard_dataset/",
        position_file="kbd1k/keyboard_label.csv",
        text_path="./data/sentences.txt",
        vision_path="outputs/vision_agent.pt",
        finger_path="outputs/finger_agent.pt",
        chars=CHARS, places=PLACES, keys=KEYS,
        two_thumb=True, chords=False,
    )
    src = PPO.load(TWO_THUMB_PATH, env=two_thumb_env)
    src_params = dict(src.policy.named_parameters())

    print("Building fresh chord policy...")
    chord_env = Monitor(build_chord_env(), "logs/")
    dst = PPO("MultiInputPolicy", chord_env)

    # Compute the per-key column ranges for each obs space.
    src_offsets = _key_offsets(two_thumb_env.observation_space)
    dst_offsets = _key_offsets(chord_env.env.observation_space)

    print("\nTwo-thumb obs column layout:")
    for k, (s, e) in src_offsets.items():
        print(f"  {k}: cols {s}-{e-1}")
    print("\nChord obs column layout:")
    for k, (s, e) in dst_offsets.items():
        tag = " [NEW]" if k not in src_offsets else ""
        print(f"  {k}: cols {s}-{e-1}{tag}")

    # Shared keys that exist in both obs spaces.
    shared_keys = [k for k in src_offsets if k in dst_offsets]

    print("\nTransferring weights:")

    with torch.no_grad():
        for name, dst_param in dst.policy.named_parameters():
            if name not in src_params:
                print(f"  SKIP  {name} — not in source")
                continue

            src_param = src_params[name]
            s = src_param.shape
            d = dst_param.shape

            if s == d:
                # Identical shape — copy directly (hidden layers, value output).
                dst_param.copy_(src_param)
                print(f"  COPY  {name}: {tuple(s)}")

            elif name in ("mlp_extractor.policy_net.0.weight",
                          "mlp_extractor.value_net.0.weight"):
                # Input layer weight: (64, src_dim) → (64, dst_dim)
                # Zero-init all columns, then copy each shared key's columns
                # to their correct destination positions.
                assert s[0] == d[0], f"Hidden dim mismatch: {s[0]} vs {d[0]}"
                dst_param.fill_(0.0)
                for key in shared_keys:
                    ss, se = src_offsets[key]
                    ds, de = dst_offsets[key]
                    assert (se - ss) == (de - ds), \
                        f"Column width mismatch for key '{key}': {se-ss} vs {de-ds}"
                    dst_param[:, ds:de] = src_param[:, ss:se]
                new_keys = [k for k in dst_offsets if k not in src_offsets]
                print(f"  MAP   {name}: {tuple(s)} → {tuple(d)}")
                print(f"        shared={shared_keys}")
                print(f"        zeroed={new_keys}")

            elif name in ("mlp_extractor.policy_net.0.bias",
                          "mlp_extractor.value_net.0.bias"):
                # Input layer bias: same hidden dim — copy directly.
                assert s == d, f"Bias shape mismatch: {s} vs {d}"
                dst_param.copy_(src_param)
                print(f"  COPY  {name}: {tuple(s)}")

            elif name == "action_net.weight":
                # Action output weight: (14, 64) → (15, 64)
                # Insert a zero row at index 4 for the chord action.
                assert d == (15, 64) and s == (14, 64), \
                    f"Unexpected action_net.weight shapes: src={s} dst={d}"
                dst_param[0:4, :] = src_param[0:4, :]   # action[0] + type/backspace
                dst_param[4,   :] = 0.0                  # chord action (new)
                dst_param[5:,  :] = src_param[4:,  :]   # speeds
                print(f"  INS   {name}: {tuple(s)} → {tuple(d)} "
                      f"(row 4 zeroed for chord)")

            elif name == "action_net.bias":
                # Action output bias: (14,) → (15,)
                # Insert negative bias at index 4 to discourage chord initially.
                assert d == (15,) and s == (14,), \
                    f"Unexpected action_net.bias shapes: src={s} dst={d}"
                dst_param[0:4] = src_param[0:4]
                dst_param[4]   = CHORD_ACTION_INIT_BIAS
                dst_param[5:]  = src_param[4:]
                print(f"  INS   {name}: {tuple(s)} → {tuple(d)} "
                      f"(index 4 = {CHORD_ACTION_INIT_BIAS} for chord)")

            else:
                print(f"  WARN  {name}: shape mismatch src={s} dst={d} — skipped")

    dst.save(CHORD_PATH)
    print(f"\nSaved transferred checkpoint to {CHORD_PATH}")


def verify():
    print("\nVerifying transferred checkpoint loads correctly...")
    chord_env = build_chord_env()
    agent = PPO.load(CHORD_PATH, env=chord_env)

    obs = chord_env.reset(target_text="the cat and dog", reset_kbd=True)
    for _ in range(5):
        action, _ = agent.predict(obs, deterministic=True)
        obs, _, done, _ = chord_env.step(action)
        if done:
            break

    # Confirm chord action (action[1]=2) is not being spammed —
    # the negative bias should make the model prefer type/backspace.
    action_counts = {0: 0, 1: 0, 2: 0}
    obs = chord_env.reset(target_text="the cat and dog", reset_kbd=True)
    for _ in range(100):
        action, _ = agent.predict(obs, deterministic=True)
        action_counts[int(action[1])] += 1
        obs, _, done, _ = chord_env.step(action)
        if done:
            break

    print(f"  Action[1] distribution over 100 steps: "
          f"type={action_counts[0]} backspace={action_counts[1]} chord={action_counts[2]}")
    assert action_counts[2] < action_counts[0] + action_counts[1], \
        "Chord action should not dominate — check CHORD_ACTION_INIT_BIAS"
    print("  Verification OK")


if __name__ == "__main__":
    transfer()
    verify()
