"""evaluate.py — Systematic evaluation for OmniTypist experiments.

Run from the crtypist/ directory with the deeptyping conda env active:

    conda activate deeptyping
    cd /path/to/crtypist

Experiments
-----------
  mode            Typing mode comparison: single-finger vs two-thumb vs chording
  vocab-size      Chord vocabulary size ablation: 2 / 5 / 10 chords
  difficulty      Chord difficulty assignment: uniform-easy / medium / graduated / hard
  individual-diff Individual differences: vary memory parameter across three levels

Examples
--------
  python evaluate.py --experiment mode            --n-episodes 30 --out results/mode.csv
  python evaluate.py --experiment vocab-size      --n-episodes 30 --out results/vocab_size.csv
  python evaluate.py --experiment difficulty      --n-episodes 30 --out results/difficulty.csv
  python evaluate.py --experiment individual-diff --n-episodes 30 --out results/individual_diff.csv
"""

import argparse
import csv
import os
import random

import numpy as np
from tqdm import tqdm

# Import the live vocab dicts — these are the same objects referenced by
# chord_memory.py and internal_env.py. Modifying them in-place (via
# patch_vocab/restore_vocab below) affects all modules simultaneously, which
# is how the vocabulary ablations work without touching source files.
from data.chord_vocab import CHORD_VOCAB, CHORD_DIFFICULTY

# Capture original values before any patching so we can always restore.
_ORIG_VOCAB       = dict(CHORD_VOCAB)
_ORIG_DIFFICULTY  = dict(CHORD_DIFFICULTY)

from data.sentences import Sentences
from metrics import Metrics
from models.supervisor_agent import SupervisorAgent
from parameters import parameters as FITTED_PARAMS
from setting import KEYS, PLACES, CHARS
from typing_env.internal_env import InternalEnv


# ---------------------------------------------------------------------------
# Vocabulary presets  (Experiment 2 — vocab-size)
# ---------------------------------------------------------------------------

# 2 chords: the two highest-frequency, easiest-to-recall chords.
VOCAB_2 = {('h', 't'): "the", ('a', 'n'): "and"}
DIFFICULTY_2 = {('h', 't'): 0.1, ('a', 'n'): 0.1}

# 5 chords: top 5 by English word frequency.
VOCAB_5 = {
    ('h', 't'): "the",
    ('a', 'n'): "and",
    ('f', 'o'): "for",
    ('i', 'w'): "with",
    ('b', 'u'): "but",
}
DIFFICULTY_5 = {
    ('h', 't'): 0.1, ('a', 'n'): 0.1, ('f', 'o'): 0.3,
    ('i', 'w'): 0.3, ('b', 'u'): 0.4,
}

# 10 chords: full vocabulary (snapshot taken before any patching).
VOCAB_10      = dict(_ORIG_VOCAB)
DIFFICULTY_10 = dict(_ORIG_DIFFICULTY)


# ---------------------------------------------------------------------------
# Difficulty presets  (Experiment 3 — difficulty)
# ---------------------------------------------------------------------------

# All chords equally easy to recall (minimal decay).
DIFFICULTY_UNIFORM_EASY   = {k: 0.1 for k in _ORIG_VOCAB}
# All chords at mid-range difficulty.
DIFFICULTY_UNIFORM_MEDIUM = {k: 0.5 for k in _ORIG_VOCAB}
# Current design: harder chords decay faster (see chord_vocab.py).
DIFFICULTY_GRADUATED      = dict(_ORIG_DIFFICULTY)
# All chords hard — recall decays quickly for every chord.
DIFFICULTY_UNIFORM_HARD   = {k: 0.8 for k in _ORIG_VOCAB}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def patch_vocab(new_vocab, new_difficulty):
    """Replace CHORD_VOCAB / CHORD_DIFFICULTY contents in-place.

    Returns (orig_vocab, orig_difficulty) for restoration.  Both dicts are
    mutated in-place so chord_memory.py and internal_env.py — which hold
    references to the same dict objects — immediately see the new values.
    """
    orig_vocab = dict(CHORD_VOCAB)
    orig_diff  = dict(CHORD_DIFFICULTY)
    CHORD_VOCAB.clear()
    CHORD_VOCAB.update(new_vocab)
    CHORD_DIFFICULTY.clear()
    CHORD_DIFFICULTY.update(new_difficulty)
    return orig_vocab, orig_diff


def restore_vocab(orig_vocab, orig_diff):
    """Undo a previous patch_vocab call."""
    CHORD_VOCAB.clear()
    CHORD_VOCAB.update(orig_vocab)
    CHORD_DIFFICULTY.clear()
    CHORD_DIFFICULTY.update(orig_diff)


def chord_coverage(target_text, vocab):
    """Greedy fraction of target characters coverable by chord expansions.

    Scans left-to-right; at each position tries every expansion and advances
    past the first match.  Returns 0.0 for non-chord conditions (empty vocab).
    """
    if not target_text or not vocab:
        return 0.0
    expansions = list(vocab.values())
    covered, i = 0, 0
    while i < len(target_text):
        matched = False
        for exp in expansions:
            if target_text[i:].startswith(exp):
                covered += len(exp)
                i += len(exp)
                matched = True
                break
        if not matched:
            i += 1
    return covered / len(target_text)


_VALID_CHARS = set(CHARS)  # characters the keyboard can actually type


def sample_sentences(n, seed=42):
    """Draw n sentences from sentences.txt using a fixed random seed.

    Only sentences whose every character is in CHARS (a-z + space) are
    eligible.  Sentences with apostrophes, numbers, or other punctuation
    would cause keys.index(finger_goal) to raise ValueError at runtime.

    Using a fixed seed ensures all conditions in an experiment run on
    exactly the same set of target phrases.
    """
    db = Sentences(load_path='./data/sentences.txt')
    valid = [
        db[i] for i in range(len(db))
        if db[i] and all(c in _VALID_CHARS for c in db[i])
    ]
    if len(valid) < n:
        print(f"WARNING: only {len(valid)} valid sentences found; "
              f"requested {n}. Reducing to {len(valid)}.")
        n = len(valid)
    rng = random.Random(seed)
    indices = rng.sample(range(len(valid)), n)
    return [valid[i] for i in indices]


def make_env(two_thumb, chords, chords_active=True):
    """Instantiate InternalEnv with standard paths."""
    return InternalEnv(
        img_folder='kbd1k/keyboard_dataset/',
        position_file='kbd1k/keyboard_label.csv',
        text_path='./data/sentences.txt',
        vision_path='outputs/vision_agent.pt',
        finger_path='outputs/finger_agent.pt',
        chars=CHARS, places=PLACES, keys=KEYS,
        two_thumb=two_thumb,
        chords=chords,
        chords_active=chords_active,
    )


MAX_RETRIES = 10  # give up on a sentence after this many failed attempts


def run_episodes(env, agent, sentences, params, max_cer=0.2):
    """Run one episode per sentence; retry on CER >= max_cer (outlier filter).

    The max_cer=0.2 threshold matches the filter used in main.py's evaluate
    block to exclude degenerate episodes.  Sentences that fail MAX_RETRIES
    times in a row are skipped with a warning rather than looping forever.
    Returns one result dict per accepted sentence.
    """
    results = []
    bar = tqdm(total=len(sentences), unit='ep')
    for idx, sentence in enumerate(sentences):
        bar.set_description(f"ep {idx + 1}/{len(sentences)}")
        tqdm.write(f"  sentence: {sentence!r}")
        accepted = False
        for attempt in range(1, MAX_RETRIES + 1):
            obs  = env.reset(target_text=sentence, parameters=list(params))
            done = False
            step = 0
            while not done:
                action, _ = agent.predict(obs, deterministic=True)
                obs, _, done, _ = env.step(action)
                step += 1
                if step % 500 == 0:
                    tqdm.write(f"    ... step {step} typed={env.typed_text!r}")
            m       = Metrics(log=env.log, target_text=env.target_text)
            summary = m.summary()
            if summary['char_error_rate'] < max_cer:
                summary['chord_attempts'] = env.chord_attempts
                summary['chords_fired']   = env.chords_fired
                summary['fire_rate'] = (
                    env.chords_fired / env.chord_attempts
                    if env.chord_attempts > 0 else 0.0
                )
                summary['chord_coverage'] = chord_coverage(
                    sentence, dict(CHORD_VOCAB)
                )
                results.append(summary)
                bar.set_postfix(
                    WPM=f"{summary['WPM']:.1f}",
                    CER=f"{summary['char_error_rate']:.3f}",
                    retries=attempt - 1,
                )
                bar.update(1)
                accepted = True
                break
            else:
                tqdm.write(f"    attempt {attempt} CER={summary['char_error_rate']:.3f} >= {max_cer}, retrying…")
        if not accepted:
            tqdm.write(f"  WARNING: skipped after {MAX_RETRIES} attempts: {sentence!r}")
            bar.update(1)
    bar.close()
    return results


def print_summary(results, condition):
    """Print mean ± SD for each metric."""
    keys = [
        'WPM', 'IKI', 'char_error_rate', 'num_backspaces',
        'gaze_shift', 'gaze_kbd_ratio',
        'chord_attempts', 'chords_fired', 'fire_rate', 'chord_coverage',
    ]
    print(f"\n── {condition}  (n={len(results)}) ──")
    for k in keys:
        vals = [r[k] for r in results]
        print(f"  {k:<28s}  {np.mean(vals):7.3f}  ± {np.std(vals):.3f}")


def save_csv(all_results, path):
    """Write all conditions to a single CSV file, one row per episode."""
    if not all_results or not all_results[0][1]:
        print("No results to save.")
        return
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    fieldnames = ['condition'] + list(all_results[0][1][0].keys())
    with open(path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for condition, results in all_results:
            for r in results:
                writer.writerow({'condition': condition, **r})
    print(f"\nRaw episode data saved → {path}")


# ---------------------------------------------------------------------------
# Experiment 1 — typing mode comparison
# ---------------------------------------------------------------------------

def exp_mode(n_episodes, seed):
    """Single-finger vs two-thumb vs chording on the same sentence set."""
    sentences = sample_sentences(n_episodes, seed)
    all_results = []

    conditions = [
        (
            'single-finger', False, False, False,
            'outputs/supervisor_agent.pt',
        ),
        (
            'two-thumb', True, False, False,
            'outputs/supervisor_agent_two_thumb.pt',
        ),
        (
            'chording', True, True, True,
            'outputs/supervisor_agent_two_thumb_chord.pt',
        ),
    ]

    for name, two_thumb, chords, chords_active, ckpt in conditions:
        print(f"\n[{name}] Loading env and agent…")
        env   = make_env(two_thumb, chords, chords_active)
        agent = SupervisorAgent(env=env, load=ckpt)
        results = run_episodes(env, agent, sentences, FITTED_PARAMS)
        print_summary(results, name)
        all_results.append((name, results))

    return all_results


# ---------------------------------------------------------------------------
# Experiment 2 — chord vocabulary size
# ---------------------------------------------------------------------------

def exp_vocab_size(n_episodes, seed):
    """Chord vocabulary size ablation: 2 / 5 / 10 chords.

    The supervisor checkpoint is unchanged across conditions; only the active
    vocabulary (CHORD_VOCAB / CHORD_DIFFICULTY) is swapped in-place.
    """
    sentences = sample_sentences(n_episodes, seed)
    all_results = []

    conditions = [
        ('vocab-2',  VOCAB_2,  DIFFICULTY_2),
        ('vocab-5',  VOCAB_5,  DIFFICULTY_5),
        ('vocab-10', VOCAB_10, DIFFICULTY_10),
    ]

    for name, vocab, diff in conditions:
        print(f"\n[{name}] Patching vocab ({len(vocab)} chord(s))…")
        orig  = patch_vocab(vocab, diff)
        env   = make_env(True, True, True)
        agent = SupervisorAgent(
            env=env,
            load='outputs/supervisor_agent_two_thumb_chord.pt',
        )
        results = run_episodes(env, agent, sentences, FITTED_PARAMS)
        restore_vocab(*orig)
        print_summary(results, name)
        all_results.append((name, results))

    return all_results


# ---------------------------------------------------------------------------
# Experiment 3 — difficulty assignment
# ---------------------------------------------------------------------------

def exp_difficulty(n_episodes, seed):
    """Chord difficulty assignment: uniform-easy / medium / graduated / hard.

    Vocabulary (which chords exist) is held constant at the full 10-chord
    set; only the per-chord difficulty values (decay rates) are varied.
    """
    sentences = sample_sentences(n_episodes, seed)
    all_results = []

    conditions = [
        ('uniform-easy',   dict(_ORIG_VOCAB), DIFFICULTY_UNIFORM_EASY),
        ('uniform-medium', dict(_ORIG_VOCAB), DIFFICULTY_UNIFORM_MEDIUM),
        ('graduated',      dict(_ORIG_VOCAB), DIFFICULTY_GRADUATED),
        ('uniform-hard',   dict(_ORIG_VOCAB), DIFFICULTY_UNIFORM_HARD),
    ]

    for name, vocab, diff in conditions:
        print(f"\n[{name}] Setting difficulty scheme…")
        orig  = patch_vocab(vocab, diff)
        env   = make_env(True, True, True)
        agent = SupervisorAgent(
            env=env,
            load='outputs/supervisor_agent_two_thumb_chord.pt',
        )
        results = run_episodes(env, agent, sentences, FITTED_PARAMS)
        restore_vocab(*orig)
        print_summary(results, name)
        all_results.append((name, results))

    return all_results


# ---------------------------------------------------------------------------
# Experiment 4 — individual differences
# ---------------------------------------------------------------------------

def exp_individual_diff(n_episodes, seed):
    """Individual differences: vary the memory decay parameter.

    finger_param and vision_param are held at their fitted values.
    The memory parameter is swept across three levels:
      high-memory   (0.1)  — slow decay, large working-memory capacity
      fitted-memory (0.321) — the population-average fitted value
      low-memory    (0.8)  — fast decay, small working-memory capacity
    """
    sentences = sample_sentences(n_episodes, seed)
    all_results = []

    memory_levels = [
        ('high-memory',   0.1),
        ('fitted-memory', FITTED_PARAMS[0]),
        ('low-memory',    0.8),
    ]

    print("\nLoading env and agent (shared across memory conditions)…")
    env   = make_env(True, True, True)
    agent = SupervisorAgent(
        env=env,
        load='outputs/supervisor_agent_two_thumb_chord.pt',
    )

    for name, mem_p in memory_levels:
        params = [mem_p, FITTED_PARAMS[1], FITTED_PARAMS[2]]
        print(f"\n[{name}] memory_p={mem_p:.3f}")
        results = run_episodes(env, agent, sentences, params)
        print_summary(results, name)
        all_results.append((name, results))

    return all_results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="OmniTypist evaluation script",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--experiment',
        choices=['mode', 'vocab-size', 'difficulty', 'individual-diff'],
        required=True,
        help='Which experiment to run.',
    )
    parser.add_argument(
        '--n-episodes', type=int, default=30,
        help='Number of episodes per condition (default: 30).',
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help='Random seed for sentence sampling (default: 42).',
    )
    parser.add_argument(
        '--out', type=str, default=None,
        help='Output CSV path. Defaults to results/<experiment>.csv.',
    )
    args = parser.parse_args()

    out_path = args.out or os.path.join('results', f'{args.experiment}.csv')

    dispatch = {
        'mode':            exp_mode,
        'vocab-size':      exp_vocab_size,
        'difficulty':      exp_difficulty,
        'individual-diff': exp_individual_diff,
    }

    print(f"Experiment : {args.experiment}")
    print(f"Episodes   : {args.n_episodes} per condition")
    print(f"Seed       : {args.seed}")
    print(f"Output     : {out_path}")
    print()
    print("Note: JamSpell + model weights load before any progress bar appears.")
    print("      This can take 60–120 s on CPU — the script is not stuck.")
    print()

    all_results = dispatch[args.experiment](args.n_episodes, args.seed)
    save_csv(all_results, out_path)


if __name__ == '__main__':
    main()
