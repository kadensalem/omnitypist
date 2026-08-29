# CRTypist — Development Guidelines

## Workflow: Plan Before Code

Before writing any code, a written plan must exist and be agreed upon. This project is a research
implementation tightly coupled to the paper:

> **CRTypist: Simulating Touchscreen Typing on Any Keyboard Using a Computational Rational Model**
> (PDF: `../crtypist.pdf`)

Follow this process for every non-trivial change:

1. **Read the relevant paper section first.** Understand the model's intent before touching code.
2. **Write or update the plan.** The plan lives in the conversation or in a `PLAN.md` if the
   feature is large. The plan must cover: what changes, why, which files, and what the expected
   behavior is.
3. **Get agreement on the plan** before writing a single line of implementation code.
4. **Implement against the plan.** Each step in the plan maps to a concrete code change.
5. **If anything goes wrong during implementation** (unexpected behavior, a design assumption
   turns out to be wrong, a test fails, an import breaks), **stop coding**. Go back to the paper,
   re-read the relevant section, update the plan, then resume.

**Never skip straight to code. Never patch around a broken assumption — understand it first.**

---

## Code Quality Standard

All code must meet the bar a senior engineer would accept in a pull review:

- **Correctness first.** Code must match the paper's described model behavior exactly.
  When the paper is ambiguous, document the interpretation with an inline comment.
- **No dead code.** Remove unused variables, imports, and placeholder comments.
- **No hacks.** If something requires a workaround, explain why in a comment.
- **Single responsibility.** Each method does one thing. If a method is doing two things, split it.
- **Consistent style.** Match the style of the surrounding file (naming, spacing, type of comments).
- **Explicit over implicit.** Prefer clear variable names over terse ones. `active_finger` not `af`.
- **No silent failures.** Raise a descriptive `ValueError` or `AssertionError` rather than
  silently doing the wrong thing (e.g., if an unknown finger side is passed).

---

## Project-Specific Rules

### Environment / RL
- The `FingerEnv` (single-finger training env) must not be modified. It is used for training
  individual finger agents and must remain single-finger.
- `InternalEnv` is the only place where multi-finger coordination logic lives.
- `KeyboardEnv` (base class in `kbd_env.py`) must not be modified unless there is no other option.
  Prefer overriding in `InternalEnv`.
- `self.finger` on `KeyboardEnv` is used by `_finger_on()` and `_where()`. Always sync it to the
  active finger's position before calling those methods.

### Model checkpoints
- Single-thumb models live in `outputs/`. Do not overwrite them.
- Two-thumb supervisor checkpoints must be saved separately, e.g. `outputs/supervisor_agent_two_thumb.pt`.
- Finger agent weights are shared between left and right thumbs — do not create duplicate weight files.
- Chord supervisor checkpoint: `outputs/supervisor_agent_two_thumb_chord.pt`.
  Before any experiment that might overwrite it, save a backup:
  `cp outputs/supervisor_agent_two_thumb_chord.pt outputs/supervisor_agent_two_thumb_chord_vN.pt`

### Adding features
- New env modes (e.g. two-thumb) must be gated behind a flag passed to `InternalEnv.__init__`
  so the existing single-thumb demo continues to work unchanged.
- New CLI flags in `main.py` must have a descriptive `help=` string.

### Chord vocabulary
- Chord vocabulary lives in `data/chord_vocab.py`. It can be changed freely without retraining
  the supervisor — the supervisor never sees which chord is available, only the binary
  `chord_available` signal and the scalar `chord_recall_strength`. The supervisor generalises to
  any vocabulary automatically.
- When adding new chords, set a sensible `CHORD_DIFFICULTY` value so recall strength decay
  remains meaningful.

---

### Weight transfer (`transfer_weights.py`)

When bootstrapping a chord supervisor from `supervisor_agent_two_thumb.pt`, use:
```bash
python transfer_weights.py
```

**Critical:** `gym.spaces.Dict` in gym 0.21 sorts keys **alphabetically**, so obs feature
columns are NOT in insertion order. The actual column layout is:

```
Two-thumb (11 cols, alphabetical):
  P(0-2), certainty(3), correctness(4),
  left_finger_in_action(5-6), right_finger_in_action(7-8), vision_in_action(9-10)

Chord (16 cols, alphabetical):
  P(0-2), backspace_ready(3-4)[NEW], certainty(5),
  chord_available(6-7)[NEW], chord_recall_strength(8)[NEW],
  correctness(9), left_finger_in_action(10-11),
  right_finger_in_action(12-13), vision_in_action(14-15)
```

Each `Discrete(2)` space contributes **2 columns** (SB3 one-hot encodes discrete obs before
the MLP extractor). `transfer_weights.py` computes the mapping dynamically from the actual obs
spaces using `_key_offsets()` — do not hard-code column numbers.

**Initial chord bias:** Set to `-0.5` in `transfer_weights.py` (`CHORD_ACTION_INIT_BIAS`).
A value of `-2.0` proved too strong — chord was effectively never sampled during phase 2
training (`attempts=1` in 240k steps). `-0.5` gives a reasonable starting point.

Action output transfer (action_net rows):
- Two-thumb `(14, 64)` → chord `(15, 64)`
- Rows 0-3: copy directly (vision goal + type/backspace)
- Row 4: zero weight, `CHORD_ACTION_INIT_BIAS` bias (chord action, new)
- Rows 5-14: copy from rows 4-13 (speeds)

---

### Curriculum learning for chord training

Training the chord supervisor from random weights requires it to simultaneously learn
two-thumb typing, backspacing, proofreading, AND chording. This is hard. The recommended
approach is weight transfer (option A below) followed by phase 2.

**Option A — transfer weights from the existing two-thumb supervisor (preferred):**
```bash
python transfer_weights.py
```
Then run phase 2 directly (skip phase 1 training entirely).

**Option B — train phase 1 from scratch (chord signals zeroed):**
```bash
rm outputs/supervisor_agent_two_thumb_chord.pt
python main.py --train --supervisor-agent --two-thumb --chords \
    --total-timesteps 3000000
```
`--chords-active` is omitted. Chord obs dims are always zero; chord action demotes to type.
Note: phase 1 training from scratch converges very slowly (~2.22 plateau) because:
- The 3-way action[1] space (type/backspace/chord-demoted-to-type) dilutes backspace exploration
- Two "type" actions (0 and 2) both produce forward progress, so the model ignores backspace
The weight transfer approach avoids this entirely.

**Phase 2 — learn chording on top of the typing policy:**
```bash
python main.py --train --supervisor-agent --two-thumb --chords --chords-active \
    --continue-training --total-timesteps 5000000 --ent-coef 0.2
```
`--ent-coef 0.2` is required to get meaningful chord exploration. Without it, chord is
almost never sampled even with `-0.5` bias, because the trained type/backspace logits are
much larger than the chord logit in the key "both fingers idle" states. With `ent_coef=0.0`
or `0.05`, attempts stay near 0 for hundreds of thousands of steps.

**Phase 2 refinement** (after phase 2 completes):
```bash
python main.py --train --supervisor-agent --two-thumb --chords --chords-active \
    --continue-training --total-timesteps 3000000 --ent-coef 0.02
```
Reduces entropy noise while preserving the learned chord policy. The model already knows
how to chord — this phase sharpens when to chord without degrading typing quality.

The obs space shape is fixed across all phases so `--continue-training` always works.

---

### Chord mode (`--two-thumb --chords`)
- Chord mode is gated behind `chords=True` in `InternalEnv.__init__`. It requires `two_thumb=True` (raises `ValueError` otherwise).
- Action space becomes `MultiDiscrete([2, 3, len(SPEEDS)])` — `action[1]=2` means "attempt chord."
  The existing two-thumb (no-chord) env keeps `MultiDiscrete([2, 2, ...])` — its checkpoint is not affected.
- Chord execution is **atomic**: both fingers move, noise is applied, timing + motor checks resolve, and the result (expansion or fallback chars) is appended in the same step. No in-flight period.
- The chord error gate uses `last_char_wrong` (physical `typed_text[-1] != target_text[...]`) rather than `wm.correctness()`. WM correctness decays over time even on perfectly correct text, which blocked all chord opportunities for words like "and" that appear mid-phrase. `last_char_wrong` is the correct gate: it fires only on actual typing errors, not cognitive decay.
- `ChordMemory` (`models/chord_memory.py`) tracks per-chord recall strength with exponential decay. Harder chords (higher `CHORD_DIFFICULTY`) decay faster. It is reset each episode alongside working memory.
- The chord obs includes `backspace_ready` (1 when `typed_text[-1] != target_text[...]` AND both fingers idle). Deliberately uses physical `typed_text`, NOT `wm.correctness()`.
- Chord success reward: `chord_attempt_bonus + reward_shaping_value * len(expansion) + reward_shaping_value * (len(expansion) - 1)`. The attempt bonus (0.15) fires for every valid attempt regardless of motor success, providing the gradient signal that connects `chord_available=1` to choosing chord. Without it, the model receives too few rewarded chord samples to learn from.
- Chord supervisor checkpoint: `outputs/supervisor_agent_two_thumb_chord.pt` — do not overwrite the plain two-thumb checkpoint.
- `chord_attempts` and `chords_fired` are tracked per-episode on `InternalEnv` and surfaced in the `info` dict at episode end. `TrainingCallback` accumulates and prints them every `check_freq` steps. Watch `fire_rate` during phase 2: a healthy run stabilises around 50-55% and slowly climbs. Linear (non-accelerating) growth of attempts means chord is not being reinforced — increase `ent_coef`.

### Training diagnostics
- `[chord] attempts=N fired=M fire_rate=X%` is printed every 20k steps during chord training.
- `attempts=0` at 100k+ steps means chord is not being explored — `ent_coef` is too low.
- Linear attempt growth (not accelerating) means the model is not learning chord value — check that `chord_attempt_bonus` is active and `ent_coef` is high enough.
- Accelerating attempt growth + rising fire rate = healthy phase 2 training.
- Mean reward plateau during phase 2 with `ent_coef=0.2` is expected — the entropy noise
  holds down per-episode reward. Switch to refinement phase once attempts are in the hundreds
  per 100k steps.

### Testing changes
After any implementation, run:

```bash
conda activate deeptyping
cd /path/to/crtypist

# Check that the existing single-thumb demo still imports cleanly
python -c "from typing_env.internal_env import InternalEnv; print('import OK')"

# Check that the chord env can be instantiated and stepped
python -c "
from typing_env.internal_env import InternalEnv
from setting import KEYS, PLACES, CHARS
env = InternalEnv(
    img_folder='kbd1k/keyboard_dataset/',
    position_file='kbd1k/keyboard_label.csv',
    text_path='./data/sentences.txt',
    vision_path='outputs/vision_agent.pt',
    finger_path='outputs/finger_agent.pt',
    chars=CHARS, places=PLACES, keys=KEYS,
    two_thumb=True, chords=True
)
obs = env.reset(target_text='the cat', reset_kbd=True)
assert 'chord_available' in obs, 'missing chord_available'
assert 'chord_recall_strength' in obs, 'missing chord_recall_strength'
import numpy as np
action = np.array([0, 2, 5])   # chord action at speed index 5
obs, r, done, _ = env.step(action)
print('chord obs keys:', list(obs.keys()))
print('chord step OK — chords_fired:', env.chords_fired, 'chord_attempts:', env.chord_attempts)
"

# Check that the two-thumb env can be instantiated without errors
python -c "
from typing_env.internal_env import InternalEnv
from setting import KEYS, PLACES, CHARS
env = InternalEnv(
    img_folder='kbd1k/keyboard_dataset/',
    position_file='kbd1k/keyboard_label.csv',
    text_path='./data/sentences.txt',
    vision_path='outputs/vision_agent.pt',
    finger_path='outputs/finger_agent.pt',
    chars=CHARS, places=PLACES, keys=KEYS,
    two_thumb=True
)
obs = env.reset(target_text='hi', reset_kbd=True)
print('obs keys:', list(obs.keys()))
print('two-thumb env OK')
"
```

---

## Chording — Lessons Learned

### What worked

- Weight transfer from `supervisor_agent_two_thumb.pt` is the correct starting point.
  The transferred model has correct typing and backspace behavior; phase 2 only needs to
  add chord on top.
- `chord_attempt_bonus = 0.15` in `_step_chord` is essential. Without it, the gradient
  signal from chord is too sparse (model discovers chord too rarely) to learn from.
- `ent_coef=0.2` during phase 2 is necessary to overcome the high-confidence type/backspace
  logits in "both fingers idle" states. Lower values (0.0, 0.05) produced near-zero chord
  exploration even with hundreds of thousands of training steps.
- Fire rate of ~50-55% is normal and healthy. The chord timing window and motor noise mean
  roughly half of valid attempts succeed.
- The chord vocabulary can be changed freely without retraining. The supervisor responds to
  `chord_available` (binary) and `chord_recall_strength` (scalar), not the specific expansion.

### Key failure modes encountered

**Phase 1 from scratch never learned backspacing:**
The 3-way action[1] space (type=0, backspace=1, chord-demoted-to-type=2) means two actions
both produce forward progress. The model converges to always-type and ignores backspace.
Solution: use weight transfer instead of training phase 1 from scratch.

**Weight transfer column mapping bug (gym 0.21):**
`gym.spaces.Dict` sorts keys alphabetically. The first transfer implementation assumed
insertion order and mapped features to the wrong columns (e.g., certainty → backspace_ready
columns), producing all-backspace behavior. Fix: `_key_offsets()` in `transfer_weights.py`
computes the column layout dynamically from the actual obs space.

**Chord never explored with low ent_coef:**
When the trained type/backspace logits are large (~4.7 in idle states), chord logit (-0.5)
has near-zero probability even under stochastic sampling. `attempts=1` in 240k steps with
`ent_coef=0.05`. Fix: `ent_coef=0.2` forces meaningful exploration.

**Chord attempt bonus missing — linear attempt growth:**
Without the per-attempt bonus, the model receives chord reward too rarely to learn from.
Attempts grow linearly (pure entropy exploration) rather than accelerating (learned preference).
Fix: `chord_attempt_bonus=0.15` in `_step_chord`, applied before success/failure resolution.

### Reward shaping oscillated in the first implementation (reverted)

The first full chord implementation was reverted after reward shaping proved intractable.
Notes preserved for reference:
- Chord-miss penalty too weak (0.5×): chords never fired, absorbed by sequential reward.
- Chord-miss penalty too strong (2.0×): model chose action[1]=2 every step.
- Correctness gate (only penalise when correctness ≥ 0.9) is necessary: without it, the
  penalty fires even when errors exist and prevents backspace exploration.
- The current implementation avoids a chord-miss penalty entirely; instead, the attempt bonus
  provides positive gradient for valid attempts and the fallback path applies misfire penalties.
