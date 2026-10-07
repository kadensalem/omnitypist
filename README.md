# OmniTypist

OmniTypist extends CRTypist into a pixel-based typing simulator that supports
single-finger tapping, sequential two-thumb typing, and two-key chording. The
model combines learned vision and finger agents, working memory, a PPO
supervisor, motor noise, and gaze movement. Chord mode adds a procedural-memory
model whose recall strength decays according to each chord's difficulty.

## Features

- Single-finger and two-thumb typing on keyboard screenshots
- Dynamic two-thumb assignment with a no-crossing constraint
- Hybrid two-key chords that expand into words or morphemes
- Per-chord recall decay and reinforcement
- Separate checkpoints for single-finger, two-thumb, and chord supervisors
- Evaluation scripts for typing mode, vocabulary size, chord difficulty, and
  individual-memory differences
- Publication figures for performance, tap trajectories, and gaze trajectories

## Setup

The project is configured for Python 3.8 in `env.yml`.

```bash
conda env create -f env.yml
conda activate deeptyping
```

To update an existing environment:

```bash
conda env update -f env.yml
```

PyTorch is intentionally installed separately. For Linux with CUDA 11.8:

```bash
pip install torch==2.0.0+cu118 torchvision==0.15.1+cu118 \
  --index-url https://download.pytorch.org/whl/cu118
```

For a CPU-only machine or macOS, install compatible CPU builds instead:

```bash
pip install torch==2.0.0 torchvision==0.15.1
```

Run commands from the repository root. `InternalEnv` also loads the JamSpell
English language model from `outputs/en.bin`; place that model at that path if
it is not already present. Model files under `outputs/` are ignored by Git, so
create the runtime directories and supply pretrained files or train them
locally:

```bash
mkdir -p outputs results figures/plots
```

## Quick start

After placing or training the checkpoints listed below, start an interactive
demo with one of the following commands:

```bash
# Single-finger
python main.py --demo --phrase "hello world"

# Sequential two-thumb
python main.py --demo --two-thumb --phrase "hello world"

# Two-thumb chording
python main.py --demo --two-thumb --chords --chords-active \
  --phrase "the world and all"
```

Add `--gboard` to use the Gboard screenshot directory. The demo continuously
runs episodes; stop it with `Ctrl-C`.

Chord execution requires all three flags: `--two-thumb --chords
--chords-active`. With `--chords` but without `--chords-active`, the environment
uses the chord-compatible observation and action spaces but zeroes the chord
signals and demotes chord actions to normal typing. That configuration is used
only for curriculum training.

## Model checkpoints

| Component | Path |
| --- | --- |
| Foveal encoder | `outputs/f_encoder.pt` |
| Peripheral encoder | `outputs/p_encoder.pt` |
| Vision agent | `outputs/vision_agent.pt` |
| Finger agent shared by both thumbs | `outputs/finger_agent.pt` |
| Working-memory encoder | `outputs/wm_encoder.pt` |
| Single-finger supervisor | `outputs/supervisor_agent.pt` |
| Sequential two-thumb supervisor | `outputs/supervisor_agent_two_thumb.pt` |
| Chord supervisor | `outputs/supervisor_agent_two_thumb_chord.pt` |

The two thumbs use separate `FingerAgent` instances with the same pretrained
weights. Supervisor checkpoints are not interchangeable because the modes have
different observation and action spaces.

## Training

### 1. Pretrain the component models

Train the foveal and peripheral vision encoders:

```bash
python main.py --train --vision-encoder --no-cuda --epochs 5
python main.py --train --peripheral-encoder --batch-size 1 --no-cuda --epochs 50
```

Train or continue the vision agent:

```bash
python main.py --train --vision-agent --total-timesteps 500000
python main.py --train --continue-training --vision-agent \
  --total-timesteps 500000
```

Train or continue the finger agent:

```bash
python main.py --train --finger-agent --total-timesteps 1000000
python main.py --train --continue-training --finger-agent \
  --total-timesteps 1000000
```

Train the working-memory classifier:

```bash
python main.py --train --memory --epochs 50 --batch-size 64 --no-cuda
```

TensorBoard logs are written below `runs/`, for example:

```bash
tensorboard --logdir runs
```

### 2. Train the single-finger supervisor

```bash
python main.py --train --supervisor-agent --total-timesteps 20000000
```

Continue from `outputs/supervisor_agent.pt` with:

```bash
python main.py --train --supervisor-agent --continue-training \
  --total-timesteps 100000
```

### 3. Train the sequential two-thumb supervisor

The plain two-thumb supervisor is trained without chord observations or chord
actions. The existing checkpoint was trained for approximately 10 million PPO
steps:

```bash
python main.py --train --supervisor-agent --two-thumb \
  --total-timesteps 10000000
```

This writes `outputs/supervisor_agent_two_thumb.pt`. Continue it with the same
mode flags so that `main.py` loads the correct checkpoint:

```bash
python main.py --train --supervisor-agent --two-thumb --continue-training \
  --total-timesteps 1000000
```

### 4. Train the chord supervisor in stages

Learning two-thumb typing, proofreading, backspacing, and chording together
from random weights is unreliable. The recommended curriculum starts from the
trained sequential two-thumb policy.

#### Stage 1: transfer the two-thumb policy

```bash
python transfer_weights.py
```

The transfer script maps all shared observation weights into the larger chord
network, initializes the new chord-only inputs to zero, preserves the existing
type/backspace and speed actions, and adds the chord action with an initial
bias of `-0.5`. It writes
`outputs/supervisor_agent_two_thumb_chord.pt`.

#### Stage 2: learn chord selection

```bash
python main.py --train --supervisor-agent --two-thumb --chords \
  --chords-active --continue-training --total-timesteps 5000000 \
  --ent-coef 0.2
```

The higher entropy coefficient is intentional: it gives the new chord action
enough exploration to compete with the already-trained type and backspace
actions. During training, the callback reports chord attempts, successful
chords, and fire rate every 20,000 steps.

Back up the result before refinement if you want to preserve this exploratory
checkpoint:

```bash
cp outputs/supervisor_agent_two_thumb_chord.pt \
  outputs/supervisor_agent_two_thumb_chord_v1.pt
```

#### Stage 3: refine the chord policy

```bash
python main.py --train --supervisor-agent --two-thumb --chords \
  --chords-active --continue-training --total-timesteps 3000000 \
  --ent-coef 0.02
```

This lower-entropy stage sharpens chord selection while retaining the typing,
backspacing, and proofreading behavior learned earlier.

An alternative first stage can be trained from scratch with `--two-thumb
--chords` and without `--chords-active`, but it converges slowly because the
inactive chord action is demoted to typing. Weight transfer is the recommended
workflow.

## Chord vocabulary and memory

The active mappings and their difficulty values are defined in
`data/chord_vocab.py`. The current source contains 11 mappings:

| Keys | Expansion | Difficulty |
| --- | --- | ---: |
| `e` + `l` | `world` | 0.1 |
| `h` + `t` | `the` | 0.1 |
| `a` + `n` | `and` | 0.1 |
| `f` + `o` | `for` | 0.3 |
| `i` + `w` | `with` | 0.3 |
| `b` + `u` | `but` | 0.4 |
| `a` + `l` | `all` | 0.4 |
| `c` + `o` | `com` | 0.6 |
| `i` + `t` | `tion` | 0.6 |
| `e` + `n` | `ent` | 0.7 |
| `h` + `v` | `have` | 0.8 |

Keys are stored as order-independent sorted pairs. Each pair uses one key from
each side of the keyboard. Expansions must contain at least three characters
to provide a benefit over two sequential taps.

`models/chord_memory.py` initializes every chord's recall strength to `1.0` at
the beginning of an episode. Recall decays exponentially, with harder chords
decaying faster, and successful execution reinforces the corresponding chord.
The supervisor observes chord availability and recall strength rather than the
identity of a mapping, so the vocabulary can be changed without retraining the
supervisor.

A chord succeeds only when both thumbs land on their intended keys and their
movement times fall within the configured simultaneity window. A failed chord
records the individual landed characters instead of its expansion.

## Evaluation

Evaluate individual components:

```bash
python main.py --evaluate --vision-agent --no-cuda
python main.py --evaluate --finger-agent
python main.py --evaluate --memory
```

Evaluate supervisor modes:

```bash
# Single-finger
python main.py --evaluate --supervisor-agent

# Sequential two-thumb
python main.py --evaluate --supervisor-agent --two-thumb

# Chording
python main.py --evaluate --supervisor-agent --two-thumb --chords \
  --chords-active
```

The systematic experiment runner supports four experiments and saves raw
episode results as CSV files:

```bash
python evaluate.py --experiment mode --n-episodes 30 \
  --out results/mode.csv
python evaluate.py --experiment vocab-size --n-episodes 30 \
  --out results/vocab_size.csv
python evaluate.py --experiment difficulty --n-episodes 30 \
  --out results/difficulty.csv
python evaluate.py --experiment individual-diff --n-episodes 30 \
  --out results/individual_diff.csv
```

The reported metrics include WPM, inter-keystroke interval, character error
rate, backspaces, gaze behavior, chord attempts, successful chords, chord fire
rate, and target-text chord coverage. Episodes with character error rate of
`0.2` or greater are retried as outliers.

## Figures and analysis

After generating the four result CSVs, create the experiment figures with:

```bash
python export_figures.py
```

Outputs are written to `figures/plots/`. Additional trajectory visualizations
can be generated directly from the trained agents:

```bash
python visualize_demo.py
python visualize_vocab.py
python visualize_combined.py
```

The notebook `analysis.ipynb` contains the corresponding statistical summaries
and exploratory analysis.

## Parameter fitting

Run Bayesian parameter fitting with:

```bash
python -W ignore optimization.py
```

The fitted memory, finger, and vision parameters are stored in `parameters.py`.

## Repository layout

```text
data/chord_vocab.py       Chord mappings and difficulty scores
models/chord_memory.py    Chord recall decay and reinforcement
typing_env/internal_env.py
                          Single-finger, two-thumb, and chord environment logic
transfer_weights.py       Two-thumb-to-chord policy transfer
evaluate.py               Reproducible experiment runner
export_figures.py         Result plots
visualize_*.py            Tap and gaze trajectory figures
outputs/                  Trained component and supervisor checkpoints
results/                  Experiment CSV output
figures/plots/            Generated figures
```
