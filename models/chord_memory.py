"""ChordMemory: per-chord procedural memory with exponential decay.

Models the long-term recall strength of each chord in the vocabulary.
Harder chords (higher CHORD_DIFFICULTY score) decay faster when the typist
hasn't used them recently.

Design notes
------------
- Decay follows the same exponential model as working memory (parameters.py):
      recall_strength *= exp(-K * t)   where  K = lambda * 0.3 * (1 + difficulty)
  The lambda_param passed to decay_all() is parameters[0], the same
  working-memory decay parameter the supervisor already receives.

- Reinforcement: a successful chord execution increases recall_strength by
  REINFORCE_AMOUNT (default 0.2), capped at 1.0.

- recall_strength() is deterministic and used directly in the supervisor's
  observation.  Probabilistic gating (can the agent recall this chord right
  now?) is left to the supervisor's policy to learn from the continuous value.

- reset() restores all strengths to 1.0 at the start of each episode, matching
  how Memory.target() resets working-memory certainty.
"""

import numpy as np
from typing import Dict, Optional, Tuple
from data.chord_vocab import CHORD_VOCAB, CHORD_DIFFICULTY

REINFORCE_AMOUNT = 0.2  # increase in recall_strength per successful chord


class ChordMemory:
    def __init__(self):
        # All recall strengths start fully known at episode start.
        self.recall_strengths: Dict[Tuple[str, str], float] = {k: 1.0 for k in CHORD_VOCAB}

    # ------------------------------------------------------------------
    # Episode lifecycle
    # ------------------------------------------------------------------

    def reset(self):
        """Restore full recall at the start of each episode."""
        self.recall_strengths: Dict[Tuple[str, str], float] = {k: 1.0 for k in CHORD_VOCAB}

    # ------------------------------------------------------------------
    # Lookup
    # ------------------------------------------------------------------

    def next_chord(self, typed_text: str, target_text: str) -> Optional[tuple]:
        """Return the chord key whose expansion matches the start of the
        remaining target text, or None if no chord applies.

        Args:
            typed_text: what has been physically typed so far.  Callers should
                        pass typed_text (physical truth) rather than
                        wm.recall_text so that WM drift does not produce false
                        chord opportunities on already-typed prefixes.
            target_text: the full target phrase for this episode.

        Returns:
            A sorted 2-tuple (chord key) from CHORD_VOCAB, or None.
        """
        remaining = target_text[len(typed_text):]
        for chord_key, expansion in CHORD_VOCAB.items():
            if remaining.startswith(expansion):
                return chord_key
        return None

    # ------------------------------------------------------------------
    # Recall state
    # ------------------------------------------------------------------

    def recall_strength(self, chord_key: tuple) -> float:
        """Return the current recall strength for chord_key in [0.0, 1.0].

        Returns 0.0 if chord_key is not in the vocabulary.
        """
        return self.recall_strengths.get(chord_key, 0.0)

    # ------------------------------------------------------------------
    # Updates
    # ------------------------------------------------------------------

    def reinforce(self, chord_key: tuple):
        """Increase recall strength after a successful chord execution."""
        current = self.recall_strengths.get(chord_key, 0.0)
        self.recall_strengths[chord_key] = min(1.0, current + REINFORCE_AMOUNT)

    def decay_all(self, elapsed_ms: float, lambda_param: float):
        """Apply temporal decay to every chord's recall strength.

        Mirrors the decay() function in parameters.py but scales K by
        (1 + difficulty) so harder chords are forgotten faster.

        Args:
            elapsed_ms:    time elapsed since last proofreading (milliseconds).
            lambda_param:  parameters[0], the working-memory decay parameter.
        """
        t = elapsed_ms / 1000.0  # convert ms → seconds
        for chord_key in self.recall_strengths:
            difficulty = CHORD_DIFFICULTY.get(chord_key, 0.5)
            K = lambda_param * 0.3 * (1.0 + difficulty)
            self.recall_strengths[chord_key] *= np.exp(-K * t)
