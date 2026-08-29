"""Chord vocabulary for the hybrid chording scheme.

Each chord maps a sorted 2-key tuple (both keys must be in KEYS) to an
expansion string.  The expansion must be longer than 2 characters so that
the chord produces more output than typing the two keys sequentially would.

Keys are stored as sorted tuples so that lookup is order-independent:
always normalise with  tuple(sorted([k1, k2]))  before querying.

CHORD_DIFFICULTY is a per-chord float in [0.0, 1.0] where 0.0 = very easy
(high-frequency word, obvious mnemonic) and 1.0 = very hard.  ChordMemory
uses this to set per-chord decay rates: harder chords are forgotten faster.
"""
from typing import Dict, Optional, Tuple

# Sorted-tuple key → expansion string.
# Selection criteria:
#   - One key from the left half of QWERTY (x < 128 px on 256 px-wide kbd):
#       q w e r t  /  a s d f g  /  z x c v b
#   - One key from the right half:
#       y u i o p  /  h j k l  /  n m
#   - Expansion ≥ 3 chars (net benefit over sequential typing)
#   - Keys form the first two letters or a clear mnemonic for the expansion
CHORD_VOCAB: Dict[Tuple[str, str], str] = {
    ('e', 'l'): "world",
    ('h', 't'): "the",   # th → the  (most common English word)
    ('a', 'n'): "and",   # an → and
    ('f', 'o'): "for",   # fo → for
    ('i', 'w'): "with",  # wi → with
    ('b', 'u'): "but",   # bu → but
    ('a', 'l'): "all",   # al → all
    ('c', 'o'): "com",   # co → com  (common prefix: come/company/com-)
    ('i', 't'): "tion",  # ti → tion (common suffix)
    ('e', 'n'): "ent",   # en → ent  (common suffix: -ment/-ent)
    ('h', 'v'): "have",  # hv → have (non-obvious; hardest chord)
}

# Per-chord difficulty in [0.0, 1.0].  Lower = easier to recall.
# Drives decay rate in ChordMemory: K = lambda * 0.3 * (1 + difficulty).
CHORD_DIFFICULTY: Dict[Tuple[str, str], float] = {
    ('e', 'l'): 0.1,
    ('h', 't'): 0.1,   # the  — trivially memorable
    ('a', 'n'): 0.1,   # and  — trivially memorable
    ('f', 'o'): 0.3,   # for
    ('i', 'w'): 0.3,   # with
    ('b', 'u'): 0.4,   # but
    ('a', 'l'): 0.4,   # all
    ('c', 'o'): 0.6,   # com
    ('i', 't'): 0.6,   # tion
    ('e', 'n'): 0.7,   # ent
    ('h', 'v'): 0.8,   # have — non-obvious key mapping
}


def lookup_chord(k1: str, k2: str) -> Optional[str]:
    """Return the expansion for the chord (k1, k2), or None if unknown.

    Key order does not matter; the tuple is normalised internally.
    """
    return CHORD_VOCAB.get(tuple(sorted([k1, k2])))
