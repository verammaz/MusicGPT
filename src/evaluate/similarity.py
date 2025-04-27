"""
evaluate_similarity.py
~~~~~~~~~~~~~~~~~~~~~~
CLI script to measure memorisation of MusicGPT outputs.

Features
--------
- **Tokenise** training MIDI files *once* (MidiTok‑REMI) and cache them on disk.
- Tokenise generated MIDI files and compare to training set with
  *n‑gram BLEU‑4* and *normalised edit distance*.
- Early‑exit after `--max_matches` similarities per generated file.
- Nice, aligned console report.
- Option `--conditioned` → when the output is a continuation of a train song,
  only the *second half* of the generation is checked.

Usage
-----
```bash
python evaluate_similarity.py \
       --train_dir data/train_midis \
       --gen_dir   samples/midis \
       --tokenizer_path artifacts/tokenizer.json \
       --conditioned   # optional
```
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path
from typing import List, Sequence, Tuple

import editdistance as ed 

# ---------------------------------------------------------------------------
# Similarity metrics
# ---------------------------------------------------------------------------

def ngram_counts(seq: Sequence[int], n: int = 4):
    from collections import Counter

    return Counter(tuple(seq[i : i + n]) for i in range(len(seq) - n + 1))


def bleu_overlap(sample: Sequence[int], reference: Sequence[int], n: int = 4) -> float:
    s_counts, r_counts = ngram_counts(sample, n), ngram_counts(reference, n)
    overlap = sum((s_counts & r_counts).values())
    return overlap / max(1, sum(s_counts.values()))


def norm_edit_distance(a: Sequence[int], b: Sequence[int]) -> float:
    return ed.eval(a, b) / max(len(a), len(b))


# ---------------------------------------------------------------------------
# Main evaluator
# ---------------------------------------------------------------------------
class SimilarityEvaluator:
    def __init__(
        self,
        train_tokens: List[List[int]],
        bleu_thr: float = 0.80,
        edit_thr: float = 0.05,
    ):
        self.train_tokens = train_tokens
        self.bleu_thr = bleu_thr
        self.edit_thr = edit_thr

    def find_matches(
        self, sample: Sequence[int], max_matches: int = 10
    ) -> List[Tuple[int, float, float]]:
        """Return up to *max_matches* (index, bleu, edit) pairs that exceed thresholds."""
        matches = []
        for idx, ref in enumerate(self.train_tokens):
            bleu = bleu_overlap(sample, ref)
            edit = norm_edit_distance(sample, ref)
            if bleu > self.bleu_thr or edit < self.edit_thr:
                matches.append((idx, bleu, edit))
                if len(matches) >= max_matches:
                    break
        return matches

