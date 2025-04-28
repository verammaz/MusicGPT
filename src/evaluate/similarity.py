
from __future__ import annotations


from typing import List, Sequence, Tuple

import editdistance as ed 

# ---------------------------------------------------------------------------
# Similarity metrics
# ---------------------------------------------------------------------------

def ngram_counts(seq, n=4):
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
    def __init__(self, train_tokens, bleu_thr, edit_thr):
        self.train_tokens = train_tokens
        self.bleu_thr = bleu_thr
        self.edit_thr = edit_thr

    def find_matches(self, sample, max_matches=10):
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

