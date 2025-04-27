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

import editdistance as ed  # pip install editdistance
import pretty_midi  # pip install pretty_midi
from miditok import REMI, TokenizerConfig

# ---------------------------------------------------------------------------
# Tokenisation helpers
# ---------------------------------------------------------------------------

BEAT_RES = {(0, 1): 12, (1, 2): 4, (2, 4): 2, (4, 8): 1}
TOKENIZER_PARAMS = dict(
    pitch_range=(21, 109),
    beat_res=BEAT_RES,
    num_velocities=24,
    special_tokens=("PAD", "BOS", "EOS"),
    use_chords=True,
    use_rests=True,
    use_tempos=True,
    use_time_signatures=True,
    use_programs=False,
    num_tempos=32,
    tempo_range=(50, 200),
)


def load_or_train_tokenizer(train_dir: Path, tok_path: Path) -> REMI:
    """Load a MidiTok REMI tokenizer, or train & save if the file is absent."""
    if tok_path.exists():
        return REMI(params=TokenizerConfig(**TOKENIZER_PARAMS)).load(tok_path)

    print("[Tokenizer] Training new tokenizer…", file=sys.stderr)
    file_paths = list(train_dir.rglob("*.mid"))
    tokenizer = REMI(TokenizerConfig(**TOKENIZER_PARAMS))
    tokenizer.train(vocab_size=30_000, files_paths=file_paths)
    tokenizer.save(tok_path)
    print(f"[Tokenizer] Saved to {tok_path}", file=sys.stderr)
    return tokenizer


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


# ---------------------------------------------------------------------------
# Utility: cache training tokens to disk (pickle) so we don’t re‑tokenise.
# ---------------------------------------------------------------------------

def build_or_load_cache(train_dir: Path, tokenizer: REMI, cache_path: Path):
    if cache_path.exists():
        return pickle.load(open(cache_path, "rb"))

    print("[Cache] Building training token cache…", file=sys.stderr)
    tokens = []
    for midi_path in train_dir.rglob("*.mid"):
        try:
            midi = pretty_midi.PrettyMIDI(str(midi_path))
            tok = tokenizer.midi_to_tokens(midi)[0].ids
            tokens.append(tok)
        except Exception as e:
            print(f"  ! Skipped {midi_path} ({e})", file=sys.stderr)
    pickle.dump(tokens, open(cache_path, "wb"))
    return tokens


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def print_report(generated_path: Path, matches, train_index_to_path):
    print(f"\n=== {generated_path.name} ===")
    if not matches:
        print("  ✔ No significant similarity found.")
        return
    print("  ⚠ Similarities:")
    for idx, bleu, edit in matches:
        path = train_index_to_path[idx]
        print(f"   • {path}  BLEU={bleu:.2f}  edit={edit:.3f}")
    if len(matches) >= 10:
        print("   …and more (stopped after 10 matches).")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Check MusicGPT generations against training set.")
    p.add_argument("--train_dir", type=Path, required=True)
    p.add_argument("--gen_dir", type=Path, required=True)
    p.add_argument("--tokenizer_path", type=Path, required=True)
    p.add_argument("--cache_path", type=Path, default=Path("train_tokens.pkl"))
    p.add_argument("--conditioned", action="store_true", help="Outputs are continuations; skip first half.")
    p.add_argument("--max_matches", type=int, default=10)
    args = p.parse_args()

    tokenizer = load_or_train_tokenizer(args.train_dir, args.tokenizer_path)
    train_tokens = build_or_load_cache(args.train_dir, tokenizer, args.cache_path)
    evaluator = SimilarityEvaluator(train_tokens)

    # Build index → path list for reporting
    train_paths = [str(p) for p in args.train_dir.rglob("*.mid")]

    for gen_path in args.gen_dir.rglob("*.mid"):
        try:
            midi = pretty_midi.PrettyMIDI(str(gen_path))
            seq = tokenizer.midi_to_tokens(midi)[0].ids
            if args.conditioned:
                seq = seq[len(seq) // 2 :]  # take second half only
            matches = evaluator.find_matches(seq, args.max_matches)
            print_report(gen_path, matches, train_paths)
        except Exception as e:
            print(f"! Error on {gen_path}: {e}", file=sys.stderr)


if __name__ == "__main__":
    main()
