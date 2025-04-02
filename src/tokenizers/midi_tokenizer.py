import os
import sys
import json
import random

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from pathlib import Path

from miditok import REMI, TokenizerConfig


class MidiTokenizer():
    # Source: https://github.com/Natooz/MidiTok

    def __init__(self, midi_data_path, tokenizer_path):
        
        BEAT_RES = {(0, 1): 12, (1, 2): 4, (2, 4): 2, (4, 8): 1}
        TOKENIZER_PARAMS = {
            "pitch_range": (21, 109),
            "beat_res": BEAT_RES,
            "num_velocities": 24,
            "special_tokens": ["PAD", "BOS", "EOS"],
            "use_chords": True,
            "use_rests": True,
            "use_tempos": True,
            "use_time_signatures": True,
            "use_programs": False,  # no multitrack here
            "num_tempos": 32,
            "tempo_range": (50, 200),  # (min_tempo, max_tempo)
            }

        self.config = TokenizerConfig(**TOKENIZER_PARAMS)
        self.tokenizer = REMI(self.config)
        self.datapath = midi_data_path
        self.tokenizer_path = tokenizer_path

        print("Training tokenizer...")
        self.train_tokenizer()
    
    def train_tokenizer(self):
        file_paths = list(Path(self.datapath).glob("**/*.mid"))
        self.tokenizer.train(vocab_size=30000, files_paths=file_paths)
        self.tokenizer.save(Path(self.tokenizer_path, "MidiTokenizer.json"))
        print(f"Tokenizer saved to {Path(self.tokenizer_path, "MidiTokenizer.json")}")