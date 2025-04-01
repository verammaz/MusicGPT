
import os
import sys
import json
import random

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from pathlib import Path

from miditok.utils import split_files_for_training
from miditok.data_augmentation import augment_dataset
from miditok.pytorch_data import DatasetMIDI, DataCollator
from miditok import REMI, TokenizerConfig


class MidiTokenizer():
    # Source: https://github.com/Natooz/MidiTok

    def __init__(self, midi_data_path, tokenizer_path, train = True):
        
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

        if train:
            self.train_tokenizer()
    
    def train_tokenizer(self):
        file_paths = list(Path(self.datapath).glob("**/*.mid"))
        self.tokenizer.train(vocab_size=30000, files_paths=file_paths)
        self.tokenizer.save(Path(self.tokenizer_path, "MidiTokenizer.json"))



def get_dataloader(tokenizer):
    # Split MIDI paths in train/valid/test sets
    midipaths = list(Path(tokenizer.datapath).glob("**/*.mid"))
    total_num_files = len(midipaths)
    num_files_valid = round(total_num_files * 0.15)
    num_files_test = round(total_num_files * 0.15)
    random.shuffle(midipaths)
    midi_paths_valid = midipaths[:num_files_valid]
    midi_paths_test = midipaths[num_files_valid:num_files_valid + num_files_test]
    midi_paths_train = midipaths[num_files_valid + num_files_test:]
    
    # Chunk MIDIs and perform data augmentation on each subset independently
    for files_paths, subset_name in ((midi_paths_train, "train"), (midi_paths_valid, "valid"), (midi_paths_test, "test")):

        # Split the MIDIs into chunks of sizes approximately about 1024 tokens
        subset_chunks_dir = Path(f"Midi_{subset_name}")
        split_files_for_training(
            files_paths=files_paths,
            tokenizer=tokenizer,
            save_dir=subset_chunks_dir,
            max_seq_len=1024,
            num_overlap_bars=2)

        # Perform data augmentation
        augment_dataset(
            subset_chunks_dir,
            pitch_offsets=[-12, 12],
            velocity_offsets=[-4, 4],
            duration_offsets=[-0.5, 0.5])

    # Create Dataset and Collator for training
    midi_paths_train = list(Path("Midi_train").glob("**/*.mid")) + list(Path("Maestro_train").glob("**/*.midi"))
    midi_paths_valid = list(Path("Midi_valid").glob("**/*.mid")) + list(Path("Maestro_valid").glob("**/*.midi"))
    midi_paths_test = list(Path("Midi_test").glob("**/*.mid")) + list(Path("Maestro_test").glob("**/*.midi"))
    kwargs_dataset = {"max_seq_len": 1024, "tokenizer": tokenizer, "bos_token_id": tokenizer["BOS_None"], "eos_token_id": tokenizer["EOS_None"]}
    dataset_train = DatasetMIDI(midi_paths_train, **kwargs_dataset)
    dataset_valid = DatasetMIDI(midi_paths_valid, **kwargs_dataset)
    dataset_test = DatasetMIDI(midi_paths_test, **kwargs_dataset)
        
        
    



