
import os
import sys
import json
import random

import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from pathlib import Path

from miditok.utils import split_files_for_training
from miditok.pytorch_data import DatasetMIDI, DataCollator

def get_dataloader(tokenizer):
    dataset_chunks_dir = Path(tokenizer.datapath, "..", "midi_chunks")
    split_files_for_training(files_paths=tokenizer.datapath,
                             tokenizer=tokenizer,
                             save_dir=dataset_chunks_dir,
                             max_seq_len=1024)
    dataset = DatasetMIDI(
        files_paths=list(dataset_chunks_dir.glob("**/*.mid")),
        tokenizer=tokenizer,
        max_seq_len=1024,
        bos_token_id=tokenizer["BOS_None"],
        eos_token_id=tokenizer["EOS_None"])

    collator = DataCollator(tokenizer.pad_token_id, copy_inputs_as_labels=True)
    dataloader = DataLoader(dataset, batch_size=64, collate_fn=collator)
    return dataloader



