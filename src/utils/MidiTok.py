import json
import sys
from miditok import REMI, TokenizerConfig
from miditok.pytorch_data import DatasetMIDI, DataCollator
from miditok.utils import split_files_for_training
from torch.utils.data import DataLoader
from pathlib import Path

# Source: https://github.com/Natooz/MidiTok

class MidiTokenizer():

    def __init__(self, midi_data_path, tokenizer_path, train = True):
        self.config = TokenizerConfig(num_velocities=16, use_chords=True, use_programs=True)
        self.tokenizer = REMI(self.config)
        self.datapath = midi_data_path
        self.tokenizer_path = tokenizer_path

        if train:
            self.train_tokenizer()
    
    def train_tokenizer(self):
        file_paths = list(Path(self.datapath).glob("**/*.mid"))
        self.tokenizer.train(vocab_size=30000, files_paths=file_paths)
        self.tokenizer.save(Path(self.tokenizer_path, "tokenizer.json"))


# Split MIDIs into smaller chunks for training
dataset_chunks_dir = Path("path", "to", "midi_chunks")
split_files_for_training(
    files_paths=files_paths,
    tokenizer=tokenizer,
    save_dir=dataset_chunks_dir,
    max_seq_len=1024,
)

# Create a Dataset, a DataLoader and a collator to train a model
dataset = DatasetMIDI(
    files_paths=list(dataset_chunks_dir.glob("**/*.mid")),
    tokenizer=tokenizer,
    max_seq_len=1024,
    bos_token_id=tokenizer["BOS_None"],
    eos_token_id=tokenizer["EOS_None"],
)
collator = DataCollator(tokenizer.pad_token_id, copy_inputs_as_labels=True)
dataloader = DataLoader(dataset, batch_size=64, collate_fn=collator)