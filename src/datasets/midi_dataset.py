from miditok import REMI, TokenizerConfig
from miditok.pytorch_data import DatasetMIDI, DataCollator
from miditok.utils import split_files_for_training
from torch.utils.data import DataLoader
from pathlib import Path


class MidiDataset(Dataset):

    def __init__(self,
                 data_path,
                 tokenizer,
                 max_len):

        self.max_len = max_len
        self.data_path = data_path 
        self._tracks = self.load_tracks()
        self.tokenizer = tokenizer

    @property
    def tracks(self):
        return self._tracks
        
    def load_tracks(self):
        pass

    def __len__(self):
        pass

    def __getitem__(self, idx):
        pass
    
    def get_vocab_size(self):
        return self.tokenizer.vocab_size
    
    def get_block_size(self):
        return self.max_len


