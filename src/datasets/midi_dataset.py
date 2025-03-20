# Source : https://github.com/eyalmazuz/MolGen/blob/master/MolGen/src/datasets/smiles_dataset.py

from random import sample
import pandas as pd

from rdkit import Chem
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from  src.tokenizers import CharTokenizer


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


