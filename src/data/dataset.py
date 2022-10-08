import os
import pickle
import torch
from torch.utils.data import Dataset

class MiniFlickrDataset(Dataset):
    def __init__(self, path): 
        with open(path, 'rb') as f:
            self.data = pickle.load(f)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]