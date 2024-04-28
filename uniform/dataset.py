import torch
from torch.utils.data import Dataset
import pickle
import numpy as np

class LazyChemDataset(Dataset):
    def __init__(self, pickle_file, index_file):
        self.pickle_file = pickle_file
        with open(index_file, 'rb') as file:
            self.offsets = pickle.load(file)

    def __len__(self):
        return len(self.offsets)

    def __getitem__(self, idx):
        with open(self.pickle_file, 'rb') as file:
            # Seek to the position of the desired item
            file.seek(self.offsets[idx])
            chem = pickle.load(file)
        
        # Assuming chem[0] is features and chem[1] is label
        x_tensor = torch.tensor(chem[0], dtype=torch.float32)
        y_tensor = torch.tensor(chem[1], dtype=torch.float32).reshape(1)  # Adjust reshape as needed
        
        return x_tensor, y_tensor
