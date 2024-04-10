import torch
from torch.utils.data import Dataset
import pickle
import numpy as np

class LazyChemDataset(Dataset):
    def __init__(self, pickle_file, index_file, n_mixture, num_classes):
        self.pickle_file = pickle_file
        self.num_classes = num_classes
        with open(index_file, 'rb') as file:
            self.offsets = pickle.load(file)
        self.n = len(self.offsets)
        self.n_mixture = n_mixture

    def __len__(self):
        return self.n ** self.n_mixture

    def index_to_tuple(self, idx):
        """Compute the tuple corresponding to the given index `idx`."""
        indices = []
        for _ in range(self.n_mixture):
            indices.append(idx % self.n)
            idx //= self.n
        return tuple(indices)

    def __getitem__(self, idx):
        with open(self.pickle_file, 'rb') as file:
            odor_ids = self.index_to_tuple(idx)
            odors = []
            y_tensor = torch.zeros(self.num_classes)
            for id in odor_ids:
                file.seek(self.offsets[id])
                chem, label = pickle.load(file)
                odors.append(chem)
                y_tensor[label] = 1

        x_tensor = torch.tensor(sum(odors), dtype=torch.float32)
        return x_tensor, y_tensor
