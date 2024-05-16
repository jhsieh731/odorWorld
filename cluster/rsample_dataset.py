import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
import random

# default settings are made for Gaussian (most complex)
class RSampleDataset(Dataset):
    def __init__(
            self,
            pickle_file,
            index_file,
            n_mixture, # odors per mixture
            num_classes, # total number of odor classes
            transform=None,
            target_transform=None,
            total_samples=10000,
        ):
        self.transform = transform
        self.target_transform = target_transform
        self.pickle_file = pickle_file
        self.num_classes = num_classes
        self.total_samples = total_samples

        with open(index_file, 'rb') as file:
            self.offsets = pickle.load(file)
        self.n = len(self.offsets)
        self.n_mixture = n_mixture
        total_combinations = self.n ** self.n_mixture
        # self.indices = np.random.randint(total_combinations, size=total_samples)
        self.indices = [random.randrange(0, total_combinations) for _ in range(total_samples)]


    def __len__(self):
        return self.total_samples

    def _index_to_tuple(self, idx):
        """Compute the tuple corresponding to the given index"""
        indices = []
        for _ in range(self.n_mixture):
            indices.append(idx % self.n)
            idx //= self.n
        return tuple(indices)

    def __getitem__(self, idx):
        with open(self.pickle_file, 'rb') as file:
            odor_ids = self._index_to_tuple(self.indices[idx])
            odors = []
            y_tensor = torch.zeros(self.num_classes)
            for id in odor_ids:
                file.seek(self.offsets[id])
                chem, label = pickle.load(file)
                if self.transform:
                    chem = self.transform(chem)
                odors.append(chem)
                y_tensor[label] = 1
            x_tensor = torch.tensor(sum(odors), dtype=torch.float32)
            if self.target_transform:
                y_tensor = self.target_transform(y_tensor)

        return x_tensor, y_tensor
