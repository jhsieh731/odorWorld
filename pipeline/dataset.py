import torch
from torch.utils.data import Dataset
import pickle
import numpy as np

# default settings are made for Gaussian (most complex)
class LazyChemDataset(Dataset):
    def __init__(
            self,
            pickle_file,
            index_file,
            n_mixture, # odors per mixture
            num_classes, # total number of odor classes
            dset_type=None, # unif, noisy unif, gaussian
            # n_samples=1, # samples per odor
            transform=None, # data augmentation: ONLY USE if n_samples > 1
            target_transform=None,
        ):
        self.dset_type = dset_type
        # self.n_samples = n_samples
        self.transform = transform
        self.target_transform = target_transform
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
                if self.transform:
                    chem = self.transform(chem)
                odors.append(chem)
                y_tensor[label] = 1
            x_tensor = torch.tensor(sum(odors), dtype=torch.float32)
            if self.target_transform:
                y_tensor = self.target_transform(y_tensor)

        return x_tensor, y_tensor
