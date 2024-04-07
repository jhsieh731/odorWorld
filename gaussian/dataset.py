import torch
from torch.utils.data import Dataset
import pickle
import numpy as np
from itertools import product


def gen_index(n, n_mixtures):
	res = [() for _ in range(n ** n_mixtures)]

	lists = [list(range(n)) for _ in range(n_mixtures)]
	things = product(*lists)

	for i, thing in enumerate(things):
		res[i] = tuple(thing)

	return res

class LazyChemDataset(Dataset):
    def __init__(self, pickle_file, index_file, n_mixture, num_classes):
        self.pickle_file = pickle_file
        self.num_classes = num_classes
        with open(index_file, 'rb') as file:
            self.offsets = pickle.load(file)
        self.indices = gen_index(len(self.offsets), n_mixture)
        


    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        with open(self.pickle_file, 'rb') as file:
            # Seek to the position of the desired item
            odor_ids = self.indices[idx]
            odors = []
            y_tensor = torch.zeros(self.num_classes)
            for id in odor_ids:
                file.seek(self.offsets[id])
                chem, label = pickle.load(file)
                odors.append(chem)
                y_tensor[label] = 1
        
        x_tensor = torch.tensor(sum(odors), dtype=torch.float32)
        
        return x_tensor, y_tensor
