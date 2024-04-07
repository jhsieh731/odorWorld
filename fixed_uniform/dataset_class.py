import torch
from torch.utils.data import Dataset
import pickle as pkl

class FixedUniformDataset(Dataset):
    def __init__(self, split_file, transform=None, target_transform=None):
        self.transform = transform
        self.target_transform = target_transform

        # Load all items at once
        self.items = []  # Use a list to collect items
        with open(split_file, 'rb') as readfile:
            while True:
                try:
                    item = pkl.load(readfile)
                    self.items.append(item)  # Append to the list
                except EOFError:
                    break

        # Optionally convert list to a DataFrame here if needed
        # self.items = pd.DataFrame(self.items, columns=['data', 'label'])

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        odor, label = self.items[idx]  # Directly use list indexing

        # Directly convert to tensor
        odor = torch.tensor(odor, dtype=torch.float32)  # Use torch.tensor() directly
        
        if self.transform:
            odor = self.transform(odor)
        
        if self.target_transform:
            label = self.target_transform(label)
        
        return odor, label
