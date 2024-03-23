import torch
from torch.utils.data import Dataset
import pandas as pd
import pickle as pkl
  
class FixedUniformDataset(Dataset):
  def __init__(self, split_file, transform=None, target_transform=None):
    self.transform = transform
    self.target_transform = target_transform
    data_label_dict = {}

    self.items = pd.DataFrame(columns=['data', 'label'])
    with open(split_file, 'rb') as readfile:
      while True:
        try:
          self.items.loc[len(self.items.index)] = pkl.load(readfile)
        except EOFError:
            break
  

  def __len__(self):
    return len(self.items)


  def __getitem__(self, idx):
    odor, label = self.items.iloc[idx, :]

    odor = torch.Tensor(odor)
    if self.transform:
        odor = self.transform(odor)
        # print(image.shape)
    if self.target_transform:
        label = self.target_transform(label)
    
    return odor, label
