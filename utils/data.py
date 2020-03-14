import numpy as np
from PIL import Image
import os
import pandas as pd

from torch.utils.data import Dataset

class ImportDataset(Dataset):
    def __init__(self, csv, transforms):
        self.transforms = transforms

        self.data = pd.read_csv(csv)

        self.imgs = self.data['img']
        self.keys = self.data.drop(['img'], axis=1)      

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        img = self.transforms(Image.open(self.imgs[idx]))

        keys = self.keys.values[idx]

        return img, keys
