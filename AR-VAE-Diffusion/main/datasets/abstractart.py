import os
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
import glob
import pandas as pd

class AbstractArt(Dataset):
    def __init__(self, root, attributes="structural_complexity,color_diversity", attribute_data=None, norm=True, subsample_size=None, transform=None, **kwargs):
        if not os.path.isdir(root):
            raise ValueError(f"The specified root: {root} does not exist")
        self.root = root
        self.transform = transform
        self.norm = norm

        if attributes is not None and attribute_data is not None:
            self.attributes = attributes.split(",")
            self.attribute_data = pd.read_csv(attribute_data)
            self.attr_vals = np.array([[row[1][attr] for row in self.attribute_data.iterrows()] for attr in self.attributes])
        else:
            self.attributes = None
            self.attribute_data = None
        
            
        self.images = []
        for img in tqdm(glob.iglob(root + "/*.png")):
            self.images.append(img)

        # Subsample the dataset (if enabled)
        if subsample_size is not None:
            self.images = self.images[:subsample_size]

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)
        if self.norm:
            img = (np.asarray(img).astype(np.float) / 127.5) - 1.0
        else:
            img = np.asarray(img).astype(np.float) / 255.0
        
        if self.attributes is not None and self.attribute_data is not None:
            return torch.from_numpy(img).permute(2, 0, 1).float(), torch.from_numpy(self.attr_vals[:,idx]).float()
        else:
            return torch.from_numpy(img).permute(2, 0, 1).float()
    def __len__(self):
        return len(self.images)