import os
import numpy as np
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import Dataset
from tqdm import tqdm
import glob

class CurlNoise(Dataset):
    def __init__(self, root, norm=True, subsample_size=None, transform=None, **kwargs):
        print(root)
        if not os.path.isdir(root):
            raise ValueError(f"The specified root: {root} does not exist")
        self.root = root
        self.transform = transform
        self.norm = norm

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
        
        return torch.from_numpy(img).permute(0, 1).float()[None,:,:]

    def __len__(self):
        return len(self.images)