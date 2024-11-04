import torch
from PIL import Image
from torch.utils.data import Dataset
import pandas as pd
import os

class WikiArtDataset(Dataset):
    def __init__(self, annotation_file, transform=None):
        self.annotations = pd.read_csv(annotation_file, sep=" ")
        self.transform = transform

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_path, label = self.annotations.iloc[index]
        # load the RGB image
        image = Image.open(img_path).convert("RGB")
        label = torch.tensor(int(label))
        if self.transform:
            image = self.transform(image)
        return (image, label)
