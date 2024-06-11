import os
from torch.utils.data import Dataset
from preprocess import preprocess_image
import cv2
import numpy as np

class PersonDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.images = []
        self.labels = []
        for i, folder in enumerate(['fanruoruo', 'liyunrui', 'silili', 'yelinger', 'yuanmeng']):
            folder_path = os.path.join(self.root_dir, folder)
            for filename in os.listdir(folder_path):
                img_path = os.path.join(folder_path, filename)
                self.images.append(preprocess_image(img_path))
                self.labels.append(i)

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        return self.images[idx], self.labels[idx]