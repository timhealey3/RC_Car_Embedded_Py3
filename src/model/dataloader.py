import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
import numpy as np
import cv2
import os

def img_preproceesing(image):
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    image = np.flipud(image)
    return image

def custom_transform(image):
    image = img_preproceesing(image)
    return image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, folder_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.img_labels = pd.read_csv(os.path.join(self.img_dir, annotations_file))
        self.folder_dir = folder_dir
        self.transform = transform
    
    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.folder_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

c = CustomImageDataset('training_data.csv', '../camera', '/training', custom_transform)
print(c.__len__())
c.__getitem__(1)