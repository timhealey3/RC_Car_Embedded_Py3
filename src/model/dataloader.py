from random import shuffle
import pandas as pd
import torch
from torchvision.io import read_image
from torch.utils.data import Dataset
import numpy as np
import cv2
import os
from PIL import Image
from imgaug import augmenters as iaa


def balance_data(data):
    num_bins = 3 
    samples_per_bin = 1400
    
    print("Class distribution before balancing:")
    print(data.iloc[:, 2].value_counts())
    
    print("Balancing data...")
    
    indices_to_remove = []
    class_labels = [-1, 0, 1]
    
    for class_label in class_labels:
        class_indices = data[data.iloc[:, 2] == class_label].index.tolist()
        
        if len(class_indices) > samples_per_bin:
            shuffle(class_indices)
            excess_indices = class_indices[samples_per_bin:]  # Remove excess
            indices_to_remove.extend(excess_indices)
    
    # Remove excess samples
    data_balanced = data.drop(indices_to_remove).reset_index(drop=True)
    
    print("Class distribution after balancing:")
    print(data_balanced.iloc[:, 2].value_counts())
    
    return data_balanced    

def custom_transform(image, label):
    np_img = np.array(image)
    height = np_img.shape[0]
    #image = image[height // 3 + 60:-350, :, :] # Crop top and bottom
    image = image[height // 3 + 50: -300, :, :]
    np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
    #np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2HSV)
    np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2GRAY)
    np_img = cv2.GaussianBlur(np_img, (3, 3), 0)
    np_img = cv2.resize(np_img, (200, 66))
    np_img = np_img / 255
    label = label + 1
    return np_img, label

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, folder_dir, target_transform=None):
        self.img_dir = img_dir
        self.img_labels = pd.read_csv(os.path.join(self.img_dir, annotations_file))
        self.folder_dir = folder_dir
        self.target_transform = target_transform
        self.img_labels = balance_data(self.img_labels)

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.folder_dir, self.img_labels.iloc[idx, 0])
        image = cv2.imread(img_path)
        label = self.img_labels.iloc[idx, 2]
        image, label = custom_transform(image, label)
        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long)
        return image, label
