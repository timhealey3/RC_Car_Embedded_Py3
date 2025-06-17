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
    
    # Get indices to remove for each class
    indices_to_remove = []
    class_labels = [-1, 0, 1]  # Explicitly define the class labels we expect
    
    for class_label in class_labels:
        # Find all indices for this class
        class_indices = data[data.iloc[:, 2] == class_label].index.tolist()
        
        # If we have more than samples_per_bin, remove excess
        if len(class_indices) > samples_per_bin:
            # Randomly shuffle and keep only samples_per_bin
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
    np_img = np_img[height // 3 + 45:-315, :, :]
    np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2RGB)
    np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2HSV)
    np_img = cv2.GaussianBlur(np_img, (3, 3), 0)
    np_img = cv2.resize(np_img, (200, 66))
    np_img = np_img / 255
    return label, np_img

"""
Augmenting the data by randomly changing data.
"""
def augment_data(image):
    # .2 seems to work pretty well, but if 
    if np.random.rand() < 0.3:
        image = pan(image)
    if np.random.rand() < 0.3:
        image = zoom(image)
    if np.random.rand() < 0.3:
        image = img_random_brightness(image)
    if np.random.rand() < 0.3:
        image = shear(image)

    return image

"""
Create shear augmenter with 1 to 1.5 shear range
"""
def shear(img):
    # Horizontal shear
    shear_x = iaa.ShearX(shear=(-1.2, 1.2))
    # Vertical shear  
    shear_y = iaa.ShearY(shear=(-1.2, 1.2))
    # Combine both (randomly apply one or both)
    shear_augmenter = iaa.OneOf([
        shear_x,
        shear_y,
        iaa.Sequential([shear_x, shear_y])
    ])
    
    return shear_augmenter.augment_image(img)

"""
zooms into the image up to 30%
"""
def zoom(img):
    zoom = iaa.Affine(scale=(1, 1.2))
    image = zoom.augment_image(img)
    return image

"""
pan the image up to 10% on both x and y axis
"""
def pan(img):
    pan = iaa.Affine(translate_percent={"x": (-0.1, 0.1), "y": (-0.1, 0.1)})
    image = pan.augment_image(img)
    return image

"""
Change the brightness of the image up to 20%
"""
def img_random_brightness(img):
    brightness = iaa.Multiply((0.8, 1.2))
    image = brightness.augment_image(img)
    return image

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, folder_dir, augment, target_transform=None):
        self.img_dir = img_dir
        self.img_labels = pd.read_csv(os.path.join(self.img_dir, annotations_file))
        self.folder_dir = folder_dir
        self.augment = augment
        self.target_transform = target_transform
        self.img_labels = balance_data(self.img_labels)

    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.folder_dir, self.img_labels.iloc[idx, 0])
        image = cv2.imread(img_path)
        label = self.img_labels.iloc[idx, 2]
        try:
            if self.augment:
                image = augment_data(image)
            if self.target_transform:
                label = self.target_transform(label)
        except Exception as e:
            print(f"Error during transform: {str(e)}")
            return None, None
        label, image = custom_transform(image, label)
        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long) + 1
        return image, label
