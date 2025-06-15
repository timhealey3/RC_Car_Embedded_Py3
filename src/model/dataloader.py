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
    samples_per_bin = 400
    
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
    np_img, steering_angle = augment_data(np_img, label)
    height = np_img.shape[0]
    np_img = np_img[height // 3:, :, :]
    # np_img = cv2.cvtColor(np_img, cv2.COLOR_RGB2YUV)
    np_img = cv2.cvtColor(np_img, cv2.COLOR_BGR2HLS)
    np_img = cv2.resize(np_img, (200, 66))
    np_img = np_img / 255
    return steering_angle, np_img

"""
Augmenting the data by randomly changing data.
"""
def augment_data(image, steering_angle):
    if np.random.rand() < 0.9:
        image = pan(image)
    if np.random.rand() < 0.9:
        image = zoom(image)
    if np.random.rand() < 0.9:
        image = img_random_brightness(image)
    if np.random.rand() < 0.25:
        image, steering_angle = flipping(image, steering_angle)

    return image, steering_angle

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
    pan = iaa.Affine(translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)})
    image = pan.augment_image(img)
    return image

"""
Change the brightness of the image up to 20%
"""
def img_random_brightness(img):
    brightness = iaa.Multiply((0.8, 1.2))
    image = brightness.augment_image(img)
    return image

"""
Flip the image and change the steering angle to the opposite direction
"""
def flipping(img, steering_angle):
    image = cv2.flip(img, 1)
    if steering_angle == 1:
        steering_angle = -1
    elif steering_angle == -1:
        steering_angle = 1
    return image, steering_angle

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, folder_dir, transform=True, target_transform=None):
        self.img_dir = img_dir
        self.img_labels = pd.read_csv(os.path.join(self.img_dir, annotations_file))
        self.folder_dir = folder_dir
        self.transform = transform
        self.target_transform = target_transform
        self.img_labels = balance_data(self.img_labels)

    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.folder_dir, self.img_labels.iloc[idx, 0])
        image = cv2.imread(img_path)
        label = self.img_labels.iloc[idx, 2]
        try:
            if self.transform:
                label, image = custom_transform(image, label)
            if self.target_transform:
                label = self.target_transform(label)
        except Exception as e:
            print(f"Error during transform: {str(e)}")
            print(img_path)
            return None, None
        image = torch.tensor(image, dtype=torch.float32)
        label = torch.tensor(label, dtype=torch.long) + 1
        return image, label

# c = CustomImageDataset('training_data.csv', '../camera', 'training_converted', custom_transform)
# print(c.__len__())
# image, label = c.__getitem__(5)
# image = Image.fromarray((image * 255).astype(np.uint8))  # Scale back to 0-255 if normalized
# image.show()