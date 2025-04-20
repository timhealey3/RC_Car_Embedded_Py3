from random import shuffle
import pandas as pd
from torchvision.io import read_image
from torch.utils.data import Dataset
import numpy as np
import cv2
import os
from PIL import Image
from imgaug import augmenters as iaa

def balance_data(data):
    num_bins = 25
    samples_per_bin = 400
    hist, bins = np.histogram(data['steering_angle'], num_bins)
    print(hist)
    print(bins)
    remove_list = []
    for j in range(num_bins):
        list_ = []
        for i in range(len(data['steering_angle'])):
            if data['steering_angle'][i] >= bins[j] and data['steering_angle'][i] <= bins[j + 1]:
                list_.append(i)
            shuffle(list_)
            list_ = list_[samples_per_bin:] 
            remove_list.extend(list_)
    data.drop(data.index[remove_list], inplace=True)
    return data

def custom_transform(image, label):
    np_img = np.array(image)
    steering_angle, np_img = augment_data(image, label)
    height = np_img.shape[0]
    image = np_img[height // 3:, :, :]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
    image = cv2.GaussianBlur(image, (3, 3), 0)
    image = cv2.resize(image, (200, 66))
    image = image / 255
    return steering_angle, image

"""
Augmenting the data by randomly changing data.
"""
def augment_data(image, steering_angle):
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    # Scale to 0-255 and convert to uint8
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8) 
    if np.random.rand() < 0.5:
        image = pan(image)
    if np.random.rand() < 0.5:
        image = zoom(image)
    if np.random.rand() < 0.5:
        image = img_random_brightness(image)
    if np.random.rand() < 0.5:
        image, steering_angle = flipping(image, steering_angle)

    return steering_angle, image

"""
zooms into the image up to 30%
"""
def zoom(img):
    zoom = iaa.Affine(scale=(1, 1.3))
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
    brightness = iaa.Multiply((0.2, 1.2))
    image = brightness.augment_image(img)
    return image

"""
Flip the image and change the steering angle to the opposite direction
"""
def flipping(img, steering_angle):
    # flip some images and steering angle
    image = cv2.flip(img, 1)
    steering_angle = -steering_angle
    return image, steering_angle

class CustomImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, folder_dir, transform=None, target_transform=None):
        self.img_dir = img_dir
        self.img_labels = pd.read_csv(os.path.join(self.img_dir, annotations_file))
        self.folder_dir = folder_dir
        self.transform = transform
        self.target_transform = target_transform
        # self.img_labels = balance_data(self.img_labels)

    def __len__(self):
        return len(self.img_labels)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.folder_dir, self.img_labels.iloc[idx, 0])
        #image = Image.open(img_path)
        image = cv2.imread(img_path)
        label = self.img_labels.iloc[idx, 2]
        if self.transform:
            label, image = self.transform(image, label)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label

# c = CustomImageDataset('training_data.csv', '../camera', 'training_converted', custom_transform)
# print(c.__len__())
# image, label = c.__getitem__(5)
# image = Image.fromarray((image * 255).astype(np.uint8))  # Scale back to 0-255 if normalized
# image.show()