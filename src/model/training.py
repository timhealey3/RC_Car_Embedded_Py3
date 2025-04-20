from datetime import datetime
from random import shuffle
import random
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import torch
import cv2
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from dataloader import CustomImageDataset, custom_transform
from imgaug import augmenters as iaa

from model import NeuralNetwork

def training_process():
    data = CustomImageDataset('training_data.csv', './camera', 'training_converted', custom_transform)
    # balanced_data = balance_data(data)
    train_loader, test_loader = data_split(data)
    #X_train_gen, y_train_gen = next(batch_generator(train_loader, 32, True))
    #X_test_gen, y_test_gen = next(batch_generator(test_loader, 32, False))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = NeuralNetwork()
    model = NeuralNetwork().to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_model(model, loss_fn, optimizer)

def train_model(model, loss_fn, optimizer):
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    writer = SummaryWriter('runs/steering_trainer_{}'.format(timestamp))
    epoch_number = 0
    num_epochs = 10
    best_vloss = 1_000_000.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = CustomImageDataset('training_data.csv', './camera', 'training_converted', custom_transform)
    train_loader, test_loader = data_split(data)

    for epoch in range(num_epochs):
        print('EPOCH {}:'.format(epoch_number + 1))

        model.train(True)
        avg_loss = one_epoch(epoch_number, writer, loss_fn, optimizer, model, train_loader)

        # set model to evaluation mode, disable dropout and using population stats for batch normalization
        running_vloss = 0.0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.eval()
        # disable gradient tracking and run on test validation dataset
        with torch.no_grad():
            for i, vdata in enumerate(test_loader):
                vinputs, vlabels = vdata
                for x in range(len(vlabels)):
                    vlabels[x] += 1
                vinputs = torch.tensor(vinputs, dtype=torch.float32).to(device)
                vlabels = torch.tensor(vlabels, dtype=torch.long).to(device)
                if vinputs.dim() == 3:
                    vinputs = vinputs.unsqueeze(0)
                vinputs = vinputs.permute(0, 3, 1, 2)
                voutputs = model(vinputs)
                vloss = loss_fn(voutputs, vlabels)
                running_vloss += vloss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
        # Log the running loss averaged per batch for both training and validation
        writer.add_scalars('Training vs. Validation Loss',
                        { 'Training' : avg_loss, 'Validation' : avg_vloss },
                        epoch_number + 1)
        writer.flush()

        # Track best performance, and save the model's state
        if avg_vloss < best_vloss:
            best_vloss = avg_vloss
            model_path = 'model_{}_{}'.format(timestamp, epoch_number)
            torch.save(model.state_dict(), model_path)

        epoch_number += 1

def one_epoch(epoch_index, tb_writer, loss_fn, optimizer, model, training_data):
    running_loss = 0
    last_loss = 0
    counter = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    for i, data in enumerate(training_data):
        counter += 1
        inputs, labels = data[0], data[1]
        # print("labels: ", labels)
        for x in range(len(labels)):
            labels[x] += 1
        # custom_augment_data(inputs, labels)
        inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
        labels = torch.tensor(labels, dtype=torch.long).to(device)
        
        inputs = inputs.permute(0, 3, 1, 2)

        optimizer.zero_grad()
        # prediction
        outputs = model(inputs)
        # compute loss and gradient
        loss = loss_fn(outputs, labels)
        loss.backward()

        optimizer.step()

        # gather data and report every 1000 batches
        running_loss += loss.item()
        if i % 10 == 9:
            last_loss = running_loss / 1000
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            tb_x = epoch_index * len(training_data) + i + 1
            tb_writer.add_scalar('Loss/train', last_loss, tb_x)
            running_loss = 0

    return last_loss

"""
Splitting the data set into training and testing sets.
The training set is 80% of the data, and the testing set is 20% of the data.
"""
def data_split(data):
    # split the data into training and testing sets
    train_size = int(0.8 * len(data))
    test_size = len(data) - train_size
    train_data, test_data = random_split(data, [train_size, test_size])
    # create data loaders
    train_loader = DataLoader(train_data, batch_size=16, shuffle=True)
    #train_loader = custom_augment_data(train_loader)
    test_loader = DataLoader(test_data, batch_size=16, shuffle=False)
    return train_loader, test_loader


def custom_augment_data(image, steeering_angle):
    augmented_image = []
    augmented_steering_angle = []
    for i in range(len(image)):
        img, steering = augment_data(image[i], steeering_angle[i])
        augmented_image.append(img)
        custom_transform(image)
        augmented_steering_angle.append(steering)
    return augmented_image, augmented_steering_angle

"""
custom batch generator
"""
def batch_generator(data, batch_size, isTrainingInd):
    while True:
        batch_img = []
        batch_steering  = []
        for i in range(batch_size):
            random_index = random.randint(0, len(data) - 1)
            if isTrainingInd:
                img, steering_angle = augment_data(data.dataset.__getitem__(random_index)[0], data.dataset.__getitem__(random_index)[1])
            else:
                img = data.dataset.__getitem__(random_index)[0]
                steering_angle = data.dataset.__getitem__(random_index)[1]
            batch_img.append(img)
            batch_steering.append(steering_angle)
        yield (np.asarray(batch_img), np.asarray(batch_steering))

"""
Augmenting the data by randomly changing data.
"""
def augment_data(image, steering_angle):
    if not isinstance(image, np.ndarray):
        image = np.array(image)
    # Scale to 0-255 and convert to uint8
    if image.dtype != np.uint8:
        image = (image * 255).astype(np.uint8) 
    if np.random.rand() < 50:
        image = pan(image)
    if np.random.rand() < 50:
        image = zoom(image)
    if np.random.rand() < 50:
        image = img_random_brightness(image)
    if np.random.rand() < 50:
        image, steering_angle = flipping(image, steering_angle)

    # Convert back to float32 and normalize to 0-1 if needed
    return image, steering_angle

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


def balance_data(data):
    num_bins = 25
    samples_per_bin = 400
    hist, bins = np.histogram(data['steering_angle'], num_bins)
    remove_list = []
    for j in range(num_bins):
        list_ = []
        for i in range(len(data['steering_angle'])):
            if data['steering_angle'][i] >= bins[j] and data['steering_angle'][i] <= bins[j+1]:
                list_.append(i)
            list_ = shuffle(list_)
            list_ = list_[samples_per_bin:]
            remove_list.extend(list_)
    

training_process()
