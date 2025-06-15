import os
from datetime import datetime
from random import shuffle
import random
import numpy as np
import pandas as pd
import matplotlib.image as mpimg
import torch
import cv2
from torch.utils.data import DataLoader, random_split
from dataloader import CustomImageDataset, custom_transform
from imgaug import augmenters as iaa

from model import NeuralNetwork

def training_process():
    # Remove the redundant data loading here since we do it in train_model
    camera_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../camera'))
    csv_path = os.path.join(camera_dir, 'training_data_filtered_20250608_161947.csv')
    image_dir = 'training'
    
    print(f"CSV path: {csv_path}")
    print(f"Image directory: {os.path.join(camera_dir, image_dir)}")
    print(f"CSV exists: {os.path.exists(csv_path)}")
    print(f"Image dir exists: {os.path.exists(os.path.join(camera_dir, image_dir))}")
    
    data = CustomImageDataset(csv_path, camera_dir, image_dir, custom_transform)
    train_loader, test_loader = data_split(data)

    for i, data in enumerate(train_loader):
        print(train_loader.__len__())
        inputs, labels = data[0], data[1]
        print("labels: ", labels)
        break
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = NeuralNetwork().to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=.0001)
    train_model(model, loss_fn, optimizer, train_loader)
    test_model(model, loss_fn, test_loader)

def train_model(model, loss_fn, optimizer, train_loader):
    epoch_number = 0
    num_epochs = 4

    for epoch in range(num_epochs):
        print('EPOCH {}:'.format(epoch_number + 1))
        avg_loss = one_epoch(epoch_number, loss_fn, optimizer, model, train_loader)
        epoch_number += 1

def test_model(model, loss_fn, test_loader):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    running_vloss = 0
    model.eval()
    test_loss = 0
    total = 0
    correct = 0
    
    with torch.no_grad():
        for i, vdata in enumerate(test_loader):
            inputs, labels = vdata[0], vdata[1]
            for x in range(len(labels)):
                labels[x] += 1
            inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
            labels = torch.tensor(labels, dtype=torch.long).to(device)
            
            inputs = inputs.permute(0, 3, 1, 2)

            outputs = model(inputs)
            test_loss += loss_fn(outputs, labels).item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        running_vloss += test_loss
    avg_vloss = running_vloss / (i + 1)
    print('LOSS train {} valid {}'.format(avg_vloss, avg_vloss))
    latest_model_path = 'models/latest_model.pth'
    os.makedirs(os.path.dirname(latest_model_path), exist_ok=True)
    torch.save(model.state_dict(), latest_model_path)

def one_epoch(epoch_index, loss_fn, optimizer, model, training_data):
    running_loss = 0
    last_loss = 0
    counter = 0
    model.train(True)
    for i, data in enumerate(training_data):
        optimizer.zero_grad()
        counter += 1
        images, labels = data[0], data[1]
        for x in range(len(labels)):
            labels[x] += 1
        images = torch.tensor(images, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
        images = images.permute(0, 3, 1, 2)
        outputs = model(images)
        loss = loss_fn(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        if i % 10 == 9:
            last_loss = running_loss / 1000
            print('  batch {} loss: {}'.format(i + 1, last_loss))
            running_loss = 0

    return last_loss


def data_split(data):
    # split the data into training and testing sets
    train_size = int(0.6 * len(data))
    test_size = len(data) - train_size
    train_data, test_data = random_split(data, [train_size, test_size])
    # create data loaders
    train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
    #train_loader = custom_augment_data(train_loader)
    test_loader = DataLoader(test_data, batch_size=32, shuffle=False)
    return train_loader, test_loader

training_process()
