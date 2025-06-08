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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    model = NeuralNetwork().to(device)
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    train_model(model, loss_fn, optimizer)

def train_model(model, loss_fn, optimizer):
    epoch_number = 0
    num_epochs = 10
    best_vloss = 1_000_000.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    camera_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '../camera'))
    csv_path = os.path.join(camera_dir, 'training_data_filtered_20250608_161947.csv')
    image_dir = 'training'  # Changed from 'training_converted' to 'training'
    
    print(f"CSV path: {csv_path}")
    print(f"Image directory: {os.path.join(camera_dir, image_dir)}")
    print(f"CSV exists: {os.path.exists(csv_path)}")
    print(f"Image dir exists: {os.path.exists(os.path.join(camera_dir, image_dir))}")
    
    data = CustomImageDataset('training_data_filtered_20250608_161947.csv', camera_dir, image_dir, custom_transform)
    train_loader, test_loader = data_split(data)

    os.makedirs('models', exist_ok=True)
    
    for epoch in range(num_epochs):
        print('EPOCH {}:'.format(epoch_number + 1))
        avg_loss = one_epoch(epoch_number, loss_fn, optimizer, model, train_loader)
        running_vloss = 0
        model.eval()
        with torch.no_grad():
            for i, vdata in enumerate(test_loader):
                inputs, labels = vdata[0], vdata[1]
                for x in range(len(labels)):
                    labels[x] += 1
                inputs = torch.tensor(inputs, dtype=torch.float32).to(device)
                labels = torch.tensor(labels, dtype=torch.long).to(device)
                
                inputs = inputs.permute(0, 3, 1, 2)

                outputs = model(inputs)
                loss = loss_fn(outputs, labels)
            running_vloss += loss

        avg_vloss = running_vloss / (i + 1)
        print('LOSS train {} valid {}'.format(avg_loss, avg_vloss))
        latest_model_path = 'models/latest_model.pth'
        torch.save(model.state_dict(), latest_model_path)
        epoch_number += 1

def one_epoch(epoch_index, loss_fn, optimizer, model, training_data):
    running_loss = 0
    last_loss = 0
    counter = 0
    model.train(True)
    for i, data in enumerate(training_data):
        counter += 1
        inputs, labels = data[0], data[1]
        # print("labels: ", labels)
        for x in range(len(labels)):
            labels[x] += 1
        # custom_augment_data(inputs, labels)
        inputs = torch.tensor(inputs, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)
        
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
            running_loss = 0

    return last_loss


def data_split(data):
    # split the data into training and testing sets
    train_size = int(0.6 * len(data))
    test_size = len(data) - train_size
    train_data, test_data = random_split(data, [train_size, test_size])
    # create data loaders
    train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
    #train_loader = custom_augment_data(train_loader)
    test_loader = DataLoader(test_data, batch_size=8, shuffle=False)
    return train_loader, test_loader

training_process()
