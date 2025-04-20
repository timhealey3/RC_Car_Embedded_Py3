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

training_process()
