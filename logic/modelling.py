import pandas as pd
import numpy as np
import torch

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset



class NeuralNet(nn.Module):
    """Some Information about MyModule"""
    def __init__(self, input_size, hidden_size_layer1, hidden_size_layer2, output_size):
        
        super(NeuralNet, self).__init__()

        self.input_size = input_size
        self.hidden_size_layer1 = hidden_size_layer1
        self.hidden_size_layer2 = hidden_size_layer2
        self.output_size = output_size

        self.l1 = nn.Linear(input_size, hidden_size_layer1)
        self.relu = nn.ReLU()
        self.l2 = nn.Linear(hidden_size_layer1, hidden_size_layer2)
        self.l3 = nn.Linear(hidden_size_layer2, output_size)
        self.sigmoid = nn.Sigmoid()

        self.params = {}


    def set_parameters(self, n_epochs, learning_rate, optimizer, criterion):
       
       self.params['n_epochs'] = n_epochs
       self.params['learning_rate'] = learning_rate
       #optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)
       self.params['optimizer'] = optimizer
       self.params['criterion'] = criterion


    def forward(self, x):

        out1 = self.l1(x)
        out2 = self.relu(out1)
        out3 = self.l2(out2)
        out4 = self.relu(out3)
        out = self.l3(out4)
        out = self.sigmoid(out)
        return out
    
    
    def fit(self, params, x_data_loader):
        
        n_epochs = params['n_epochs']
        criterion = params['criterion']
        learning_rate = params['learning_rate']
        optimizer = torch.optim.SGD(self.parameters(), lr=learning_rate)

        losses = []
        self.epoch_loss = []

        for i in range (n_epochs):
            for j, (data, labels) in enumerate(x_data_loader):
            
                labels = labels.view(-1,1)

            
                data = data.to(torch.float32)
                predictions = self.forward(data)

                loss = criterion(predictions, labels)
                losses.append(loss.item()) 

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            res = np.mean(np.array(losses))
            self.epoch_loss.append(res)
            if (i+1) % 10 == 0:
                print(f'Epoch [{i+1}/{n_epochs}], Loss: {self.epoch_loss:.4f}')

    
    def predict(self, x_test_loader):
        
        with torch.no_grad():
            
            n_samples = 0
            self.n_correct = 0
            losses = []

            for i, (sample, target) in enumerate(x_test_loader):
                target = target.view(-1,1)

                data = data.to(torch.float32)
                outputs = self.forward(sample)
                loss = self.params['criterion'](outputs, target)
                losses.append(loss.item())
            

                n_samples += sample.size(0)

                prediction = torch.where(outputs < 0.5, 0, 1)
                prediction = prediction.reshape(-1,1)

                self.n_correct += (prediction == target).sum().item()
            
            self.test_loss = np.mean(losses)



class FraudDetection(Dataset):

  def __init__(self, x, y):

    self.x_data = torch.from_numpy(x)
    self.y_data = torch.from_numpy(y.to_numpy())

    self.n_samples, self.n_features = x.shape

  def __getitem__(self, index):
    sample = self.x_data[index].float(), self.y_data[index].float()
    return sample
  
  def __len__(self):
    return self.n_samples
  


   