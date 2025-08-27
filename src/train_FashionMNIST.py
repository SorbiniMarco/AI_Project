import time
import os
import torch # tensor computation, deep neural networks
import torch.nn as nn #foundation for building and training neural network models
import torch.optim as optim # optimization algorithms
import torchvision # ready to use datasets
import torchvision.transforms as transforms #manage matrix images
import numpy as np
import matplotlib.pyplot as plt #statical and dinamic visualization
import seaborn as sns #statistical data visualization
from sklearn.metrics import classification_report, confusion_matrix

# Neural network model creation
class SimpleCNN(nn.Module): #Base class for all neural network modules.
    def __init__(self):
        super(SimpleCNN, self).__init__()
        
        # Feature extraction of every image
        self.conv_layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3), #input, output, kernel_size: capture local pattern
            nn.ReLU(), #rectified Linear Unit: introduce non-linearity
            nn.MaxPool2d(2), #reduce the tensor dimension (height, width)
            
            nn.Conv2d(32, 64, kernel_size=3), # input (=first convulation's output), output, kernel_size
            nn.ReLU(), #rectified Linear Unit: introduce non-linearity (again)
            nn.MaxPool2d(2) #reduce the tensor dimension (height, width)
        )
        
        # Classification
        self.fc_layers = nn.Sequential(
            nn.Flatten(), # from 4D tensor to a 2D tensor
            nn.Linear(64 * 5, * 5, 64), # mapping of 1600 input in 64 neurons
            nn.ReLU(),
            nn.Linear(64, 10) #last output's layer: 1 neuron each class (10)
        ) # result: logits
    
    def forward(self, x): #definition 'forward' function 
        x = self.conv_layers(x) # convolution -> ReLU -> MaxPool2d -> feature's tensore
        x = self.fc_layers(x) # from 4D to 2D tensor -> ReLU -> classification's logits
        return x # return logits