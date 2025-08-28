import time
import os
import torch # tensor computation, deep neural networks
import torch.nn as nn #foundation for building and training neural network models
import torch.optim as optim # optimization algorithms
import torchvision # ready to use datasets
import torchvision.transforms as transforms #manage matrix images
from tqdm import tqdm #Decorate an iterable object - prints a dynamically updating progressbar every time a value is requested.
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
            nn.Linear(64 * 5 * 5, 64), # mapping of 1600 input in 64 neurons
            nn.ReLU(),
            nn.Linear(64, 10) #last output's layer: 1 neuron each class (10)
        ) # result: logits
    
    def forward(self, x): #definition 'forward' function 
        x = self.conv_layers(x) # convolution -> ReLU -> MaxPool2d -> feature's tensore
        x = self.fc_layers(x) # from 4D to 2D tensor -> ReLU -> classification's logits
        return x # return logits
    
def train_FashionMNIST(epochs: int, save_path: str= "FashionMNIST_model.pth"):
    transform = transforms.Compose([transforms.ToTensor()]) #processing: from image to multidimentionals matrix
    train_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True) # extraction of 64 shuffled data per time
    
    model = SimpleCNN()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0023835315571002513) #optimization algorithms
    #lr choosen based on the best performance with optuna_search.py
    #Trial 18 finished with value: 0.8413 and parameters: {'lr': 0.0023835315571002513, 'dropout': 0.168692179463072}.
    
    # Train model on CPU
    start = time.time()
    for epoch in tqdm(range(epochs)):
        model.train() # model in training mode
        running_loss = 0.0
        for images, labels in train_loader:
            optimizer.zero_grad() #instead of setting to zero, set the grads to None. This will in general have lower memory footprint
            outputs = model(images) # it gives the images to the model
            loss = criterion(outputs, labels) #prediction error (the model will learn from his own mistakes)
            loss.backward() #learning on his prediction mistakes
            optimizer.step() # = continue, go a the next step.
            running_loss += loss.item()
        print(f"Epoch [{epoch + 1}/5], Loss: {running_loss/len(train_loader):.4f}")
    print(f"Training time: {time.time() - start:.2f} seconds")
    
    torch.save(model.state_dict(), save_path)
    
def evaluate_FashionMNIST(model_path: str = 'FashionMNIST_model.pth'):
    transform = transforms.Compose([transforms.ToTensor()]) #processing: from image to multidimentionals matrix
    test_set = torchvision.datasets.FashionMNIST(root='./data', train=True, download=True, transform=transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=1000, shuffle=False) # extraction of 1000 NOT shuffled data per time
    
    assert os.path.exists(model_path), f"Model file {model_path} does not exist"
    
    model = SimpleCNN()
    model.load_state_dict(torch.load(model_path))
    
    # Evaluate the model
    model.eval() # remove the model from training mode and it sets it in analysis mode to check the performance
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in test_loader:
            outputs = model(images) # it gives the images to the model
            _, predicted = torch.max(outputs.data, 1) # index corrisponding at the predicted class
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())
            
    accuracy = correct / total # accuracy's percentage 
    print(f"\nTest accuracy: {accuracy:.4f}")
        
    # Classification report
    print("\nClassification Report")
    print(classification_report(all_labels, all_preds))
    
    # Plot confusion matrix
    classes_x_rapr = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
    conf_matrix = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8, 6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', xticklabels=classes_x_rapr, yticklabels=classes_x_rapr)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    plt.savefig("confusion_matrix.png")
    
if __name__ == "__main__":
    train_FashionMNIST(5)
    evaluate_FashionMNIST()