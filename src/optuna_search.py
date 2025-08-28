import optuna #hyperparameters optimization
import torch
from torchvision import datasets, transforms

def objective(trial):
    # Hyperparameters to tune
    lr = trial.suggest_loguniform('lr', 1e-5, 1e-1)
    dropout = trial.suggest_float('dropout', 0.1, 0.5)
    
    # Model definition
    model = torch.nn.Sequential(
        torch.nn.Flatten(), #a 28x28 image become a vector of 784 values
        torch.nn.Linear(28 * 28, 128), #first layer fully connected
        torch.nn.ReLU(), #introduce non-linearity
        torch.nn.Dropout(dropout), #randomly turn off a neuron
        torch.nn.Linear(128, 10)
    )
    
    # Training loop (one epoch for demo)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = torch.nn.CrossEntropyLoss()
    
    train_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('.', train=True, download=True, transform=transforms.ToTensor()),
        batch_size=64, shuffle=True
    )
    
    model.train() #model in training mode
    for batch in train_loader:
        data, target = batch
        optimizer.zero_grad() #we do not wanna consider the previous gradients
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        
    # Validation accuracy as objective
    val_loader = torch.utils.data.DataLoader(
        datasets.FashionMNIST('.', train=False, download=True, transform=transforms.ToTensor()),
        batch_size=1000, shuffle=False
    )
    
    correct = 0
    with torch.no_grad():
        for data, target in val_loader:
            preds = model(data).argmax(dim=1) #model(data) produce logits, .argmax(dim=1) take the highest probability class (less mistakes)
            correct += (preds == target).sum().item()
    accuracy = correct / len(val_loader.dataset) # %
    return accuracy

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=20) # number of trial