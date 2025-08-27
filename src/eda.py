import time
import pandas as pd
import torch # tensor computation, deep neural networks
import torch.nn as nn #foundation for building and training neural network models
import torch.optim as optim # optimization algorithms
import torchvision # ready to use datasets
import torchvision.transforms as transforms #manage matrix images
import numpy as np
import matplotlib.pyplot as plt #statical and dinamic visualization
import seaborn as sns #statistical data visualization
from sklearn.metrics import classification_report, confusion_matrix

#torch.manual_seed(17) #https://docs.pytorch.org/vision/0.9/transforms.html

# Sets
transform = transforms.Compose([transforms.ToTensor()]) #processing: transform the image in a multimedial matrix (=tensor)
train_set = torchvision.datasets.FashionMNIST(root='.data/', train=True, download=True, transform=transform) 
test_set = torchvision.datasets.FashionMNIST(root='.data/', train=False, download=True, transform=transform) 

# Loader
train_loader = torch.utils.data.DataLoader(train_set, batch_size=64, shuffle=True) #extract data, 64 times, shuffled
test_loader = torch.utils.data.DataLoader(test_set, batch_size=1000, shuffle=False) #extact data, 1k times, not shuffled

# EDA
print(f"train_set images available (data size): {len(train_set)}")
print(f"test_set images available (data size): {len(test_set)}")

# distribution of classes 
labels = train_set.targets.numpy()
unique, counts = np.unique(labels, return_counts=True) #Find the unique elements of an array.
class_dist = dict(zip(unique, counts, strict=True)) #strict: if one of the arguments is exhausted before the others, raise a ValueError.
print("Distribution of classes:")
for unique, count in class_dist.items():
    print(f"{unique}: {count}")

# Plot class distribution
classes_x_rapr = ["T-shirt/top", "Trouser", "Pullover", "Dress", "Coat", "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]
df = pd.DataFrame({"class": [classes_x_rapr[i] for i in class_dist.keys()], "count": counts})

plt.subplots(figsize=(8,6))
sns.barplot(x="count", y="class", data=df, palette=None, color="b")

plt.xlabel("Number of images")
plt.ylabel("Clothing class")
plt.title("Classes distribution")
plt.xlim(0, 7000) # max over 6k to show clearly that every class has 6k items(images) 
plt.show()

# Show sample images
examples = enumerate(train_loader)
batch_idx, (example_data, example_targets) = next(examples)

plt.figure(figsize=(10,4))
for i in range(10):
    plt.subplot(2,5,i+1) #frameon=False
    if i >= 5:
        plt.title(f"Item (cmap='gray'): {example_targets[i].item()}")
        plt.imshow(example_data[i][0], cmap='gray', interpolation='none')
    if i < 5:
        plt.title(f"Item (cmap='hot'): {example_targets[i].item()}")
        plt.imshow(example_data[i][0], cmap='hot', interpolation='none')
    plt.axis('off')
plt.tight_layout()
plt.show()