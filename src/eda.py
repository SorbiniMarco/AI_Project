from data.dataset import get_dataset

dataset = get_dataset()

print(dataset['train'][0])

train_set = dataset()