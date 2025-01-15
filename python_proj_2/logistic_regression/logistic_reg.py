import torch
import torchvision
import torch.nn as nn
import matplotlib.pyplot as plt
import torchvision.transforms as tt
import torch.utils as utils

input_size = 28 * 28
num_classes = 10
num_epochs = 5
batch_size = 100
learning_rate = 0.001

# train_data = torchvision.datasets.MNIST('./data',download=True)
# test_data = torchvision.datasets.MNIST('data',train=False)

train_data = torchvision.datasets.MNIST('data', train=True, transform=tt.ToTensor())
test_data = torchvision.datasets.MNIST('data', train=False, transform=tt.ToTensor())

train_dataLoader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_dataLoader = torch.utils.data.DataLoader(test_data, batch_size=batch_size, shuffle=False)


# test
accurate = 0
total = 0
for images, labels in test_dataLoader:
    images = torch.autograd.Variable(images.view(-1, input_size))
    output = model(images)
    _, predicted = torch.max(output.data, 1)
    # total labels
    total += labels.size(0)

    # Total correct predictions
    accurate += (predicted == labels).sum()
    accuracy_score = 100 * accurate / total
