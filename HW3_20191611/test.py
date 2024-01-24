# data manipulation
import pandas as pd
import numpy as np

# data visualization
import matplotlib.pyplot as plt

# pytorch
import torch
import torchvision
from torchvision import datasets, transforms
import torch.nn.functional as F
from torch import nn
from torch.optim import Adam
from torch.utils.data import TensorDataset, DataLoader, random_split


# utils
import os
from tqdm import tqdm # 학습진행도 보여줌
tqdm.pandas()
from collections import Counter

# Hyper-parameters
in_channel = 1
num_classes = 10
batch_size = 64
max_pool_kernel = 2
out_node = 1
embedding_dim = 64
hidden_dim = 256
num_layers = 3
dropout = 0.1
learning_rate = 0.001
num_epochs = 20

class CRNN(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers):
        super(CRNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=32, kernel_size=3, padding=1, stride=2),
            nn.BatchNorm2d(num_features=32),  # out_channels이 그대로 들어감
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d(kernel_size=max_pool_kernel, stride=2)  # 2 * 2
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d(kernel_size=max_pool_kernel)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d(kernel_size=max_pool_kernel)
        )
        # fully - connected
        self.fc1 = nn.Linear(in_features=4608, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.dropout1 = nn.Dropout(p=0.25, inplace=False)
        self.fc3 = nn.Linear(in_features=64, out_features=num_classes)

        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = x.reshape(x.size(0), -1, embedding_dim)

        x, _ = self.lstm(x)

        x = x.reshape(x.size(0), -1)  # fully connected에 넣어주기 위해서 flatten 시켜주기

        x = self.fc1(x)
        # x = self.dropout1(x)
        x = F.relu(x)
        x = self.fc2(x)
        # x = self.dropout1(x)
        x = F.relu(x)
        x = self.fc3(x)

        # sigmoid 해도 되고 안해도 되고
        return x  # 10개의 출력 return

    def forward1(self, x):
        x = self.layer1(x)
        return x

    def forward2(self, x):
        x = self.layer2(x)
        return x

    def forward3(self, x):
        x = x.reshape(x.size(0), -1, embedding_dim)
        x, _ = self.lstm(x)
        return x

    def forward4(self, x):
        x = x.reshape(x.size(0), -1)
        x = self.fc1(x)
        return x

    def forward5(self, x):
        x = self.fc3(self.fc2(x))
        return x

# define training device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

test_data = torchvision.datasets.FashionMNIST(root='~/.pytorch/F_MNIST_data/',
                                       train=False,
                                       transform=transforms.ToTensor(),
                                       download=True)

test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                         batch_size=batch_size,
                                         shuffle=True)




model = CRNN(embedding_dim, hidden_dim, num_layers)
criterion = nn.CrossEntropyLoss().to(device)

tmp = torch.load('crnn_20191611.pth')
model.load_state_dict(tmp, strict = False)

# test loop
model.eval()

# metrics
test_loss = 0
test_acc = 0

all_target = []
all_predicted = []

testloop = tqdm(test_loader, leave=True, desc='Inference')
with torch.no_grad():
    for text, label in testloop:
        text, label = text.to(device), label.to(device)

        # forward pass
        out = model(text)

        # acc
        test_acc += (torch.argmax(out, 1) == label).sum().item()

        # loss
        loss = criterion(out, label)
        test_loss += loss.item()


    print("\n")
    print(f'Accuracy: {test_acc/len(test_data) * 100}% / Loss: {test_loss/len(test_loader):.4f}')

