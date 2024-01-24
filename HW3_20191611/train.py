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
from re import X

# define training device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

# Adjust Dataset classes and channels
num_classes = 10 # last output
in_channel = 1

# Hyper-parameters
batch_size = 64
max_pool_kernel = 2
out_node = 1
embedding_dim = 64
hidden_dim = 256
num_layers = 3
dropout = 0.1
learning_rate = 0.001
num_epochs = 10

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


train_data = torchvision.datasets.FashionMNIST(root='~/.pytorch/F_MNIST_data/',
                                        train=True,
                                        transform=transforms.ToTensor(),
                                        download=True)
test_data = torchvision.datasets.FashionMNIST(root='~/.pytorch/F_MNIST_data/',
                                       train=False,
                                       transform=transforms.ToTensor(),
                                       download=True)

train_size = 0.7
val_size = 0.5

train_size = int(0.7 * len(train_data))
valid_size = len(train_data) - train_size
train_dataset, valid_dataset = random_split(train_data, [train_size, valid_size])
print(len(train_dataset), len(valid_dataset))

train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset= valid_dataset,
                                        batch_size = batch_size,
                                        shuffle=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                         batch_size=batch_size,
                                         shuffle=True)

# model initialization
model = CRNN(embedding_dim, hidden_dim, num_layers)

criterion = nn.CrossEntropyLoss().to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

num_epochs = 10
# Train Loop
tr_loss = []
tr_acc = []
v_loss = []
v_acc = []

#best_valid_loss = torch.inf
best_valid_loss = 100
best_epoch = 0
model = model.to(device)
epochloop = tqdm(range(num_epochs), position=0, desc='Training', leave=True) ## 학습을 하면서 progress 확인

for epoch in epochloop:
    model.train()
    train_loss = 0
    train_acc = 0
    total_acc = 0
    total_samples = 0

    ## Train
    for idx, (image, label) in enumerate(train_loader):
        epochloop.set_postfix_str(f'Training batch {idx}/{len(train_loader)}') # visualize
        image, label = image.to(device), label.to(device)

        out = model(image)

        # acc
        train_acc += (torch.argmax(out, 1) == label).sum().item()
        total_samples += len(label)
        #train_acc = total_acc / total_samples

        # loss
        optimizer.zero_grad()
        loss = criterion(out, label)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()

  ## Validation
  ## epoch마다 validation을 진행해서 epoch_loss를 계산해봄
    model.eval()
    val_loss = 0
    val_samples = 0
    valid_acc = 0
    val_acc = 0

    with torch.no_grad():
        for idx, (image, label) in enumerate(val_loader):
            epochloop.set_postfix_str(f'Validation batch {idx}/{len(val_loader)}')
            image, label = image.to(device), label.to(device)

            # forward pass
            out = model(image)

           # acc
            valid_acc += (torch.argmax(out, 1) == label).sum().item()
            val_samples += len(label)
            #val_acc = valid_acc / val_samples

            # loss
            loss = criterion(out, label)
            val_loss += loss.item()

        val_acc = valid_acc


    model.train()
    # save model if validation loss decrease
    # validation이 가장 작은 loss를 가질 때, 그 모델을 저장해두자.
    if val_loss / len(val_loader) <= best_valid_loss:
        best_valid_loss = val_loss / len(val_loader)
        best_epoch = epoch
        #print("best_valid_loss : {}, val_loss / len(valid_dataset) : {}".format(best_valid_loss, val_loss/len(valid_dataset)))
        torch.save(model.state_dict(), "crnn_20191611.pth")

    # print epoch loss & accuracy
    print(f'Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss / len(train_loader):.3f} | Train Acc: {train_acc / len(train_dataset):.3f} | Val Loss: {val_loss / len(val_loader):.3f} Val Acc: {val_acc / len(valid_dataset) * 100}%')
    tr_loss.append(train_loss / len(train_loader))
    tr_acc.append(train_acc / len(train_dataset) * 100)
    v_loss.append(val_loss / len(val_loader))
    v_acc.append(val_acc / len(valid_dataset) * 100)

#print("best epoch : {}".format(best_epoch))
#torch.save(model.state_dict(), "crnn_20191611.pth".format(best_epoch))


plt.plot(range(num_epochs), tr_loss, label="Train Loss")
plt.plot(range(num_epochs), v_loss, label="Val Loss")
plt.legend()
plt.title("Loss - epoch")
plt.show()
plt.plot(range(num_epochs), tr_acc, label="Train Acc")
plt.plot(range(num_epochs), v_acc, label="Val Acc")
plt.legend()
plt.title("Accuracy - epoch")
plt.show()

'''
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
'''
