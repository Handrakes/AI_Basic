import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
import torch.nn.init as init

## HyperParameter
num_classes = 10
in_channel = 3
batch_size = 128
max_pool_kernel = 2
learning_rate = 0.005

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'mps')

train_data = torchvision.datasets.CIFAR10(root='./datasets',
                                        train=True,
                                        transform=transforms.ToTensor(),
                                        download=True)
train_loader = torch.utils.data.DataLoader(dataset = train_data,
                                          batch_size = batch_size,
                                          shuffle=True)

# Device Configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'mps')
print(f"device: {device}")

class CNN(nn.Module):
    def __init__(self, num_classes=10):

        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=32, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(num_features=32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=max_pool_kernel)  # 2 * 2
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=max_pool_kernel)
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.MaxPool2d(kernel_size=max_pool_kernel)
        )

        self.fc1 = nn.Linear(in_features=128 * 6 * 6, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=num_classes)

        self.apply(he_init)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        x = F.relu(x)
        x = x.reshape(x.size(0), -1)

        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.fc3(x)

        return x

def he_init(layer):
    if isinstance(layer, nn.Conv2d):
        init.kaiming_uniform_(layer.weight, nonlinearity='relu')
    elif isinstance(layer, nn.Linear):
        init.kaiming_uniform_(layer.weight, nonlinearity = 'linear')

model = CNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

total_loss = []

def train(model, train_loader, optimizer):
    model.train()
    for idx, (images, label) in enumerate(train_loader):

        images = images.to(device)
        label = label.to(device)

        outputs = model(images)

        optimizer.zero_grad()
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

        total_loss.append(loss.item())
        if idx % 200 == 0:
            if 100. * idx / len(train_loader) == 0:
                print(
                    f"train Epoch: {epoch} [{idx * len(images)}/{len(train_loader.dataset)}({100. * idx / len(train_loader):.0f}%)]\t\tTrain Loss: {loss.item()}")
            else:
                print(
                    f"train Epoch: {epoch} [{idx * len(images)}/{len(train_loader.dataset)}({100. * idx / len(train_loader):.0f}%)]\tTrain Loss: {loss.item()}")


## train model
epoch = 20
for epoch in range(1, epoch + 1):
    train(model, train_loader, optimizer)

torch.save(model.state_dict(),"model.pth")
plt.plot(total_loss)
plt.title("CIFAR10 CNN loss example")
plt.show()
print("Test Completed")

