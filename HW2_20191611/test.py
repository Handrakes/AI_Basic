import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow

## train에서 직접 만드신 model class 코드가 들어가는 자리입니다
## train.py에서의 model class를 복사해서 붙이시면 됩니다.

class CNN(nn.Module):
    def __init__(self, num_classes=10):
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=32, kernel_size=2, stride=1, padding=2),
            nn.BatchNorm2d(num_features=32),  # out_channels이 그대로 들어감
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
            nn.MaxPool2d(kernel_size=max_pool_kernel)
        )
        # fully - connected
        self.fc1 = nn.Linear(in_features=128 * 6 * 6, out_features=128)
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.fc3 = nn.Linear(in_features=64, out_features=num_classes)

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


        return x  # 10개의 출력 return


## 위에 model class 코드를 입력하셨다면 test는 여기서부터 진행하시면 됩니다.
# Device Configuration
max_pool_kernel = 2
batch_size = 128
learning_rate = 0.005
num_classes = 10
in_channel= 3

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

test_data = torchvision.datasets.CIFAR10(root='./datasets',
                                       train=False,
                                       transform=transforms.ToTensor(),
                                       download=True)
test_loader = torch.utils.data.DataLoader(dataset=test_data,
                                          batch_size=batch_size,
                                          shuffle=False)

test_model = CNN().to(device)
criterion = nn.CrossEntropyLoss()

tmp = torch.load('model.pth')
test_model.load_state_dict(tmp, strict = False)


test_model.eval()

def test(model, test_loader):
    test_model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(device)
            label = label.to(device)
            output = model(image)
            prediction = output.max(1, keepdim=True)[1]
            correct += prediction.eq(label.view_as(prediction)).sum().item()

    test_accuracy = 100. * correct / len(test_loader.dataset)
    print("Accuracy : {} %".format(test_accuracy))


test(test_model, test_loader)
