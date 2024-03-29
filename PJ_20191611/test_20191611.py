# -*- coding: utf-8 -*-
"""Untitled1.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/15Sa7qw7-JuTujPucNDaLYku21vjhkuFs
"""

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import models, transforms


import matplotlib.pyplot as plt
import random
import time
import os

import numpy as np


# mount google drive
from google.colab import drive
drive.mount('/content/gdrive')

# unzip train, valid dataset
# target 을 정하지 않으면 google drive 내의 content 드라이브에 위치시킴
# 런타임을 다시 시작할 때 마다 unzip 을 새로 해주어야 함.
!unzip /content/gdrive/MyDrive/2023_EEE4178_project/train.zip
!unzip /content/gdrive/MyDrive/2023_EEE4178_project/valid.zip
# !unzip /content/gdrive/MyDrive/Project_dataset/Font_npy_100_test.zip

import os
import glob
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms
import matplotlib.pyplot as plt
import torchvision.transforms.functional as TF


class MyDataset(Dataset):
    def __init__(self, npy_dir, label_dict=None):
        self.dir_path = npy_dir
        self.to_tensor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.transpose(0, 1)),
            transforms.Lambda(lambda x: TF.rotate(x, -90))
        ])
        self.npy_path = glob.glob(os.path.join(npy_dir, '*', '*.npy'))
        self.label_dict = label_dict or self.create_label_dict()

    def create_label_dict(self):
        label_dict = {}
        for path in self.npy_path:
            label_name = os.path.basename(os.path.dirname(path))
            if label_name not in label_dict:
                label_dict[label_name] = len(label_dict)
        return label_dict

    def __getitem__(self, index):
        single_data_path = self.npy_path[index]
        data = np.load(single_data_path, allow_pickle=True)

        image = data['image']
        image = self.to_tensor(image)
        image = TF.hflip(image)

        label_name = os.path.basename(os.path.dirname(single_data_path))
        label = self.label_dict[label_name]
        label = torch.tensor(label, dtype=torch.long)

        return (image, label)

    def __len__(self):
        return len(self.npy_path)

label_dict = {
    '30': 0, '31': 1, '32': 2, '33': 3, '34': 4, '35': 5, '36': 6, '37': 7, '38': 8, '39': 9,
    '41': 10, '42': 11, '43': 12, '44': 13, '45': 14, '46': 15, '47': 16, '48': 17, '49': 18,
    '4a': 19, '4b': 20, '4c': 21, '4d': 22, '4e': 23, '50': 24, '51': 25, '52': 26, '53': 27,
    '54': 28, '55': 29, '56': 30, '57': 31, '58': 32, '59': 33, '5a': 34, '61': 35, '62': 36,
    '64': 37, '65': 38, '66': 39, '67': 40, '68': 41, '69': 42, '6a': 43, '6d': 44, '6e': 45,
    '6f': 46, '71': 47, '72': 48, '74': 49, '75': 50, '79': 51,
}



# unzip 한 디렉토리 있는 path 그대로 넣어야, 디렉토리 옆 점 3개 누르면 '경로 복사' 있음 - 위의 사진 참조
train_data = MyDataset("/content/train", label_dict)
valid_data = MyDataset("/content/valid", label_dict)

print(len(train_data))
print(len(valid_data))

batch_size = 200

valid_loader = torch.utils.data.DataLoader(dataset=valid_data,
                                           batch_size=batch_size,
                                           shuffle=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# FIX SEED
def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)  # type: ignore
    torch.backends.cudnn.deterministic = True  # type: ignore
    torch.backends.cudnn.benchmark = True  # type: ignore

seed_everything(42)

# hyper - parameters
num_classes = 52
input_size = 100
num_epochs = 13
learning_rate = 0.001 # 0.001
hidden_dim = 128
num_layers = 3
in_channel = 1
embedding_dim = 64 #64
dropout = 0.1
max_pool_kernel = 2

class CNN(nn.Module):
  def __init__(self, embedding_dim, hidden_dim, num_layers):
        super(CNN, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=32, kernel_size=3, padding=2, stride=1),
            nn.BatchNorm2d(num_features=32),  # out_channels이 그대로 들어감
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d(kernel_size=max_pool_kernel, stride = 2)  # 2 * 2
        )
        #nn.init.kaiming_normal_(self.layer1.weight, mode='fan_out', nonlinearity='relu')
        self.layer1.apply(self._init_weights)
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=5, padding = 2, stride = 2),
            nn.BatchNorm2d(num_features=64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d(kernel_size=max_pool_kernel, stride = 2)
        )
        #nn.init.kaiming_normal_(self.layer2.weight, mode='fan_out', nonlinearity='relu')
        self.layer2.apply(self._init_weights)
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size = 3, stride = 2, padding = 2),
            nn.BatchNorm2d(num_features=128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d(kernel_size=max_pool_kernel)
        )
        #nn.init.kaiming_normal_(self.layer3.weight, mode='fan_out', nonlinearity='relu')
        self.layer3.apply(self._init_weights)
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels = 128, out_channels = 256, kernel_size = 3, stride = 1, padding = 2),
            nn.BatchNorm2d(num_features = 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d(kernel_size=max_pool_kernel)
        )
        self.layer4.apply(self._init_weights)
        #nn.init.kaiming_normal_(self.layer4.weight, mode='fan_out', nonlinearity='relu')
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels = 256, out_channels = 512, kernel_size = 2, stride = 1, padding = 2),
            nn.BatchNorm2d(num_features = 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.MaxPool2d(kernel_size=max_pool_kernel)
        )
        #nn.init.kaiming_normal_(self.layer5.weight, mode='fan_out', nonlinearity='relu')
        # fully - connected
        self.fc1 = nn.Linear(in_features=2304, out_features=256)
        nn.init.kaiming_normal_(self.fc1.weight, mode='fan_out', nonlinearity='relu')
        self.fc2 = nn.Linear(in_features=256, out_features=128)
        nn.init.kaiming_normal_(self.fc2.weight, mode='fan_out', nonlinearity='relu')
        #self.dropout1 = nn.Dropout(p=0.25, inplace=False)
        self.fc3 = nn.Linear(in_features=128, out_features=num_classes)

        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, num_layers=num_layers, batch_first=True)
        self.dropout = nn.Dropout(dropout)


  def _init_weights(self, m):
        if isinstance(m, nn.Conv2d):
          nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

  def forward(self, x):
    x = self.layer1(x)
    x = self.layer2(x)
    x = self.layer3(x)
    x = self.layer4(x)

    #x = x.reshape(x.size(0), -1, embedding_dim)

    #x, _ = self.lstm(x)

    x = x.reshape(x.size(0), -1)  # fully connected에 넣어주기 위해서 flatten 시켜주기

    x = self.fc1(x)
    #x = self.dropout(x)
    x = F.relu(x)
    x = self.fc2(x)
    #x = self.dropout(x)
    x = F.relu(x)
    x = self.fc3(x)

    return x  # 10개의 출력 return

model = CNN(embedding_dim, hidden_dim, num_layers).to(device)
model.eval()
model.load_state_dict(torch.load('./20191611.pth'))

with torch.no_grad():
  correct = 0

  for image, label in valid_loader:
    image = image.to(device)
    label = label.to(device)
    output = model(image)
    _ , pred = torch.max(output.data, 1)
    correct += (pred == label).sum().item()

  print('Accuracy of the last_model network on the {} valid images: {} %'.format(len(valid_data), 100 * correct / len(valid_data)))

