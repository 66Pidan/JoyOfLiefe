import torch.nn as nn
import torch.nn.functional as F
import torch

class PersonFeatureExtractor(nn.Module):
    def __init__(self, num_classes):
        super(PersonFeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1, padding=1, bias=True)
        nn.init.kaiming_normal_(self.conv1.weight, mode='fan_out', nonlinearity='relu')
        self.conv1.bias.data.fill_(0.0)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 32 * 32, 64)
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = torch.relu(self.conv1(x))
        x = self.pool1(x)
        x = torch.relu(self.conv2(x))
        x = self.pool2(x)
        x = x.view(-1, 64 * 32 * 32)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x