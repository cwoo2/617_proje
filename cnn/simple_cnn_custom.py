import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.python.conv import KERNEL_SIZE, Conv3x3Function, CustomConv2d
from layers.python.maxpool import CustomMaxPool2d
from layers.python.relu import CustomReLU


class SimpleCNNCustom(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # Same as SimpleCNN, but convs and relus are the custom kernels (no padding)
        self.conv1 = CustomConv2d(3, 32)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = CustomReLU()
        self.mp1 = CustomMaxPool2d(2)
        self.conv2 = CustomConv2d(32, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = CustomReLU()
        self.mp2 = CustomMaxPool2d(2)
        self.conv3 = CustomConv2d(64, 128)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = CustomReLU()
        self.mp3 = CustomMaxPool2d(2)
        self.aap = nn.AdaptiveAvgPool2d(1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)
        x = self.mp1(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu2(x)
        x = self.mp2(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu3(x)
        x = self.mp3(x)

        x = self.aap(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x
