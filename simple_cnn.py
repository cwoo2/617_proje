import torch
import torch.nn as nn

KERNEL_SIZE = 3
KERNEL_RADIUS = KERNEL_SIZE // 2

class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # Explicitly define each layer to enable getting outputs from intermediate layers
        self.conv1 = nn.Conv2d(3, 32, KERNEL_SIZE, padding=KERNEL_RADIUS, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.mp1 = nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(32, 64, KERNEL_SIZE, padding=KERNEL_RADIUS, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.mp2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(64, 128, KERNEL_SIZE, padding=KERNEL_RADIUS, bias=False)
        self.bn3 = nn.BatchNorm2d(128)
        self.relu3 = nn.ReLU(inplace=True)
        self.mp3 = nn.MaxPool2d(2)
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