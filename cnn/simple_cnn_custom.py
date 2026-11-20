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


if __name__ == "__main__":
    # Compares the custom kernels against their torch equivalents for fwd and bwd.
    torch.manual_seed(0)
    device = "cuda"

    x = torch.randn(4, 3, 20, 20, device=device)
    ref = nn.Conv2d(3, 8, KERNEL_SIZE, bias=False).to(device)
    custom = CustomConv2d(3, 8).to(device)
    custom.load_state_dict(ref.state_dict())

    y_ref = ref(x)
    y_custom = custom(x)
    grad_output = torch.randn_like(y_ref)

    xr = x.clone().requires_grad_()
    wr = ref.weight.detach().clone().requires_grad_()
    grads_ref = torch.autograd.grad(
        (nn.functional.conv2d(xr, wr) * grad_output).sum(), [xr, wr]
    )

    xc = x.clone().requires_grad_()
    wc = ref.weight.detach().clone().requires_grad_()
    grads_custom = torch.autograd.grad(
        (Conv3x3Function.apply(xc, wc) * grad_output).sum(), [xc, wc]
    )

    print("shape:", y_ref.shape, y_custom.shape)
    print("forward max err:", (y_ref - y_custom).abs().max().item())
    print(
        "grad max err (input, weight):",
        [(a - b).abs().max().item() for a, b in zip(grads_ref, grads_custom)],
    )

    # maxpool fwd/bwd
    p = torch.randn(4, 8, 11, 11, device=device)
    y_ref = F.max_pool2d(p, 2)
    y_custom = CustomMaxPool2d(2)(p)
    g = torch.randn_like(y_ref)

    pr = p.clone().requires_grad_()
    pc = p.clone().requires_grad_()
    grad_ref = torch.autograd.grad((F.max_pool2d(pr, 2) * g).sum(), pr)[0]
    grad_custom = torch.autograd.grad((CustomMaxPool2d(2)(pc) * g).sum(), pc)[0]

    print("maxpool shape:", y_ref.shape, y_custom.shape)
    print("maxpool forward max err:", (y_ref - y_custom).abs().max().item())
    print("maxpool grad max err:", (grad_ref - grad_custom).abs().max().item())
