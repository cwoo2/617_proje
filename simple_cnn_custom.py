import math
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.cpp_extension import load

KERNEL_SIZE = 3

conv_ext = load(
    name="simple_cnn_conv",
    sources=[str(Path(__file__).resolve().parent / "conv.cu")],
    extra_cuda_cflags=["-O3"],
)


class Conv3x3Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        x = x.contiguous()
        weight = weight.contiguous()
        ctx.save_for_backward(x, weight)
        return conv_ext.my_convolution(x, weight)

    @staticmethod
    def backward(ctx, grad_output):
        x, weight = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_input = torch.nn.grad.conv2d_input(x.shape, weight, grad_output)
        grad_weight = torch.nn.grad.conv2d_weight(x, weight.shape, grad_output)
        return grad_input, grad_weight


class CustomConv2d(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.weight = nn.Parameter(
            torch.empty(out_channels, in_channels, KERNEL_SIZE, KERNEL_SIZE)
        )
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))

    def forward(self, x):
        return Conv3x3Function.apply(x, self.weight)


class SimpleCNNCustom(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        # Same as SimpleCNN, but every conv is the custom 3x3 kernel (no padding)
        self.conv1 = CustomConv2d(3, 32)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu1 = nn.ReLU(inplace=True)
        self.mp1 = nn.MaxPool2d(2)
        self.conv2 = CustomConv2d(32, 64)
        self.bn2 = nn.BatchNorm2d(64)
        self.relu2 = nn.ReLU(inplace=True)
        self.mp2 = nn.MaxPool2d(2)
        self.conv3 = CustomConv2d(64, 128)
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


if __name__ == "__main__":
    # Compares the custom kernel against nn.Conv2d (padding=0) for fwd and bwd.
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
