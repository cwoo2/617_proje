import argparse
import math
import os
import random
import time
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.cpp_extension import load
from torchvision import datasets, transforms

from simple_cnn import KERNEL_SIZE

if torch.cuda.is_available():
    major, minor = torch.cuda.get_device_capability()
    os.environ.setdefault("TORCH_CUDA_ARCH_LIST", f"{major}.{minor}")

SOURCE_DIR = Path(__file__).resolve().parent
CONV_SOURCE = (SOURCE_DIR / "conv.cu").read_text()

conv_ext = load(
    name="thrust_conv_stl10",
    sources=["conv.cu"],
    extra_cuda_cflags=["-O3"],
    verbose=True,
)


class ThrustConv3x3Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        x = x.contiguous()
        weight = weight.contiguous()
        bias = bias.contiguous()
        ctx.save_for_backward(x, weight, bias)
        return conv_ext.my_convolution(x, weight, x.shape[2], x.shape[3])

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, bias = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        grad_input = torch.nn.grad.conv2d_input(
            x.shape, weight, grad_output, stride=1, padding=1
        )
        grad_weight = torch.nn.grad.conv2d_weight(
            x, weight.shape, grad_output, stride=1, padding=1
        )
        grad_bias = grad_output.sum(dim=(0, 2, 3))
        return grad_input, grad_weight, grad_bias


class ThrustConv3x3(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(out_channels, in_channels, KERNEL_SIZE, KERNEL_SIZE))
        self.bias = nn.Parameter(torch.empty(out_channels))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1 / math.sqrt(in_channels * KERNEL_SIZE ** 2)
        nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        return ThrustConv3x3Function.apply(x, self.weight, self.bias)


class SimpleCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            ThrustConv3x3(32, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


def check_custom_conv(device):
    torch.manual_seed(123)
    x = torch.randn(8, 32, 48, 48, device=device)
    reference = nn.Conv2d(32, 64, 3, padding=1).to(device)
    custom = ThrustConv3x3(32, 64).to(device)
    custom.load_state_dict(reference.state_dict())

    with torch.no_grad():
        y_ref = reference(x)
        y_custom = custom(x)
    forward_error = (y_ref - y_custom).abs().max().item()

    grad_output = torch.randn_like(y_ref)
    x_ref = x.clone().requires_grad_()
    w_ref = reference.weight.detach().clone().requires_grad_()
    b_ref = reference.bias.detach().clone().requires_grad_()
    grads_ref = torch.autograd.grad(
        (nn.functional.conv2d(x_ref, w_ref, b_ref, padding=1) * grad_output).sum(),
        [x_ref, w_ref, b_ref],
    )

    x_new = x.clone().requires_grad_()
    w_new = reference.weight.detach().clone().requires_grad_()
    b_new = reference.bias.detach().clone().requires_grad_()
    grads_new = torch.autograd.grad(
        (ThrustConv3x3Function.apply(x_new, w_new, b_new) * grad_output).sum(),
        [x_new, w_new, b_new],
    )

    grad_errors = [
        (a - b).abs().max().item() for a, b in zip(grads_ref, grads_new)
    ]
    print(f"Custom conv check | forward max error: {forward_error:.2e}")
    print(
        "Custom conv check | grad max error (input, weight, bias): "
        f"{grad_errors[0]:.2e}, {grad_errors[1]:.2e}, {grad_errors[2]:.2e}"
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train a simple CNN on STL-10 with a Thrust conv2d kernel"
    )
    parser.add_argument("--data-dir", default="./data")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            predictions = model(images).argmax(dim=1)
            correct += (predictions == labels).sum().item()
            total += labels.size(0)
    return correct / total


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("The custom Thrust conv2d requires a CUDA device")
    print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")

    check_custom_conv(device)

    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    train_transform = transforms.Compose([
        transforms.RandomCrop(96, padding=8),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    train_set = datasets.STL10(
        args.data_dir, split="train", download=True, transform=train_transform
    )
    test_set = datasets.STL10(
        args.data_dir, split="test", download=True, transform=test_transform
    )
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=True,
    )

    model = SimpleCNN().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"Train images: {len(train_set)} | Test images: {len(test_set)}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("Custom kernel: Thrust conv3x3 on layer 2 (32->64, tied-heaviest layer)")

    epoch_times = []
    torch.cuda.synchronize()
    train_start = time.perf_counter()
    for epoch in range(1, args.epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in train_loader:
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            correct += (outputs.argmax(dim=1) == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / total
        train_acc = correct / total
        test_acc = evaluate(model, test_loader, device)
        torch.cuda.synchronize()
        epoch_time = time.perf_counter() - epoch_start
        epoch_times.append(epoch_time)
        print(
            f"Epoch {epoch:2d}/{args.epochs} | "
            f"train loss {train_loss:.4f} | "
            f"train acc {train_acc * 100:5.2f}% | "
            f"test acc {test_acc * 100:5.2f}% | "
            f"{epoch_time:6.2f}s"
        )

    total_time = time.perf_counter() - train_start
    print(
        f"Total training time: {total_time:.2f}s | "
        f"{total_time / args.epochs:.2f}s/epoch avg | "
        f"{len(train_set) * args.epochs / total_time:.1f} images/s"
    )

    torch.save(model.state_dict(), "stl10_cnn_thrust.pt")
    print("Saved model to stl10_cnn_thrust.pt")


if __name__ == "__main__":
    main()
