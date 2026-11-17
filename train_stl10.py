import argparse
import random
import time

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from simple_cnn import SimpleCNN

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

def parse_args():
    parser = argparse.ArgumentParser(description="Train a simple CNN on STL-10")
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
    if device.type == "cuda":
        print(f"Using device: {device} ({torch.cuda.get_device_name(0)})")
    else:
        print(f"Using device: {device}")

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
        pin_memory=device.type == "cuda",
    )
    test_loader = DataLoader(
        test_set,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        pin_memory=device.type == "cuda",
    )

    model = SimpleCNN().to(device)
    model.conv1.register_forward_hook(get_activation("conv1"))
    model.conv2.register_forward_hook(get_activation("conv2"))
    model.conv3.register_forward_hook(get_activation("conv3"))
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    print(f"Train images: {len(train_set)} | Test images: {len(test_set)}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

    epoch_times = []
    if device.type == "cuda":
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
        if device.type == "cuda":
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

    torch.save(model.state_dict(), "stl10_cnn.pt")
    print("Saved model to stl10_cnn.pt")


if __name__ == "__main__":
    main()
