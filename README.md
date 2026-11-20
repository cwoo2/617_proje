# CNN From Scratch Using CUDA

## Summary

In this project, my goal is to implement the computationally heavy operations of a convolutional neural network from scratch using CUDA and compare the performance of my implementations against PyTorch's CUDA-enabled implementations. I don't expect to match or beat their implementations, but I plan on using Nsight Compute to see why theirs is faster.

## Dataset

Training uses [STL-10](https://ai.stanford.edu/~acoates/stl10/), an image-classification benchmark with 10 classes of 96x96 color images. The labeled split contains 5,000 training images and 8,000 test images; the dataset also provides an unlabeled split. This project trains and evaluates on the labeled splits. The script downloads the dataset to `./data` by default if needed.

## Training

Run from the repository root with Python, PyTorch (CUDA-enabled), and torchvision installed:

```bash
python train_stl10.py                 # PyTorch layers (default)
python train_stl10.py --custom        # custom CUDA kernels
```

The custom model uses this project's CUDA kernels and requires a CUDA-capable GPU and CUDA toolkit. The default model runs on CUDA when available, otherwise on CPU. The model weights are saved to `stl10_cnn.pt` in the current directory.

Available options:

| Option | Default | Description |
|---|---:|---|
| `--custom` | off | Use the custom-kernel model instead of the PyTorch-layer model |
| `--data-dir` | `./data` | Dataset download/storage directory |
| `--epochs` | `10` | Number of training epochs |
| `--batch-size` | `64` | Batch size for training and evaluation |
| `--lr` | `0.001` | Adam learning rate |
| `--workers` | `2` | Data-loader worker processes |
| `--seed` | `0` | Random seed |

## Current Progress

- Naive CUDA implementations of forward pass for convolution, ReLU, and max-pooling operations on 4D tensors

## To-do

- Profile current kernels and identify specific opportunities for performance gains
- Implement forward pass for remaining layer types (batch norm, adaptive avg pooling, flatten, linear)
- Implement backward pass for layers (stretch goal)