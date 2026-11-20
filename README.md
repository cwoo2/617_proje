# CNN From Scratch Using CUDA

## Summary

In this project, my goal is to implement the computationally heavy operations of a convolutional neural network from scratch using CUDA and compare the performance of my implementations against PyTorch's CUDA-enabled implementations. I don't expect to match or beat their implementations, but I plan on using Nsight Compute to see why theirs is faster.

## Current Progress

- Naive CUDA implementations of forward pass for convolution, ReLU, and max-pooling operations on 4D tensors

## To-do

- Profile current kernels and identify specific opportunities for performance gains
- Implement forward pass for remaining layer types (batch norm, adaptive avg pooling, flatten, linear)
- Implement backward pass for layers (stretch goal)