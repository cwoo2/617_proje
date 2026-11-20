import math

import torch
import torch.nn as nn

from my_cnn_ext import my_cnn

KERNEL_SIZE = 3


class Conv3x3Function(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight):
        x = x.contiguous()
        weight = weight.contiguous()
        ctx.save_for_backward(x, weight)
        return my_cnn.my_convolution(x, weight)

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
