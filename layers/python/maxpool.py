import torch
import torch.nn as nn
import torch.nn.functional as F

from my_cnn_ext import my_cnn


class MaxPool2dFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, kernel_size, stride):
        x = x.contiguous()
        ctx.save_for_backward(x)
        ctx.kernel_size = kernel_size
        ctx.stride = stride
        # -1 tells the kernel to use stride = kernel_size
        return my_cnn.my_maxpool(x, kernel_size, -1 if stride is None else stride)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        grad_output = grad_output.contiguous()
        _, indices = F.max_pool2d(
            x, ctx.kernel_size, ctx.stride, return_indices=True
        )
        return F.max_unpool2d(
            grad_output,
            indices,
            ctx.kernel_size,
            ctx.stride,
            output_size=x.shape[-2:],
        ), None, None


class CustomMaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride = stride

    def forward(self, x):
        return MaxPool2dFunction.apply(x, self.kernel_size, self.stride)
