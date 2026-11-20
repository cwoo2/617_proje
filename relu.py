import torch
import torch.nn as nn

from my_cnn_ext import my_cnn


class ReluFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        x = x.contiguous()
        ctx.save_for_backward(x)
        return my_cnn.my_relu(x)

    @staticmethod
    def backward(ctx, grad_output):
        (x,) = ctx.saved_tensors
        return grad_output * (x > 0)


class CustomReLU(nn.Module):
    def forward(self, x):
        return ReluFunction.apply(x)
