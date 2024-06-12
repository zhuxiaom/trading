import torch
import torch.nn as nn
import numpy

""" Normalize data with logistic function (https://en.wikipedia.org/wiki/Logistic_function)"""
class LogisticLayer(nn.Module):
    def __init__(self, shape: int|tuple, scale: float = 2.0) -> None:
        super(LogisticLayer, self).__init__()
        # The num of input channel.
        self.shape = shape
        # the supremum of the values of the function.
        self.A = nn.Parameter(torch.Tensor([scale] * self.shape))
        # the x value of the function's midpoint.
        self.x0 = nn.Parameter(torch.zeros(self.shape))
        # the logistic growth rate or steepness of the curve.
        self.k = nn.Parameter(torch.ones(self.shape))
        # the bias
        self.bias = nn.Parameter(torch.Tensor([0.5] * self.shape))
    
    def forward(self, x, invert: bool = False):
        if not invert:
            # scale value to [-scale/2, scale/2]
            return self.A * (1 / (1.0 + torch.exp(-self.k * (x - self.x0))) - self.bias)
        else:
            A = self.A.detach()
            bias = self.bias.detach()
            k = self.k.detach()
            x0 = self.x0.detach()
            return (torch.log(1.0 / (x / A + bias) - 1.0) / -k) + x0
