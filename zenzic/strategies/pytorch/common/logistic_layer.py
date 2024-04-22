import torch
import torch.nn as nn
import numpy

""" Normalize data with logistic function (https://en.wikipedia.org/wiki/Logistic_function)"""
class LogisticLayer(nn.Module):
    def __init__(self, c_in: int, scale: float = 2.0) -> None:
        super(LogisticLayer, self).__init__()
        # The num of input channel.
        self.c_in = c_in
        # the supremum of the values of the function.
        self.A = nn.Parameter(torch.Tensor([scale] * self.c_in))
        # the x value of the function's midpoint.
        self.x0 = nn.Parameter(torch.zeros(self.c_in))
        # the logistic growth rate or steepness of the curve.
        self.k = nn.Parameter(torch.ones(self.c_in))
        # the bias
        self.bias = nn.Parameter(torch.Tensor([0.5] * self.c_in))
        # constants
        self.register_buffer('const_one', torch.Tensor([1.0]))
    
    def forward(self, x, invert: bool = False):
        if not invert:
            return self.A / (self.const_one + torch.exp(-self.k * (x - self.x0))) + self.bias
        else:
            A = self.A.detach()
            bias = self.bias.detach()
            k = self.k.detach()
            x0 = self.x0.detach()
            return (torch.log(A / (x - bias) - self.const_one) / -k) + x0
