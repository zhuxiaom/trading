import math
import torch
import torch.nn as nn
import torch.nn.init as init

""" Learnable masks layer """
class MaskLayer(nn.Module):
    """ Constructor 

        params:
            shape - The shape of the mask.
            k - The logistic growth rate or steepness of the curve.
            bias - Initial bias. Depending on the input, bias can be
                    used to control the initial masks.
    """
    def __init__(self, shape: int | tuple, k: float=1.0) -> None:
        super(MaskLayer, self).__init__()
        self.activated_count = 0
        # The num of input channel.
        self.shape = shape
        # the coefficiency of linear transfomation.
        self.weights = nn.Parameter(torch.empty(self.shape))
        # the bias of linear transfomation.
        self.bias = nn.Parameter(torch.empty(self.shape))
        # the logistic growth rate or steepness of the curve.
        self.register_buffer('k', torch.tensor(k, requires_grad=False))
        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        init.kaiming_uniform_(self.weights, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = init._calculate_fan_in_and_fan_out(self.weights)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            init.uniform_(self.bias, -bound, bound)
    
    def forward(self, x):
        # linear transformation of x.
        trans_x = self.weights * x + self.bias
        # logistic transformation to produce values between 0 and 1 that are
        # used as masks. This is because we need differentiable version of:
        #   f(x) = (0 if x < 0 else 1)
        mask = 1 / (1 + torch.exp(-self.k * trans_x))
        res = x * mask
        self.activated_count = torch.count_nonzero(mask > 0.5) * 1.0
        return res
