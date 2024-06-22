import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

""" Normalize data with logistic function (https://en.wikipedia.org/wiki/Logistic_function)"""
class LogisticLayer(nn.Module):

    __constants__ = ['max_k',]
    shape: int|tuple
    k: nn.Parameter
    x_0: nn.Parameter

    def __init__(self, shape: int|tuple, max_k: float=3.0) -> None:
        super(LogisticLayer, self).__init__()
        self.max_k = max_k
        self.shape = shape
        # the x value of the function's midpoint.
        self.x_0 = nn.Parameter(torch.empty(self.shape))
        # the logistic growth rate or steepness of the curve.
        self.k = nn.Parameter(torch.empty(self.shape))

        self.reset_parameters()
    
    def reset_parameters(self) -> None:
        # Setting a=sqrt(5) in kaiming_uniform is the same as initializing with
        # uniform(-1/sqrt(in_features), 1/sqrt(in_features)). For details, see
        # https://github.com/pytorch/pytorch/issues/57109
        if self.shape != 0:
            init.kaiming_uniform_(self.k, a=math.sqrt(5))
            if self.x_0 is not None:
                fan_in, _ = init._calculate_fan_in_and_fan_out(self.k)
                bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
                init.uniform_(self.x_0, -bound, bound)
    
    def forward(self, x):
        return F.sigmoid(self.max_k*F.sigmoid(self.k)*(x-self.x_0))

class LazyLogisticLayer(nn.modules.lazy.LazyModuleMixin, LogisticLayer):

    cls_to_become = LogisticLayer
    k: nn.UninitializedParameter
    x_0: nn.UninitializedParameter

    def __init__(self, max_k: float=3.0, device=None, dtype=None) -> None:
        super(LazyLogisticLayer, self).__init__(0, max_k)
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.k = nn.UninitializedParameter(**factory_kwargs)
        self.x_0 = nn.UninitializedParameter(**factory_kwargs)
    
    def reset_parameters(self) -> None:
        if not self.has_uninitialized_params():
            super().reset_parameters()
    
    def initialize_parameters(self, input) -> None:
        if self.has_uninitialized_params():
            with torch.no_grad():
                self.shape = input.shape[1:]    # skip batch_size
                self.k.materialize(self.shape)
                self.x_0.materialize(self.shape)
                self.reset_parameters()
