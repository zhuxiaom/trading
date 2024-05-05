import torch
import torch.nn as nn

""" Min and max of sliding windows in 1D tensor with the shape of (batch_size, data_size, channel_size) """
class Range1DLayer(nn.Module):
    def __init__(self, window_size: int, channel_last: bool = True) -> None:
        super(Range1DLayer, self).__init__()
        self.windows_size = window_size
        self.channel_last = channel_last
        self.max_pool = nn.MaxPool1d(self.windows_size, stride=1)
    
    """ Return the min and max values in sliding window of x with the second half of the channel dim being min values. """
    def forward(self, x: torch.Tensor):
        if self.channel_last:
            x = x.permute(0, 2, 1)
        channels = x.shape[1]
        # replicate x with negative values to find min by using max_pool.
        x = torch.cat([x, -x], dim=1)
        padding = torch.ones(x.shape[0], x.shape[1], self.windows_size-1, device=x.device) * torch.tensor([-float('inf')], device=x.device)
        x = torch.cat([padding, x], dim=2)
        range = self.max_pool(x)
        range[:, channels:, :] = -range[:, channels:, :]  # Set correct min values.
        if self.channel_last:
            range = range.permute(0, 2, 1).contiguous()
        return range
