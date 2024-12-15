import torch
import torch.nn as nn

class Chebyshev(nn.Module):
    def __init__(self, order, alpha=1.0):
        super(Chebyshev, self).__init__()
        self.order = order
        self.alpha = alpha

    def forward(self, x):
        x_val = x / self.alpha
        out = torch.cos(self.order * torch.acos(x_val))

        return out

