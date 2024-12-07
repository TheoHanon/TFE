import torch
import torch.nn as nn

class Chebyshev(nn.Module):
    def __init__(self, order, alpha=1.0, eps=1e-6):
        super(Chebyshev, self).__init__()
        self.order = order
        self.alpha = alpha
        self.eps = eps


    def forward(self, x):
        x_val = x / self.alpha

        safe_x_val = x_val.clamp(-1.0 + self.eps, 1.0 - self.eps)
        out = torch.cos(self.order * torch.acos(safe_x_val))

        return out

