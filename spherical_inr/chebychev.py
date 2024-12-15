import torch
import torch.nn as nn

class Chebyshev(nn.Module):
    def __init__(self, order, alpha=1.0):
        super(Chebyshev, self).__init__()
        self.order = order
        self.alpha = alpha

    def forward(self, x):
    
        x_val = x / self.alpha
        eps = 1e-7  # small epsilon for numerical stability
        
        # Determine which values are within [-1, 1]
        within_mask = (x_val.abs() <= 1)
        out = torch.empty_like(x_val)
        
        # For values within [-1, 1], clamp to [-1+eps, 1-eps] before acos
        x_clamped = x_val[within_mask].clamp(-1 + eps, 1 - eps)
        out[within_mask] = torch.cos(self.order * torch.acos(x_clamped))
        
        outside_mask = ~within_mask
        if outside_mask.any():
            x_out = x_val[outside_mask]
            out[outside_mask] = 0.5 * ((x_out + torch.sqrt(x_out**2 - 1))**self.order + (x_out - torch.sqrt(x_out**2 - 1))**self.order)
        return out


