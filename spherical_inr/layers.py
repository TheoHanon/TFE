import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np
from spherical_harmonics_ylm import get_SH




class SphericalHarmonicsEmbedding(nn.Module):
    """Generates the spherical harmonics positional encoding of the input coordinates.

        gamma(theta, phi) = (Y_{0,0}(theta, phi), Y_{1,-1}(theta, phi), ..., Y_{L0,L0}(theta, phi))^T \in R^{(L0+1)^2}

    """

    def __init__(self, L0 : int, device : torch.device) -> None:

        super(SphericalHarmonicsEmbedding, self).__init__()

        self.L0 = L0
        self.device = device
        self.spherical_harmonics_func = []
    
        for l in range(self.L0 + 1):
            for m in range(-l, l + 1):
                self.spherical_harmonics_func.append(get_SH(m, l)) # pre-load the spherical harmonics functions        

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        
        theta, phi = x[...,0], x[...,1]
        emb = torch.zeros(x.size(0), (self.L0+1)**2, dtype=torch.float32, device = self.device)
        
        for idx, sh_func in enumerate(self.spherical_harmonics_func):
            emb[:, idx] = 1/np.sqrt(2) * sh_func(theta, phi)

        return emb
    

class MLPLayer(nn.Module):

    def __init__(self, input_features : int , output_features : int , bias : bool, activation : callable, spectral_norm : bool = False) -> None:
        super(MLPLayer, self).__init__()
        linear = nn.Linear(input_features, output_features, bias = bias)
        self.layer = nn.utils.spectral_norm(linear) if spectral_norm else linear
        self.activation = activation

        self.spectral_norm = spectral_norm


        self.fan_in = input_features
        self.fan_out = output_features

    def forward(self, x : torch.Tensor) -> torch.Tensor:

        out = self.layer(x)

        if self.spectral_norm:
            out /= np.sqrt(self.fan_in)

        return self.activation(out)
    
    def init_weights(self) -> None:
        with torch.no_grad():
            init.xavier_uniform_(self.layer.weight)
            if self.layer.bias is not None:
                init.zeros_(self.layer.bias)



class SineLayer(nn.Module):
    """
    A single layer with a sine activation function for SIREN models.

    Args:
    -----
        in_features (int): Number of input features.
        out_features (int): Number of output features.
        omega_0 (float, optional): Frequency scaling factor. Defaults to 30.
        is_first (bool, optional): Whether this layer is the first layer. Defaults to False.
        bias (bool, optional): Whether to include a bias term in the linear layer. Defaults to True.
    """
    def __init__(self, in_features: int, out_features: int, omega_0: float = 30.0, is_first: bool = False, bias: bool = True) -> None:
        super(SineLayer, self).__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features

        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self) -> None:
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.omega_0 * self.linear(input))