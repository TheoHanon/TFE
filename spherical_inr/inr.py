import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init
from layers import *
from typing import Union, Tuple


class SphericalNet(nn.Module):

    """Spherical SIREN model. The model performs a spherical harmonics embedding of the input coordinates and then applies a fully connected neural network to the embedded coordinates.

    Args:
    -----
        L0 (int): The maximum degree of the spherical harmonics embedding.
        Q (int): The depth of the neural network.
        hidden_features (int): The number of hidden features per layer.
        out_features (int, optional): The number of output features. Defaults to 2.
        activation (callable, optional): The activation function of the neural network. Defaults to nn.sin.
        bias (bool, optional): Whether to include bias in the neural network. Defaults to False.
    

    """


    def __init__(
            self, 
            L0: int, 
            Q : int,  
            hidden_features : int, 
            out_features : int = 1, 
            activation : callable = torch.sin, 
            first_activation : callable = None,
            bias : bool = True, 
            device : torch.device = torch.device("cpu"), 
            spectral_norm: bool = False
        ) -> None :

        super(SphericalNet, self).__init__()

        self.L0 = L0
        self.Q = Q
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.activation = activation
        self.first_activation = activation if first_activation is None else first_activation
        self.device = device

        self.spectral_norm = spectral_norm
        self.spherical_harmonics_embedding = SphericalHarmonicsEmbedding(L0, device=self.device)
        self.net = []

        for i in range(Q):
            if i == Q - 1:
                self.net.append(
                    nn.utils.spectral_norm(nn.Linear(hidden_features, out_features, bias=bias)) if spectral_norm 
                    else nn.Linear(hidden_features, out_features, bias=bias)
                )
            elif i == 0:
                self.net.append(
                    MLPLayer((L0+1)**2, hidden_features, bias=bias, activation=self.first_activation, spectral_norm=spectral_norm)
                )
            else:
                self.net.append(
                    MLPLayer(hidden_features, hidden_features, bias=bias, activation=self.activation, spectral_norm=spectral_norm)
                )
        
        self.net = nn.Sequential(*self.net)
        self.init_weights()

        
    def init_weights(self) -> None:

        for layer in self.net:
            if isinstance(layer, MLPLayer):
                layer.init_weights()
            elif isinstance(layer, nn.Linear):
                init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    init.zeros_(layer.bias)

    
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        emb = self.spherical_harmonics_embedding(x)
        out = self.net(emb)
        return out
    
    
    def forward_inside(self, x : torch.Tensor) -> dict:
        history = {}
        with torch.no_grad():
            out = self.spherical_harmonics_embedding(x)
            history['emb'] = out.clone().detach()

            for i, layer in enumerate(self.net[:-1]): 
                out = layer(out)
                history[f"layer_{i}"] = out.clone().detach()

            out = self.net[-1](out)
            history["output"] = out.clone().detach()

        return history
        

class SirenNet(nn.Module):
    """
    SIREN (Sinusoidal Representation Network) model.

    Args:
    -----
        in_features (int): Number of input features.
        hidden_features (int): Number of hidden features per layer.
        out_features (int): Number of output features.
        first_omega_0 (float): Frequency scaling factor for the first layer.
        hidden_omega (float): Frequency scaling factor for the hidden layers.
        n_hidden_layers (int, optional): Number of hidden layers. Defaults to 1.
        last_linear (bool, optional): Whether the final layer is a linear layer. Defaults to True.
    """
    def __init__(
        self,
        in_features: int,
        hidden_features: int,
        out_features: int,
        first_omega_0: float,
        hidden_omega: float,
        n_hidden_layers: int = 1,
        last_linear: bool = True
    ) -> None:
        super(SirenNet, self).__init__()

        self.first_layer = SineLayer(in_features, hidden_features, omega_0=first_omega_0, is_first=True)
        
        self.net = []
        for _ in range(n_hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, omega_0=hidden_omega))
        self.net = nn.Sequential(*self.net)

        if last_linear:
            self.final_layer = nn.Linear(hidden_features, out_features, bias=False)
            with torch.no_grad():
                self.final_layer.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega, np.sqrt(6 / hidden_features) / hidden_omega)
        else:
            self.final_layer = SineLayer(hidden_features, out_features, omega_0=hidden_omega)

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        x = self.first_layer(coords)
        x = self.net(x)
        return self.final_layer(x)
    