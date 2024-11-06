import torch
import numpy as np
import torch.nn as nn
import torch.nn.init as init
from spherical_harmonics_ylm import get_SH
from typing import Union, Tuple

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
            emb[:, idx] = sh_func(theta, phi)

        return emb
    

class MLPLayer(nn.Module):

    def __init__(self, input_features : int , output_features : int , bias : bool, activation : callable) -> None:
        super(MLPLayer, self).__init__()
        self.layer = nn.Linear(input_features, output_features, bias = bias)
        self.activation = activation


    def forward(self, x : torch.Tensor) -> torch.Tensor:
        return self.activation(self.layer(x))



class SphericalSiren(nn.Module):

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
            bias : bool = True, 
            device : torch.device = torch.device("cpu"), 
            init : str = "siren",    
        ) -> None :

        super(SphericalSiren, self).__init__()

        self.L0 = L0
        self.Q = Q
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.activation = activation
        self.device = device
        self.init = init
    
        self.spherical_harmonics_embedding = SphericalHarmonicsEmbedding(L0, device=self.device)
        self.net = []

        for i in range(Q+1):
            if i == 0:
                self.net.append(MLPLayer((L0+1)**2, hidden_features, bias = bias, activation=self.activation).to(self.device))

            elif i == Q:
                self.net.append(nn.Linear(hidden_features, out_features, bias = bias).to(self.device))
            
            else :
                self.net.append(MLPLayer(hidden_features, hidden_features, bias = bias, activation=self.activation).to(self.device))
                
        
        self.net = nn.Sequential(*self.net)
        self.init_weights()


    def _laurent_init(self, weight : torch.tensor) -> None:
        fan_in = weight.size(1)
        with torch.no_grad():
            for l in range(weight.size(0)):
                weight[l].uniform_(-1/(l+1), 1/(l+1))
        
    def init_weights(self) -> None:

        if self.init == "xavier":   
            for layer in self.net:
                if isinstance(layer, MLPLayer):
                    init.xavier_uniform_(layer.layer.weight)
                    if layer.layer.bias is not None:
                        init.zeros_(layer.layer.bias)
                elif isinstance(layer, nn.Linear):
                    init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        init.zeros_(layer.bias)

        elif self.init == "kaiming":

            for layer in self.net:
                if isinstance(layer, MLPLayer):
                    init.kaiming_uniform_(layer.layer.weight)
                    if layer.layer.bias is not None:
                        init.zeros_(layer.layer.bias)
                elif isinstance(layer, nn.Linear):
                    init.kaiming_uniform_(layer.weight)
                    if layer.bias is not None:
                        init.zeros_(layer.bias)

        elif self.init == "laurent_xavier" :
            
            self._laurent_init(self.net[0].layer.weight)
            init.zeros_(self.net[0].layer.bias)

            for layer in self.net[1:]:
                if isinstance(layer, MLPLayer):
                    init.xavier_uniform_(layer.layer.weight)
                    if layer.layer.bias is not None:
                        init.zeros_(layer.layer.bias)
                elif isinstance(layer, nn.Linear):
                    init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        init.zeros_(layer.bias) 

        else: return 

    
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

class SIREN(nn.Module):
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
        super(SIREN, self).__init__()

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
    

def train(
        x : torch.tensor, 
        y : torch.tensor, 
        model : nn.Module,
        optimizer : torch.optim.Optimizer,
        loss_fn : callable, 
        epochs : int, 
        batch_size : int, 
        validation_data: Union[None, Tuple[torch.tensor, torch.tensor]] = None, 
        device : torch.device = torch.device("cpu"),
        **kwargs
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:


    dataloader = torch.utils.data.DataLoader(list(zip(x, y)), batch_size= batch_size, shuffle=True)

    losses_train = np.zeros((epochs, len(dataloader)))
    losses_val = np.zeros(epochs) if validation_data is not None else None

    model.train()
    for epoch in range(epochs):
        for i, (x_batch, y_batch) in enumerate(dataloader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            outputs = model(x_batch)
            loss = loss_fn(outputs, y_batch)
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses_train[epoch, i] = loss.item()

        if validation_data is not None:
            with torch.no_grad():
                model.eval()
                x_val, y_val = validation_data
                x_val, y_val = x_val.to(device), y_val.to(device)
                outputs = model(x_val)
                loss = loss_fn(outputs, y_val)
                losses_val[epoch] = loss.item()
                model.train()


        if validation_data is not None:
            print(f'Epoch [{epoch + 1}/{epochs}], training_loss: {losses_train[epoch].mean():.4f}, val_loss: {losses_val[epoch]:.4f}', end = '\r')
        else :
            print(f'Epoch [{epoch + 1}/{epochs}], training_loss: {losses_train[epoch].mean():.4f}', end = '\r')

    return losses_train, losses_val if validation_data is not None else losses_train