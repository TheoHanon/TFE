import yaml
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch_harmonics as th
from spectrum_expension_utils import *
from inr import SphericalSiren, train, get_activation
from spherical_harmonics_ylm import get_SH

parser = argparse.ArgumentParser()
parser.add_argument("-exp", type=str, default="Spec1", help="Path to the config file")

def loss_fn(outputs, targets):
    return torch.mean((outputs.squeeze() - targets)**2)


# Create data

def main(parser):

    with open("configs/" + parser.exp + ".yaml", "r") as ymlfile:
        config = yaml.safe_load(ymlfile)

    if not os.path.exists(config["EXPERIMENT_PARAMS"]["results_path"] + parser.exp):
        os.makedirs(config["EXPERIMENT_PARAMS"]["results_path"] + parser.exp)

    if not os.path.exists((config["EXPERIMENT_PARAMS"]["results_path"] + parser.exp + "/model/")):
        os.makedirs(config["EXPERIMENT_PARAMS"]["results_path"] + parser.exp + "/model/")

    save_path = config["EXPERIMENT_PARAMS"]["results_path"] + parser.exp + "/"
    model_path = config["EXPERIMENT_PARAMS"]["results_path"] + parser.exp + "/model/"
    
    if config["EXPERIMENT_PARAMS"]["target"] == "cartoon_shape":

        def f(theta, phi):
            mask = phi < np.pi
            out = torch.zeros_like(theta)
            out[~mask] = 1.0
            return out
        
    else :

        # def create_target(Y):
        #     coeff = np.random.randn(len(Y))
        #     def target(theta, phi):
        #         return sum(Y[i](theta, phi)*coeff[i] for i in range(len(Y)))
        #     return target
        raise ValueError("Invalid target")

    N = config["EXPERIMENT_PARAMS"]["N"]
    theta, phi = torch.meshgrid(torch.linspace(0, np.pi, N), torch.linspace(0, 2*np.pi, N), indexing='ij')
    X_data = torch.stack([theta.flatten(), phi.flatten()], dim=-1)
    y_data = f(theta.flatten(), phi.flatten())

    network_params = config["NETWORK_PARAMS"]
    network_params["activation"] = get_activation(network_params["activation"])

  
    spherical_siren = SphericalSiren(device = config["TRAINING_PARAMS"]["device"] ,**network_params)

    training_params = config["TRAINING_PARAMS"]

    losses_train, losses_val = train(
        x=X_data,
        y=y_data,
        model=spherical_siren,
        loss_fn=loss_fn,
        optimizer=optim.Adam(spherical_siren.parameters(), lr=training_params["lr"]),
        epochs=training_params["epochs"],
        batch_size=training_params["batch_size"],
        device=training_params["device"],
        validation_data=(X_data, y_data)
    )

    # Save results
    torch.save({'losses_train': losses_train, 'losses_val': losses_val}, save_path+"losses.pth")
    torch.save(spherical_siren.state_dict(), model_path + "model.pth")

    # Compute spectrum
    
    sht = th.RealSHT(nlat=N, nlon=N, lmax=config["EXPERIMENT_PARAMS"]["lmax"] + 1, mmax=config["EXPERIMENT_PARAMS"]["lmax"]+1)
    coeffs = sht(y_data.reshape((N, N)))

    torch.save(coeffs.abs(), save_path + "spectrum_coeffs.pth")

    inside = spherical_siren.forward_inside(X_data)
    coeffs_inside = {}

    for layer, neurons in inside.items():
        neurons = neurons.reshape((-1, N, N))
        coeffs_inside[layer] = []
        for neuron in neurons:
            coeffs_inside[layer].append(sht(neuron).abs())

        coeffs_inside[layer] = torch.max(torch.stack(coeffs_inside[layer]), dim=0).values.cpu()

    torch.save(coeffs_inside, save_path + "spectrum_coeffs_inside.pth")

# Run both experiments
if __name__ == "__main__":
    parser = parser.parse_args()
    main(parser)
