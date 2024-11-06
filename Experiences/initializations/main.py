import yaml
import argparse
import os

import numpy as np
import torch
import torch.nn as nn
from initialization_utils import *
from inr import get_activation
from spherical_harmonics_ylm import get_SH

parser = argparse.ArgumentParser()
parser.add_argument("-exp", type=str, default="Init1", help="Path to the config file")


def loss_fn(outputs, targets):
    return torch.mean((outputs.squeeze() - targets)**2)

# Experiment 1: Initialization training_params


def main(parser):

    with open("configs/" + parser.exp + ".yaml", "r") as ymlfile:
        config = yaml.safe_load(ymlfile)

    if not os.path.exists(config["EXPERIMENT_PARAMS"]["results_path"] + parser.exp):
        os.makedirs(config["EXPERIMENT_PARAMS"]["results_path"] + parser.exp)
        
    save_path = config["EXPERIMENT_PARAMS"]["results_path"] + parser.exp + "/"

    if config["EXPERIMENT_PARAMS"]["target"] == "cartoon_shape":

        def f(theta, phi):
            mask = phi < np.pi
            out = torch.zeros_like(theta)
            out[~mask] = 1.0
            return out
        
    else :
        # TODO: Add other targets
        # def create_target(Y):
        #     coeff = np.random.randn(len(Y))
        #     def target(theta, phi):
        #         return sum(Y[i](theta, phi)*coeff[i] for i in range(len(Y)))
        #     return target

        raise ValueError("Invalid target")
    
    training_params = config["TRAINING_PARAMS"]
    training_params["loss_fn"] = loss_fn

    network_params = config["NETWORK_PARAMS"]
    network_params["activation"] = get_activation(network_params["activation"])

    N = config["EXPERIMENT_PARAMS"]["N"]

    theta, phi = torch.meshgrid(torch.linspace(0, np.pi, N), torch.linspace(0, 2*np.pi, N), indexing='ij')
    X_data = torch.stack([theta.flatten(), phi.flatten()], dim=-1)
    y_data = f(theta.flatten(), phi.flatten())

    train_loss_dict, val_loss_dict = compare_initialization(
        n_run=config["EXPERIMENT_PARAMS"]["n_run"],
        training_data=(X_data, y_data),
        validation_data=(X_data, y_data),
        training_params=training_params,
        network_params=network_params,
        lr=config["EXPERIMENT_PARAMS"]["lr"],
    )

    torch.save({'train_loss_dict': train_loss_dict, 'val_loss_dict': val_loss_dict}, save_path + "losses.pth")
    

# Run both experiments
if __name__ == "__main__":
    parser = parser.parse_args()
    main(parser)
