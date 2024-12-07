import torch 
import numpy as np
import matplotlib.pyplot as plt
from inr import SphericalSiren, train



def compare_initialization(n_run, training_data, validation_data, network_params, training_params, **kwargs): 

    X_train, y_train = training_data
    
    inits = ['random', 'xavier', 'kaiming', 'laurent_xavier', 'laurent_kaiming']

    train_losses_dict = {init : [] for init in inits}
    val_losses_dict = {init : [] for init in inits}

    for _ in range(n_run):

        models = [SphericalSiren(**network_params, init = init, device = training_params['device']) for init in inits]

        for iInit, model in enumerate(models):

            optimizer = torch.optim.Adam(model.parameters(), **kwargs)
            train_loss, val_loss = train(X_train, y_train, model, validation_data = validation_data, optimizer = optimizer, **training_params)

            train_losses_dict[inits[iInit]].append(train_loss)
            val_losses_dict[inits[iInit]].append(val_loss)

    return train_losses_dict, val_losses_dict