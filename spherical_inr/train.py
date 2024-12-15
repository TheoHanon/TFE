import torch
from typing import Union, Tuple
import torch.nn as nn
import numpy as np


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
        log_interval: int = 1,
        **kwargs
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:


    dataloader = torch.utils.data.DataLoader(list(zip(x, y)), batch_size= batch_size, shuffle=True)

    losses_train = np.zeros(epochs)
    losses_val = np.zeros(epochs) if validation_data is not None else None

    model.train()
    
    for epoch in range(epochs):
        epoch_loss = 0.0
        for i, (x_batch, y_batch) in enumerate(dataloader):
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            
            outputs = model(x_batch)
            loss = loss_fn(outputs, y_batch)
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        losses_train[epoch] = epoch_loss / len(dataloader)

        if validation_data is not None:
            with torch.no_grad():
                model.eval()
                x_val, y_val = validation_data
                x_val, y_val = x_val.to(device), y_val.to(device)
                outputs = model(x_val)
                loss = loss_fn(outputs, y_val)
                losses_val[epoch] = loss.item()
                model.train()


        if (epoch + 1) % log_interval == 0:
            if validation_data is not None:
                print(
                    f"Epoch [{epoch + 1}/{epochs}], "
                    f"Training Loss: {losses_train[epoch]:.4f}, "
                    f"Validation Loss: {losses_val[epoch]:.4f}", 
                    end = "\r"
                )
            else:
                print(f"Epoch [{epoch + 1}/{epochs}], Training Loss: {losses_train[epoch]:.4f}", end = "\r")


    return (losses_train, losses_val) if validation_data is not None else losses_train