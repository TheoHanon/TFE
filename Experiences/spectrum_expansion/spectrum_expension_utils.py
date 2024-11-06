import torch 
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def show_learning_curve(train_losses, val_losses, log = False):

    mean_train_losses = np.mean(train_losses, axis=1)
    std_train_losses = np.std(train_losses, axis=1)

    plt.figure(figsize=(10, 5))

    if log :
        plt.loglog(mean_train_losses, label='train loss', marker = 'o', fillstyle='none')
        plt.loglog(val_losses, label='validation loss', marker = 'o', fillstyle='none')  
    else:
        plt.plot(mean_train_losses, label='train loss', marker = 'o', fillstyle='none')
        plt.plot(val_losses, label='validation loss', marker = 'o', fillstyle='none')
        plt.fill_between(range(len(mean_train_losses)), mean_train_losses - std_train_losses, mean_train_losses + std_train_losses, alpha=0.2)
        
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.grid(True, which='both', linestyle='--', linewidth=0.5, alpha = 0.4)
    plt.legend()
    plt.show()


def show_comparisons(X_plot, y_plot, y_pred):
    theta_plot, phi_plot = X_plot[:, 0].clone().detach().numpy(), X_plot[:, 1].clone().detach().numpy()
    y_plot = y_plot.clone().detach().numpy()
    y_pred = y_pred.clone().detach().numpy()

    x_coords = np.sin(theta_plot) * np.cos(phi_plot)
    y_coords = np.sin(theta_plot) * np.sin(phi_plot)
    z_coords = np.cos(theta_plot)

    fig = plt.figure(figsize=(16, 8))

    # Ground Truth Plot
    ax1 = fig.add_subplot(121, projection='3d')
    norm = plt.Normalize(vmin=np.min(y_plot), vmax=np.max(y_plot))
    scatter1 = ax1.scatter(
        x_coords, y_coords, z_coords,
        c=y_plot, cmap='binary_r', marker='o', s=20, alpha=0.8
    )
    cbar1 = fig.colorbar(scatter1, ax=ax1, shrink=0.6, aspect=10)
    cbar1.set_label('Value')
    ax1.set_xlabel('X', fontsize=12)
    ax1.set_ylabel('Y', fontsize=12)
    ax1.set_zlabel('Z', fontsize=12)
    ax1.set_title('Ground Truth', fontsize=15)
    ax1.view_init(elev=20, azim=60)

    # Prediction Plot
    ax2 = fig.add_subplot(122, projection='3d')
    norm = plt.Normalize(vmin=np.min(y_pred), vmax=np.max(y_pred))
    scatter2 = ax2.scatter(
        x_coords, y_coords, z_coords,
        c=y_pred, cmap='binary_r', marker='o', s=20, alpha=0.8
    )
    cbar2 = fig.colorbar(scatter2, ax=ax2, shrink=0.6, aspect=10)
    cbar2.set_label('Value')
    ax2.set_xlabel('X', fontsize=12)
    ax2.set_ylabel('Y', fontsize=12)
    ax2.set_zlabel('Z', fontsize=12)
    ax2.set_title('Prediction', fontsize=15)
    ax2.view_init(elev=20, azim=60)

    # Set common styling
    for ax in [ax1, ax2]:
        ax.grid(True)
        ax.xaxis.labelpad = 10
        ax.yaxis.labelpad = 10
        ax.zaxis.labelpad = 10

    plt.tight_layout()
    plt.show()
