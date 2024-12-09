import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
import torch


class PAlphaActivation(torch.nn.Module):
    def __init__(self, alpha=10, coeff0=1):
        super(PAlphaActivation, self).__init__()
        self.alpha = alpha
        self.coeff0 = coeff0

    def forward(self, x):
        out = torch.sin(x * self.coeff0)
        return out

def plot_earth_topography(grid, colormap=cm.RdBu_r, earth_radius=6371e3, save_path = None):
    """
    Plots Earth's topography on a 3D sphere.

    Parameters:
    - grid (numpy.ndarray): Topography grid (elevation values in meters).
    - colormap (matplotlib.colors.Colormap): Colormap for topography visualization.
    - earth_radius (float): Mean radius of Earth in meters. Defaults to 6371 km.

    Returns:
    - None
    """
    # Extract latitude, longitude, and elevation
    nlat, nlon = grid.shape
    lats = np.linspace(-90, 90, nlat)  # Latitude values (degrees)
    lons = np.linspace(0, 360, nlon)  # Longitude values (degrees)

    # Convert lat/lon to spherical coordinates
    lon_radians = np.radians(lons)
    lat_radians = np.radians(lats)

    # Create a 2D mesh for lat/lon
    lon_mesh, lat_mesh = np.meshgrid(lon_radians, lat_radians)

    # Define Earth's radius (mean radius + topography)
    radius = earth_radius + grid  # Add topography to radius

    # Convert to Cartesian coordinates
    x = radius * np.cos(lat_mesh) * np.cos(lon_mesh)
    y = radius * np.cos(lat_mesh) * np.sin(lon_mesh)
    z = radius * np.sin(lat_mesh)

    # Normalize topography data for color mapping
    normalized_grid = (grid - np.min(grid)) / (np.max(grid) - np.min(grid))

    # Create a 3D plot
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Plot topography on the sphere
    ax.plot_surface(
        x, y, z, facecolors=colormap(normalized_grid),
        rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False
    )

    # Add colorbar for elevation
    mappable = cm.ScalarMappable(cmap=colormap)
    mappable.set_array(grid)  # Associate the grid data with the colormap
    cbar = fig.colorbar(mappable, ax=ax, shrink=0.6, aspect=10, pad=0.1)
    cbar.set_label('Elevation (meters)')

    # Adjust aspect ratio and remove axis
    ax.set_box_aspect([1, 1, 1])  # Equal aspect ratio
    ax.axis('off')
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path + ".pdf")
    # Show plot
    plt.show()



def plot_SHT_coeffs(coeffs, ticks_m=10, ticks_l=10, cmap="RdBu_r", vmin = None, vmax = None, save_path = None):
    """
    Plots the spherical harmonic transform (SHT) coefficients with improved visuals.

    Parameters:
    - coeffs (2D array): Coefficients matrix to plot.
    - ticks_m (int): Step for m-axis ticks.
    - ticks_l (int): Step for l-axis ticks.
    - cmap (str): Colormap for the plot.
    """
    mask = (np.abs(coeffs) == 0) 
    coeffs[mask] = 1.0

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(6, 6))

    log_coeffs = np.log(np.abs(coeffs)) 
    log_coeffs[mask] = np.nan
    
    # Plot log of the absolute values of coefficients
    im = ax.imshow(
        log_coeffs,  # Use log1p for numerical stability
        cmap=cmap, 
        origin="lower", 
        extent=[0, coeffs.shape[1], 0, coeffs.shape[0]],  # Set extent for equal scaling
        aspect='equal',  # Ensure equal scaling for both axes
        # norm = norm, 
        vmin = vmin, 
        vmax = vmax
    )

    # Set labels and ticks
    ax.set_xlabel(r"$m$", fontsize=16)
    ax.set_ylabel(r"$\ell$", rotation=0, fontsize=16, labelpad=10)
    ax.set_xticks(np.arange(0, coeffs.shape[1] + 1, ticks_m))
    ax.set_xticklabels(np.arange(0, coeffs.shape[1] + 1, ticks_m))
    ax.set_yticks(np.arange(0, coeffs.shape[0] + 1, ticks_l))
    ax.set_yticklabels(np.arange(0, coeffs.shape[0] + 1, ticks_l))

    # Move x-axis label and ticks to the top
    ax.xaxis.set_label_position("top")
    ax.xaxis.tick_top()

    # Remove the bottom axis
    ax.spines['bottom'].set_visible(False)
    ax.spines['right'].set_visible(False)
    

    # Add colorbar
    cbar = fig.colorbar(im, ax=ax, shrink=0.7, aspect=20, pad=0.05, orientation='horizontal')
    cbar.set_label('Log SHT Coeffs', fontsize=12)
    
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path + ".pdf", dpi = 300)
    else :
        plt.show()
    # plt.close(fig)



def plot_max_SHT_coeffs(coeffs_dict, ticks_l=1, legend_title = None, save_path=None):
    """
    Plots intervals (min-max range) of SHT coefficients for each l across multiple signals with distinct line styles.

    Parameters:
    - coeffs_dict (dict or 2D array): Dictionary where keys are signal names and values are 2D arrays (or lists) of coefficients.
                                      Rows correspond to l, columns to m for each signal.
    - ticks_l (int): Step size for l-axis ticks.
    - save_path (str): Path to save the plot (without extension). If None, the plot is not saved.
    """
    # Prepare the plot
    fig, ax = plt.subplots(figsize=(8, 6))

    if not isinstance(coeffs_dict, dict):
        # Single signal case
        coeffs = coeffs_dict
        max_coeffs = np.max(np.abs(coeffs), axis=1)
        l_values = np.arange(len(max_coeffs))
        
        # Plot the max coefficients
        ax.plot(
            l_values, 
            max_coeffs,
            linestyle="-",  # Default style for a single signal
            marker="o"
        )
    else:
        # Process multiple signals
        line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 5)), (0, (3, 5, 1, 5))]
        for i, (label, coeffs) in enumerate(coeffs_dict.items()):
            if isinstance(coeffs, np.ndarray) or isinstance(coeffs, list):
                max_coeffs = np.max(np.abs(coeffs), axis=1)[:-1]
                l_values = np.arange(len(max_coeffs))
            else:
                raise ValueError(f"Unsupported coefficient format for signal '{label}'. Expected 2D array or list.")
            
            # Plot max coefficients with distinct line styles
            ax.plot(
                l_values, 
                max_coeffs,
                linestyle=line_styles[i % len(line_styles)],  # Cycle through line styles
                label=label,
                marker="o",
                markersize=4
            )

    # Customize the plot
    ax.set_xlabel(r"$\ell$", fontsize=16)
    ax.set_ylabel("Max Coefficient", fontsize=16)
    ax.set_xticks(np.arange(0, len(l_values), ticks_l))
    ax.grid(True, linestyle='--', alpha=0.6, which='both')
    

    if isinstance(coeffs_dict, dict):
        ax.legend(title=legend_title, fontsize=12)

    # Add a title and adjust layout
    ax.set_title("Coefficient Ranges for Each $\ell$", fontsize=18)
    plt.tight_layout()
    
    # Save the plot if a path is specified
    if save_path is not None:
        plt.savefig(save_path + ".pdf", dpi=300)
    else:
        plt.show()



def plot_losses(
    loss_dict: dict,
    title: str = "Training and Validation Losses",
    xlabel: str = "Epochs",
    ylabel: str = "Loss",
    save_path: str = None
):
    """
    Plot training and validation losses for multiple models.

    Parameters:
    - loss_dict (dict): A dictionary where keys are model names, and values are either tuples of
                        (losses_train, losses_val) or just losses_train. If losses_val is None, only training loss is plotted.
                        Example:
                        {
                            "Model A": (losses_train_a, losses_val_a),
                            "Model B": losses_train_b
                        }
    - title (str): Title of the plot.
    - xlabel (str): Label for the x-axis.
    - ylabel (str): Label for the y-axis.
    """
    fig = plt.figure(figsize=(12, 8))

    for model_name, losses in loss_dict.items():
        if isinstance(losses, tuple):
            losses_train, losses_val = losses
        else:
            losses_train = losses
            losses_val = None

        plt.plot(losses_train, label=f"{model_name} - Training", linewidth=2)
        if losses_val is not None:
            plt.plot(losses_val, label=f"{model_name} - Validation", linestyle="--", linewidth=2)

    plt.title(title, fontsize=16)
    plt.xlabel(xlabel, fontsize=14)
    plt.ylabel(ylabel, fontsize=14)
    plt.legend(fontsize=12)
    plt.yscale("log")
    plt.grid(True)
    if save_path is not None:
        plt.savefig(save_path + ".pdf", dpi=300)
    else:
        plt.show()






