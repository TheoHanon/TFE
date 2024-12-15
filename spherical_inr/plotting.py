


import numpy as np
import torch

import matplotlib.pyplot as plt
from matplotlib import cm

import cartopy
import cartopy.crs as ccrs


### COPY FROM :
# SPDX-FileCopyrightText: Copyright (c) 2022 The torch-harmonics Authors. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
def plot_sphere(data,
                fig=None,
                cmap="RdBu",
                title=None,
                colorbar=False,
                coastlines=False,
                central_latitude=20,
                central_longitude=20,
                molweide=False,
                lon=None,
                lat=None,
                **kwargs):
    if fig == None:
        fig = plt.figure()

    nlat = data.shape[-2]
    nlon = data.shape[-1]

    if lon is None:
        lon = np.linspace(0, 2*np.pi, nlon)
    if lat is None:
        lat = np.linspace(np.pi/2., -np.pi/2., nlat)
    Lon, Lat = np.meshgrid(lon, lat)

    if molweide:
        proj = ccrs.Mollweide(central_longitude=central_longitude)
    else:
        proj = ccrs.Orthographic(central_longitude=central_longitude, central_latitude=central_latitude)
    
    ax = fig.add_subplot(projection=proj)
    Lon = Lon*180/np.pi
    Lat = Lat*180/np.pi

    # contour data over the map.
    im = ax.pcolormesh(Lon, Lat, data, cmap=cmap, transform=ccrs.PlateCarree(), antialiased=False, **kwargs)
    if coastlines:
        ax.add_feature(cartopy.feature.COASTLINE, edgecolor='k', facecolor='none', linewidth=1.5)
    if colorbar:
        plt.colorbar(im)
    plt.title(title, y=1.05)

    return im


def plot_sphere_scatter(data,
                        lon,
                        lat,
                        fig=None,
                        cmap="RdBu",
                        title=None,
                        colorbar=False,
                        coastlines=False,
                        central_latitude=20,
                        central_longitude=20,
                        molweide=False,
                        **kwargs):
    
    if fig == None:
        fig = plt.figure()

    Lon = lon * 180/np.pi
    Lat = 90 - lat * 180/np.pi


    if molweide:
        proj = ccrs.Mollweide(central_longitude=central_longitude)
    else:
        proj = ccrs.Orthographic(central_longitude=central_longitude, central_latitude=central_latitude)
    
    ax = fig.add_subplot(projection=proj)


    # contour data over the map.
    im = ax.scatter(Lon, Lat, c=data, cmap=cmap, transform=ccrs.PlateCarree(), **kwargs)


    if coastlines:
        ax.add_feature(cartopy.feature.COASTLINE, edgecolor='k', facecolor='none', linewidth=1.5)
    if colorbar:
        plt.colorbar(im)
    plt.title(title, y=1.05)

    return im






def plot_SHT_coeffs(coeffs, 
                    ticks_m=10, 
                    ticks_l=10, 
                    fig=None, 
                    title=None,
                    colorbar=False,
                    **kwargs):
    """
    Plots the spherical harmonic transform (SHT) coefficients with improved visuals.

    Parameters:
    - coeffs (2D array): Coefficients matrix to plot.
    - ticks_m (int): Step for m-axis ticks.
    - ticks_l (int): Step for l-axis ticks.
    - fig (matplotlib.figure.Figure): Optional pre-existing figure.
    - title (str): Title of the plot.
    - colorbar (bool): Whether to include a colorbar.
    - kwargs: Additional arguments passed to `imshow`.
    """
    coeffs = coeffs.copy()

    mask = (np.abs(coeffs) == 0)
    coeffs[mask] = 1.0

    # Create figure and axis
    if fig is None:
        fig = plt.figure(figsize=(8, 6))

    log_coeffs = np.log(np.abs(coeffs))
    log_coeffs[mask] = np.nan

    ax = fig.add_subplot()
    # Plot log of the absolute values of coefficients
    im = ax.imshow(
        log_coeffs,  
        origin="lower", 
        extent=[0, coeffs.shape[1], 0, coeffs.shape[0]],  
        aspect='equal',  
        **kwargs
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
    if colorbar:
        cbar = fig.colorbar(im, ax=ax, shrink=0.7, aspect=20, pad=0.05, orientation='horizontal')
        cbar.set_label('Log SHT Coeffs', fontsize=12)

    if title:
        plt.title(title, y=1.05)

    return im

def plot_max_SHT_coeffs(coeffs_dict, 
                        ticks_l=1, 
                        fig=None, 
                        title=None,
                        legend_title=None, 
                        **kwargs):
    """
    Plots intervals (min-max range) of SHT coefficients for each \ell across multiple signals with distinct line styles.

    Parameters:
    - coeffs_dict (dict): Dictionary where keys are signal names and values are 2D arrays (or lists) of coefficients.
                          Rows correspond to \ell, columns to m for each signal.
    - ticks_l (int): Step size for \ell-axis ticks.
    - fig (matplotlib.figure.Figure): Optional pre-existing figure.
    - title (str): Title of the plot.
    - legend_title (str): Title for the legend.
    - kwargs: Additional arguments passed to `plot`.
    """
    if not isinstance(coeffs_dict, dict):
        coeffs_dict = {"Signal": coeffs_dict}

    if fig is None:
        fig = plt.figure(figsize=(8, 6))

    ax = fig.add_subplot()

    line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 5)), (0, (3, 5, 1, 5))]
    for i, (label, coeffs) in enumerate(coeffs_dict.items()):
        max_coeffs = np.max(np.abs(coeffs), axis=1)
        l_values = np.arange(len(max_coeffs))

        ax.plot(
            l_values,
            max_coeffs,
            linestyle=line_styles[i % len(line_styles)],
            label=label,
            marker="o",
            markersize=4,
            **kwargs
        )

    ax.set_xlabel(r"$\ell$", fontsize=16)
    ax.set_ylabel("Max Coefficient", fontsize=16)
    ax.set_xticks(np.arange(0, len(l_values), ticks_l))
    ax.grid(True, linestyle='--', alpha=0.6, which='both')


    ax.legend(title = legend_title, fontsize=12)

    if title:
        ax.set_title(title, fontsize=18)

    return fig


def plot_losses(
    loss_dict: dict,
    title: str = None,
    ylabel: str = "Loss",
    fig=None,
    **kwargs
):
    """
    Plot losses for multiple models with distinct colors and line styles.

    Parameters:
    - loss_dict (dict): A dictionary where keys are model names, and values are lists of losses.
                        Example:
                        {
                            "Model A": losses_a,
                            "Model B": losses_b
                        }
    - title (str): Title of the plot.
    - ylabel (str): Label for the y-axis.
    - fig (matplotlib.figure.Figure): Optional pre-existing figure.
    - kwargs: Additional arguments passed to `plot`.
    """
    if fig is None:
        fig = plt.figure(figsize=(12, 8))

    ax = fig.add_subplot()

    colors = plt.cm.tab10.colors
    line_styles = ['-', '--', '-.', ':', (0, (3, 1, 1, 1)), (0, (5, 5)), (0, (3, 5, 1, 5))]

    for i, (model_name, losses) in enumerate(loss_dict.items()):
        color = colors[i % len(colors)]
        line_style = line_styles[i % len(line_styles)]

        ax.plot(losses, label=model_name, color=color, linestyle=line_style, linewidth=2, **kwargs)

    if title:
        ax.set_title(title, fontsize=16)

    ax.set_xlabel("Epochs", fontsize=14)
    ax.set_ylabel(ylabel, fontsize=14)
    ax.legend(fontsize=12)
    ax.set_yscale("log")
    ax.grid(True)

    return fig
