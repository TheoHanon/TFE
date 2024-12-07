# SPDX-FileCopyrightText: Copyright (c) 2022 The torch-harmonics Authors. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause


import numpy as np
import matplotlib.pyplot as plt
import cartopy
import cartopy.crs as ccrs

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