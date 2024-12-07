import sys
sys.path.append('/Users/theohanon/Desktop/Master II/TFE/Experiences/SAMPTA')

import numpy as np
import pyshtools
from inr import SphericalSiren, train
from utils import plot_SHT_coeffs
from chebychev import Chebyshev
from sampling import sample_s2
from plotting import plot_sphere


import torch_harmonics as th
import torch
from sklearn.preprocessing import StandardScaler


L = 150
coefficients = pyshtools.datasets.Earth.Earth2014.tbi() 
world_map = np.flipud(coefficients.expand(lmax=120).to_array())

scaler = StandardScaler()
world_map_scaled = scaler.fit_transform(world_map)

phi, theta, (nlon, nlat) = sample_s2(L, sampling = "gl", torch_tensor = True)

X = torch.stack([theta.flatten(), phi.flatten()], dim=-1).float()
y = torch.tensor(world_map_scaled.copy()).flatten().unsqueeze(1).float()

def m2(x):
    return 1- 1/2 * x**2

sh_siren = SphericalSiren(L0 = 4, Q = 3, hidden_features = 40, activation = Chebyshev(order = 4, alpha = 1.0), first_activation = Chebyshev(order = 4, alpha = 5.0))

train(
    x = X,
    y = y,
    model = sh_siren,
    loss_fn = torch.nn.MSELoss(),
    optimizer = torch.optim.Adam(sh_siren.parameters(), lr=1e-3),
    epochs = 2000,
    batch_size = 1024,
)

y_pred_scaled = sh_siren(X).clone().detach().numpy().reshape(nlat, nlon)
y_pred = scaler.inverse_transform(y_pred_scaled)

plot_earth_topography(y_pred)
plot_earth_topography(world_map)

sht = th.RealSHT(nlat=nlat, nlon=nlon, lmax=180, mmax=180)
coeffs1 = sht(torch.tensor(y_pred_scaled)).numpy()
coeffs2 = sht(torch.tensor(world_map_scaled.copy())).numpy()

max_coeff = max(np.max(np.abs(coeffs1)), np.max(np.abs(coeffs2)))
min_coeff = min(np.min(np.abs(coeffs1[np.abs(coeffs1) > 0])), np.min(np.abs(coeffs1[np.abs(coeffs1) > 0])))


plot_SHT_coeffs(coeffs1, vmin = np.log(min_coeff), vmax = np.log(max_coeff), ticks_m=20)
plot_SHT_coeffs(coeffs2, vmin = np.log(min_coeff), vmax = np.log(max_coeff), ticks_m=20)
