import numpy as np
from inr import SphericalSiren, train
from chebychev import Chebyshev
from sampling import sample_s2
from spherical_harmonics_ylm import get_SH
from utils import plot_SHT_coeffs

import torch_harmonics as th
import torch
from sklearn.preprocessing import StandardScaler


l_max = 20
L = 40
coeffs = np.random.randn((l_max + 1) ** 2)

def create_composite_function(l_max, coeffs):
    # Precompute harmonic functions weighted by coefficients
    components = []

    for l in range(l_max + 1):
        for m in range(-l, l + 1):
            coeff = coeffs[l * (l + 1) + m]
            harmonic_func = get_SH(m, l)  # This returns a function f(theta, phi)
            # Append the weighted function
            components.append(lambda theta, phi, c=coeff, f=harmonic_func: c * f(theta, phi))

    # Combine into a single function
    def composite_function(theta, phi):
        return sum(func(theta, phi) for func in components)

    return composite_function

f_target = create_composite_function(l_max, coeffs)

phi, theta, (nlon, nlat) = sample_s2(L, sampling = "gl", torch_tensor = True)

X = torch.stack([theta.flatten(), phi.flatten()], axis=-1).float()
y = f_target(theta.flatten(), phi.flatten()).unsqueeze(1).float()

sh_siren = SphericalSiren(L0 = 2, Q = 4, hidden_features = 30, activation = Chebyshev(order = 2))

train(
    x = X,
    y = y,
    model = sh_siren,
    loss_fn = torch.nn.MSELoss(),
    optimizer = torch.optim.Adam(sh_siren.parameters(), lr=1e-3),
    epochs = 1000,
    batch_size = 128,
)

y_pred = sh_siren(X).clone().detach().reshape(nlat, nlon)
y = y.reshape(nlat, nlon)


sht = th.RealSHT(nlat=nlat, nlon=nlon, lmax=L, mmax=L, grid = "legendre-gauss")
coeffs1 = sht(y_pred).numpy()
coeffs2 = sht(y).numpy()

# max_coeff = max(np.max(np.abs(coeffs1)), np.max(np.abs(coeffs2)))
# min_coeff = min(np.min(np.abs(coeffs1[np.abs(coeffs1) > 0])), np.min(np.abs(coeffs1[np.abs(coeffs1) > 0])))

plot_SHT_coeffs(coeffs1, ticks_l=2, ticks_m=5, save_path="figures_spectrum_expansion/coeffs_shsiren")
plot_SHT_coeffs(coeffs2, ticks_l=2, ticks_m=5, save_path="figures_spectrum_expansion/coeffs_gt")
