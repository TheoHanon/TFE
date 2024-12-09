import numpy as np
import math
import torch

import torch_harmonics as th
from inr import SphericalINR, train
from chebychev import Chebyshev
from sampling import sample_s2
from plotting import plot_sphere

from spherical_harmonics_ylm import get_SH
from utils import plot_SHT_coeffs, plot_max_SHT_coeffs, plot_losses

    
l_freq = 23
coeffs = np.random.randn(l_freq+1)

def create_composite_function(l_freq, coeffs):
    # Precompute harmonic functions weighted by coefficients
    components = []
    
    for m in range(-l_freq, l_freq + 1):
            coeff = coeffs[m]
            harmonic_func = get_SH(m, l_freq)
            components.append(lambda theta, phi, c=coeff, f=harmonic_func: c * f(theta, phi))

    # Combine into a single function
    def composite_function(theta, phi):
        return sum(func(theta, phi) for func in components)

    return composite_function

f_target = create_composite_function(l_freq, coeffs)


### Coarse gird

L_coarse = 40
phi_coarse, theta_coarse, (nlon_coarse, nlat_coarse) = sample_s2(L_coarse, sampling = "gl", torch_tensor = True)
sht_coarse = th.RealSHT(nlat=nlat_coarse, nlon=nlon_coarse, lmax=L_coarse, mmax=L_coarse, grid = "legendre-gauss")

X_coarse = torch.stack([theta_coarse.flatten(), phi_coarse.flatten()], axis=-1).float()
y_coarse = f_target(theta_coarse.flatten(), phi_coarse.flatten()).unsqueeze(1).float()

coeffs_coarse = sht_coarse(y_coarse.reshape(nlat_coarse, nlon_coarse)).numpy()

plot_SHT_coeffs(coeffs_coarse, save_path="figures_aliasing/sh_coeffs_gt_coarse")

### Fine grid

L_fine = 150
phi_fine, theta_fine, (nlon_fine, nlat_fine) = sample_s2(L_fine, sampling = "gl", torch_tensor = True)
sht_fine = th.RealSHT(nlat=nlat_fine, nlon=nlon_fine, lmax=L_fine, mmax=L_fine, grid = "legendre-gauss")

X_fine = torch.stack([theta_fine.flatten(), phi_fine.flatten()], dim=-1).float()
y_fine = f_target(theta_fine.flatten(), phi_fine.flatten()).unsqueeze(1).float()

coeffs_fine = sht_fine(y_fine.reshape(nlat_fine, nlon_fine)).numpy()

# plot_SHT_coeffs(coeffs_fine)
plot_SHT_coeffs(coeffs_fine, save_path="figures_aliasing/sh_coeffs_gt_fine", ticks_l=20)


## Train SIREN for increasing alpha


activation_degrees = [(2, 2), (10, 2), (20, 2), (4, 4), (10, 4), (20, 4), (10, 10), (20, 20)]
dict_coeffs = {}

torch.manual_seed(42)
for (degree_first, degree_subseq) in activation_degrees:
    # theoretical expansion = 5 * alpha ** (2)
    sh_siren = SphericalINR(
         L0 = 10, 
         Q = 2, 
         hidden_features = 50, 
         activation =  Chebyshev(order = degree_subseq, alpha = 1.0),
         first_activation = Chebyshev(order = degree_first, alpha = 1.0)
    )

    train(
        x = X_coarse,
        y = y_coarse,
        model = sh_siren,
        loss_fn = torch.nn.MSELoss(),
        optimizer = torch.optim.Adam(sh_siren.parameters(), lr=1e-3),
        epochs = 500,
        batch_size = 128,
    )

    y_pred_fine = sh_siren(X_fine).clone().detach().reshape(nlat_fine, nlon_fine)
    coeffs_pred_fine = sht_fine(y_pred_fine).numpy()
    dict_coeffs[r"$\sigma_0(x) = T_{%.d}(x)$; $\sigma(x) = T_{%.d}(x) $"%(degree_first, degree_subseq)] = coeffs_pred_fine

    # plot_SHT_coeffs(coeffs_pred_fine)


sh_siren = SphericalINR(
         L0 = 10, 
         Q = 2, 
         hidden_features = 50, 
         activation =  torch.sin,
         first_activation = torch.sin
    )

train(
    x = X_coarse,
    y = y_coarse,
    model = sh_siren,
    loss_fn = torch.nn.MSELoss(),
    optimizer = torch.optim.Adam(sh_siren.parameters(), lr=1e-3),
    epochs = 500,
    batch_size = 128,
)

y_pred_fine = sh_siren(X_fine).clone().detach().reshape(nlat_fine, nlon_fine)
coeffs_pred_fine = sht_fine(y_pred_fine).numpy()
dict_coeffs[r"$\sigma_0(x) = \sin(5x)$; $\sigma(x) = \sin(x) $"] = coeffs_pred_fine



## Plot results

plot_max_SHT_coeffs(dict_coeffs, legend_title="Activation function", save_path="figures_aliasing/sh_coeffs_alpha", ticks_l=20)


## Train SIREN for fixed alpha and increasing coeff0

alphas = [(1/5, 1), (1/2, 1), (1, 1), (2, 1), (5, 1), (10,1), (1, 1/2), (1, 1), (1, 2), (1, 5), (1, 10)]
dict_coeffs = {}
dict_losses = {}

torch.manual_seed(42)
for (alpha_first, alpha_subseq) in alphas:
    # theoretical expansion = 5 * alpha ** (2)
    sh_siren = SphericalINR(L0 = 10, 
                            Q = 2, 
                            hidden_features = 50, 
                            activation = Chebyshev(order=10, alpha = alpha_subseq), 
                            first_activation = Chebyshev(order = 10, alpha = alpha_first))
    
    loss_train = train(
        x = X_coarse.clone().detach(),
        y = y_coarse.clone().detach(),
        model = sh_siren,
        loss_fn = torch.nn.MSELoss(),
        optimizer = torch.optim.Adam(sh_siren.parameters(), lr=1e-3),
        epochs = 500,
        batch_size = 128,
    )

    y_pred_fine = sh_siren(X_fine).clone().detach().reshape(nlat_fine, nlon_fine)
    coeffs_pred_fine = sht_fine(y_pred_fine).numpy()
    dict_coeffs[r"$\sigma_0(x) = T_{5}(\frac{x}{%.1f}); \sigma(x) = T_{5}(\frac{x}{%.1f})$" % (alpha_first, alpha_subseq)] = coeffs_pred_fine
    dict_losses[r"$\sigma_0(x) = T_{5}(\frac{x}{%.1f}); \sigma(x) = T_{5}(\frac{x}{%.1f})$" % (alpha_first, alpha_subseq)] = loss_train


sh_siren = SphericalINR(L0 = 10, 
                            Q = 2, 
                            hidden_features = 50, 
                            activation = torch.sin, 
                            first_activation = torch.sin)
    
loss_train = train(
    x = X_coarse.clone().detach(),
    y = y_coarse.clone().detach(),
    model = sh_siren,
    loss_fn = torch.nn.MSELoss(),
    optimizer = torch.optim.Adam(sh_siren.parameters(), lr=1e-3),
    epochs = 500,
    batch_size = 128,
)

y_pred_fine = sh_siren(X_fine).clone().detach().reshape(nlat_fine, nlon_fine)
coeffs_pred_fine = sht_fine(y_pred_fine).numpy()
dict_coeffs[r"$\sigma_0(x) = \sin(5x); \sigma(x) = \sin(x)$"] = coeffs_pred_fine
dict_losses[r"$\sigma_0(x) = \sin(5x); \sigma(x) = \sin(x)$"] = loss_train


## Plot results

plot_max_SHT_coeffs(dict_coeffs, save_path="figures_aliasing/sh_coeffs_w0", ticks_l=20)
plot_losses(dict_losses, save_path="figures_aliasing/losses", xlabel="Epochs", ylabel="MSE Loss")








