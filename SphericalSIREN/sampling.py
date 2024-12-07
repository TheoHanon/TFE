def sample_s2(L: int, sampling: str = "gl", torch_tensor : bool = False, meshgrid : bool = True):
    """
    Samples points on the 2-sphere for a given resolution L and sampling type.

    Parameters:
    L (int): Bandwidth of the spherical harmonics.
    sampling (str): Sampling scheme, default is 'gl' (Gauss-Legendre).

    Returns:
    tuple: A tuple containing:
        - phi (numpy.ndarray): Longitudinal angles (azimuth).
        - theta (numpy.ndarray): Latitudinal angles (colatitude).
        - (nlon, nlat) (tuple): Number of longitude and latitude points.
    """
    import jax
    jax.config.update("jax_enable_x64", True)

    from s2fft.sampling.s2_samples import phis_equiang, thetas
    import numpy as np

    if torch_tensor :
        import torch

    phi = phis_equiang(L, sampling=sampling)
    theta = thetas(L, sampling=sampling)
    nlon, nlat = phi.shape[0], theta.shape[0]

    if meshgrid:
        phi, theta = np.meshgrid(phi, theta)
    else:
        phi, theta = phi.flatten(), theta.flatten()
    
    if torch_tensor:
        phi, theta = torch.tensor(phi), torch.tensor(theta)

    return phi, theta, (nlon, nlat)

