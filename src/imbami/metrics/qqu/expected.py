"""
Conditional-mean estimator of E[S_err(e) | sigma] and its reference geometry.

The kernel weights are shared between the estimate used for expected QQU and the
reference law that renders it interpretable, so the correction can never be based
on a different bandwidth, kernel or window than the estimate it corrects.

Reference geometry
------------------
With row-normalized weights
kappa_ij = K_h(sigma_i - sigma_j) / sum_l K_h(sigma_i - sigma_l)
and a_ij = S_unc(sigma_j) / S_unc(sigma_i), the estimator's ratio under correct
tracking is

    r^E_i = sum_j kappa_ij a_ij W_j,     W_j = S_err(e_j) / S_unc(sigma_j),

where W has mean mu_ref and variance Var_ref under the Gaussian reference. Hence

    mean(r^E_i) = mu_ref * mean_factor_i,  mean_factor_i = sum_j kappa_ij a_ij
    var (r^E_i) = Var_ref * var_factor_i,  var_factor_i  = sum_j kappa_ij^2 a_ij^2

Both factors equal the locally-homogeneous values 1 and 1/n_eff_i only if sigma is
constant inside the window, which at any realistic bandwidth -- a kernel bandwidth
is proportional to the spread of sigma -- it is not.
"""
from __future__ import annotations

from typing import Literal

import numpy as np

from .utils import as_1d_float, check_choice, silverman_bandwidth

KernelType = Literal["gaussian", "uniform"]

def kernel_conditional_mean(
    sigma: np.ndarray,
    values: np.ndarray | None = None,
    *,
    scale: np.ndarray | None = None,
    bandwidth: float | None = None,
    bandwidth_factor: float = 1.0,
    kernel_type: KernelType = "gaussian",
    chunk_size: int = 1024,
) -> tuple[np.ndarray | None, np.ndarray, np.ndarray, np.ndarray]:
    """
    Kernel (Nadaraya-Watson) average of ``values`` over neighbours in ``sigma``,
    together with the reference geometry of the very same kernel weights.

    Parameters
    ----------
    sigma : array-like, shape (N,)
        Predicted uncertainties, strictly positive.
    values : array-like, shape (N,), optional
        Per-sample residual statistic S_err(e_j), e.g. e_j^2 or |e_j|. If None,
        only the reference geometry is computed.
    scale : array-like, shape (N,), optional
        S_unc(sigma_i), so that a_ij = scale_j / scale_i. If None, a_ij = 1 is
        assumed, which yields the locally-homogeneous factors 1 and 1/n_eff.
    bandwidth : float, optional
        Kernel bandwidth; Silverman's rule of thumb on sigma if None.
    bandwidth_factor : float, default 1.0
        Multiplies the provided or calculated bandwidth.
    kernel_type : {"gaussian", "uniform"}, default "gaussian"
    chunk_size : int, default 1024
        Kernel rows evaluated at once: runtime O(N^2), memory O(chunk_size * N).

    Returns
    -------
    est : np.ndarray, shape (N,) or None
        sum_j kappa_ij values_j (None if ``values`` is None).
    n_eff : np.ndarray, shape (N,)
        1 / sum_j kappa_ij^2, the effective pool size of each row.
    mean_factor : np.ndarray, shape (N,)
        sum_j kappa_ij a_ij.
    var_factor : np.ndarray, shape (N,)
        sum_j kappa_ij^2 a_ij^2.
    """
    sigma = as_1d_float(sigma, "sigma", positive=True)
    n = sigma.size
    values = None if values is None else as_1d_float(values, "values", n)
    if scale is not None:
        scale = as_1d_float(scale, "scale", n, positive=True)
    check_choice(kernel_type, ("gaussian", "uniform"), "kernel_type")
    if chunk_size < 1:
        raise ValueError(f"chunk_size must be >= 1, got {chunk_size}.")

    bandwidth = silverman_bandwidth(sigma) if bandwidth is None else float(bandwidth)
    bandwidth = bandwidth * float(bandwidth_factor)
    if not np.isfinite(bandwidth) or bandwidth <= 0:
        raise ValueError(f"bandwidth must be positive and finite, got {bandwidth}.")

    est = None if values is None else np.empty(n, dtype=float)
    n_eff = np.empty(n, dtype=float)
    mean_factor = np.ones(n, dtype=float)
    var_factor = np.empty(n, dtype=float)

    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        diff = sigma[start:stop, None] - sigma[None, :]
        if kernel_type == "gaussian":
            w = np.exp(-0.5 * (diff / bandwidth) ** 2)
        else:
            w = (np.abs(diff) <= bandwidth).astype(float)
        row_sum = w.sum(axis=1)                 # >= 1: the diagonal has weight K(0) = 1
        row_sq_sum = np.einsum("ij,ij->i", w, w)
        n_eff[start:stop] = row_sum ** 2 / row_sq_sum
        if est is not None:
            est[start:stop] = (w @ values) / row_sum
        if scale is None:
            var_factor[start:stop] = row_sq_sum / row_sum ** 2
        else:
            u_i = scale[start:stop]
            mean_factor[start:stop] = (w @ scale) / (u_i * row_sum)
            var_factor[start:stop] = (
                np.einsum("ij,j->i", w * w, scale * scale) / (u_i ** 2 * row_sum ** 2)
            )
    return est, n_eff, mean_factor, var_factor