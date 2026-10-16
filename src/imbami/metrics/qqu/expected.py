from typing import Literal
import numpy as np
from .utils import _check_inputs, _silverman_bandwidth

def kernel_conditional_mean(
    sigma: np.ndarray,
    values: np.ndarray,
    bandwidth: float | None = None,
    bandwidth_factor: float | int = 1.0,
    kernel_type: Literal["gaussian", "uniform"] = "gaussian",
    chunk_size: int = 1024,
) -> np.ndarray:
    """
    Kernel (Nadaraya-Watson) estimate of E[values | sigma].
 
    Generic version of the estimator: ``values`` is S_err(e) evaluated per
    sample (e^2 for the squared statistic, |e| for the absolute one, ...).
 
    Parameters
    ----------
    sigma : array-like, shape (N,)
        Predicted uncertainties.
    values : array-like, shape (N,)
        Per-sample residual statistic S_err(e_j).
    bandwidth : float, optional
        Kernel bandwidth. If None, Silverman's rule is used.
    bandwidth_factor : float, default 1.0
        Scales the provided (or calculated) bandwidth.
    kernel_type : {"gaussian", "uniform"}, default "gaussian"
    chunk_size : int, default 1024
        Rows of the pairwise kernel matrix computed at once. Runtime is
        O(N^2), but memory is only O(chunk_size * N) instead of O(N^2).
 
    Returns
    -------
    np.ndarray, shape (N,)
        Estimated E[S_err(e) | sigma_i] for each sample.
    """
    sigma, values = _check_inputs(sigma, values)
    if kernel_type not in ("gaussian", "uniform"):
        raise ValueError(
            f"Unknown kernel_type '{kernel_type}'. Choose 'gaussian' or 'uniform'."
        )
 
    bw = _silverman_bandwidth(sigma) if bandwidth is None else bandwidth
    bw = bw * bandwidth_factor
 
    n = sigma.size
    est = np.empty(n, dtype=float)
    for start in range(0, n, chunk_size):
        stop = min(start + chunk_size, n)
        diff = sigma[start:stop, None] - sigma[None, :]
        if kernel_type == "gaussian":
            weights = np.exp(-0.5 * (diff / bw) ** 2)
        else:
            weights = (np.abs(diff) <= bw).astype(float)
        # The diagonal always has weight 1, so the denominator is >= 1.
        est[start:stop] = (weights @ values) / weights.sum(axis=1)
    return est


def rolling_conditional_mean(
    sigma: np.ndarray,
    values: np.ndarray,
    window_size: int,
) -> np.ndarray:
    """
    Rolling-window estimate of E[values | sigma].
 
    Samples are sorted by sigma; each estimate is the mean of ``values``
    in a centered window of ``window_size`` neighbours (shifted at the
    borders so the window always has full size). Vectorized via cumulative
    sums, O(N log N).
 
    Returns
    -------
    np.ndarray, shape (N,)
        Estimates in the original input order.
    """
    sigma, values = _check_inputs(sigma, values)
    n = sigma.size
    if not (1 <= window_size <= n):
        raise ValueError("window_size must be between 1 and len(sigma).")
 
    order = np.argsort(sigma, kind="stable")
    csum = np.concatenate(([0.0], np.cumsum(values[order])))
 
    start = np.clip(np.arange(n) - window_size // 2, 0, n - window_size)
    est_sorted = (csum[start + window_size] - csum[start]) / window_size
 
    est = np.empty(n, dtype=float)
    est[order] = est_sorted
    return est