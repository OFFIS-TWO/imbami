from typing import Literal
import numpy as np



def expected_squared_error_kernel_estimator(sigma: np.ndarray,
                           errors: np.ndarray,
                           bandwidth: float|None=None,
                           bandwidth_factor: float|int = 1.0,
                           kernel_type: Literal["gaussian", "uniform"] = "gaussian"
                           ) -> np.ndarray:
    """
    Kernel regression estimate of E[e^2 | sigma].

    Parameters
    ----------
    sigma : array-like, shape (N,)
        Predicted uncertainties.
    errors : array-like, shape (N,)
        Absolute prediction errors (not squared) e = |y - mu|.
    bandwidth : float, optional
        Kernel bandwidth. If None, Silverman's rule is used.
    bandwidth_factor : float|int, default = 1.0
        Scales the provided (or calculated bandwidth).
    kernel_type : {"gaussian", "uniform"}, default="gaussian"
        Kernel function used for weighting samples.

    Returns
    -------
    est : ndarray, shape (N,)
        Estimated conditional mean squared error for each sigma.
    """
    sigma = np.asarray(sigma, dtype=float)
    errors = np.asarray(errors, dtype=float)

    if sigma.ndim != 1 or errors.ndim != 1:
        raise ValueError("sigma and errors must be 1D arrays.")
    if len(sigma) != len(errors):
        raise ValueError("sigma and errors must have the same length.")
    if kernel_type not in ("gaussian", "uniform"):
        raise ValueError(
            f"Unknown kernel_type '{kernel_type}'. "
            "Choose 'gaussian' or 'uniform'."
        )
    
    n = sigma.size

    if bandwidth is None:
        # Silvermans rule of thumb
        std = np.std(sigma, ddof=1)
        iqr = np.subtract(*np.percentile(sigma, [75, 25]))
        scale = min(std, iqr / 1.34)
        new_bandwidth = 0.9 * scale * n ** (-1 / 5)

        # Fallback if all sigma values are identical
        if new_bandwidth <= 0:
            new_bandwidth = 1e-8
    else:
        new_bandwidth = bandwidth
    new_bandwidth = new_bandwidth * bandwidth_factor

    # Pairwise differences
    diff = sigma[:, None] - sigma[None, :]

    if kernel_type == "gaussian":
        weights = np.exp(-0.5 * (diff / new_bandwidth) ** 2)

    elif kernel_type == "uniform":
        weights = (np.abs(diff) <= new_bandwidth).astype(float)

    error_sq = errors**2

    numerator = weights @ error_sq
    denominator = weights.sum(axis=1)

    return numerator / denominator



def expected_squared_error_rolling_estimator(
    sigma: np.ndarray,
    errors: np.ndarray,
    window_size: int,
) -> np.ndarray:
    """
    Rolling-window estimate of E[e^2 | sigma].

    The samples are first sorted by predicted uncertainty. For each sample,
    the expected squared error is estimated by averaging the squared errors
    within a centered rolling window in the sorted uncertainty domain.

    Parameters
    ----------
    sigma : array-like, shape (N,)
        Predicted uncertainties.
    errors : array-like, shape (N,)
        Prediction errors (not squared).
    window_size : int
        Number of neighboring samples used for the estimate.

    Returns
    -------
    np.ndarray, shape (N,)
        Estimated conditional mean squared error for each sample in the
        original input order.
    """
    sigma = np.asarray(sigma, dtype=float)
    errors = np.asarray(errors, dtype=float)

    if sigma.ndim != 1 or errors.ndim != 1:
        raise ValueError("sigma and errors must be 1D arrays.")
    if sigma.shape != errors.shape:
        raise ValueError("sigma and errors must have the same shape.")
    if not (1 <= window_size <= sigma.size):
        raise ValueError("window_size must be between 1 and len(sigma).")

    error_sq = errors**2

    # Sort by predicted uncertainty
    order = np.argsort(sigma)
    error_sq_sorted = error_sq[order]

    estimates_sorted = np.empty_like(error_sq_sorted)

    half_window = window_size // 2

    for i in range(sigma.size):
        start = max(0, i - half_window)
        end = min(sigma.size, start + window_size)
        start = max(0, end - window_size)

        estimates_sorted[i] = np.mean(error_sq_sorted[start:end])

    # Restore original ordering
    estimates = np.empty_like(estimates_sorted)
    estimates[order] = estimates_sorted

    return estimates