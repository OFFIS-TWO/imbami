import numpy as np
import pandas as pd
from scipy.stats import norm


def crps_normal_dist(
    y_true: np.ndarray | pd.Series,
    mu_predicted: np.ndarray | pd.Series,
    sigma_predicted: np.ndarray | pd.Series,
    weight: np.ndarray | pd.Series | None = None,
    return_per_sample: bool = False,
) -> float | np.ndarray | pd.Series:
    """
    Compute the (weighted) Continuous Ranked Probability Score (CRPS) for Gaussian forecasts/predictions.

    The CRPS for a normal predictive distribution
    N(mu_predicted, sigma_predicted**2) is computed as

        CRPS = sigma * [z * (2 * Phi(z) - 1)
                        + 2 * phi(z)
                        - 1 / sqrt(pi)]

    where
        z = (y_true - mu_predicted) / sigma_predicted,
        Phi is the standard normal CDF,
        phi is the standard normal PDF.

    Sample weights are applied in the standard way by multiplying the
    per-sample CRPS. If `return_per_sample=False`, the function returns the
    weighted mean CRPS

        sum(weight * CRPS) / sum(weight)

    If no weights are supplied, all samples receive weight 1.

    Parameters
    ----------
    y_true
        Observed target values.
    mu_predicted
        Predicted normal means.
    sigma_predicted
        Predicted normal standard deviations. Must be strictly positive.
    weight
        Optional non-negative sample weights.
    return_per_sample
        If True, return the weighted per-sample CRPS values.
        Otherwise, return the weighted mean CRPS.

    Returns
    -------
    float | np.ndarray | pd.Series
        Weighted mean CRPS if `return_per_sample=False`,
        otherwise the weighted per-sample CRPS.

    Raises
    ------
    ValueError
        If the input lengths differ, if any predicted standard deviation is
        non-positive, or if the sum of weights is zero.
    """
    y = np.asarray(y_true, dtype=np.float64)
    mu = np.asarray(mu_predicted, dtype=np.float64)
    sigma = np.asarray(sigma_predicted, dtype=np.float64)

    if not (y.shape == mu.shape == sigma.shape):
        raise ValueError("y_true, mu_predicted, and sigma_predicted must have identical shapes.")

    if np.any(sigma <= 0):
        raise ValueError("All sigma_predicted values must be strictly positive.")

    if weight is None:
        w = np.ones_like(y)
    else:
        w = np.asarray(weight, dtype=np.float64)
        if w.shape != y.shape:
            raise ValueError("weight must have the same shape as the inputs.")
    weight_sum = w.sum()
    if weight_sum == 0:
        raise ValueError("Sum of weights must be positive.")

    z = (y - mu) / sigma

    crps = sigma * (
        z * (2.0 * norm.cdf(z) - 1.0)
        + 2.0 * norm.pdf(z)
        - 1.0 / np.sqrt(np.pi)
    )

    weighted_crps = w * crps

    if return_per_sample:
        if isinstance(y_true, pd.Series):
            return pd.Series(weighted_crps, index = y_true.index)
        else:
            return weighted_crps

    return float(weighted_crps.sum() / weight_sum)