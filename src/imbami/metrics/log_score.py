import numpy as np
import pandas as pd


def logarithmic_score_normal_dist(
    y_true: np.ndarray | pd.Series,
    mu_predicted: np.ndarray | pd.Series,
    sigma_predicted: np.ndarray | pd.Series,
    weight: np.ndarray | pd.Series | None = None,
    return_per_sample: bool = False,
) -> float | np.ndarray | pd.Series:
    """
    Compute the logarithmic score (negative log-likelihood) for Gaussian
    probabilistic forecasts.

    For a predictive distribution N(mu_predicted, sigma_predicted**2), the
    logarithmic score for a single observation is

        LogS = 0.5 * log(2*pi)
             + log(sigma)
             + (y_true - mu_predicted)^2 / (2 * sigma^2)

    Lower values indicate better probabilistic forecasts.

    Sample weights are applied in the standard way by multiplying the
    per-sample scores. If `return_per_sample=False`, the function returns the
    weighted mean logarithmic score

        sum(weight * LogS) / sum(weight)

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
        If True, return the weighted per-sample logarithmic scores.
        Otherwise, return the weighted mean logarithmic score.

    Returns
    -------
    float | np.ndarray | pd.Series
        Weighted mean logarithmic score if `return_per_sample=False`,
        otherwise the weighted per-sample logarithmic scores.

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
        raise ValueError(
            "y_true, mu_predicted, and sigma_predicted must have identical shapes."
        )

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
    
    log_score = (
        0.5 * np.log(2.0 * np.pi)
        + np.log(sigma)
        + ((y - mu) ** 2) / (2.0 * sigma**2)
    )

    weighted_log_score = w * log_score

    if return_per_sample:
        if isinstance(y_true, pd.Series):
            return pd.Series(weighted_log_score, index=y_true.index)
        return weighted_log_score

    return float(weighted_log_score.sum() / weight_sum)