import numpy as np
import pandas as pd


def mean_error(
    errors: np.ndarray | pd.Series,
    weights: np.ndarray | pd.Series | None = None,
    normalize: bool = False
) -> float:
    """
    Compute mean error with optional weights.

    Parameters
    ----------
    errors : np.ndarray or pd.Series
        Error values.
    weights : np.ndarray or pd.Series, optional
        Weights aligned with errors. If both are pd.Series,
        alignment is performed using errors.index.
    normalize : bool, default False
        If True, returns weighted mean (sum(w * e) / sum(w)).
        If False and weights are given, returns sum(w * e).
        If False and no weights, returns mean(errors).

    Returns
    -------
    float
        Mean error (unweighted, weighted sum, or normalized weighted mean).

    Raises
    ------
    ValueError
        If normalize=True but weights is None.
        If weights and errors have incompatible shapes after alignment.
    """


    # --- case: no weights ---
    if weights is None:
        if normalize:
            raise ValueError("normalize=True requires weights.")
        return float(errors.mean() if isinstance(errors, pd.Series)
                     else np.mean(np.asarray(errors)))

    # --- case: with weights ---
    if isinstance(errors, pd.Series):
        errors_np = errors.to_numpy()

        if isinstance(weights, pd.Series):
            weights = weights.reindex(errors.index)

        weights_np = np.asarray(weights)
    else:
        errors_np = np.asarray(errors)
        weights_np = np.asarray(weights)

    if errors_np.shape != weights_np.shape:
        raise ValueError("errors and weights must have same shape")

    weighted_sum = np.sum(errors_np * weights_np)

    if normalize:
        wsum = np.sum(weights_np)
        if wsum == 0:
            raise ValueError("sum of weights is zero")
        return float(weighted_sum / wsum)

    return float(weighted_sum)