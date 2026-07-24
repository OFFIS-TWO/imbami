import numpy as np

def calculate_ENCE(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    sigma: np.ndarray,
    n_bins: int = 10
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    """
    Calculate ENCE calibration diagram values.

    Parameters
    ----------
    y_true : array-like
        Ground truth values.
    y_pred : array-like
        Predicted mean values.
    sigma : array-like
        Predicted standard deviation.
    n_bins : int
        Number of uncertainty bins.

    Returns
    -------
    rmv : np.ndarray
        Root mean variance per bin.
    rmse : np.ndarray
        Root mean squared error per bin.
    bin_centers : np.ndarray
        Mean predicted uncertainty per bin.
    counts : np.ndarray
        Number of samples per bin.
    ence : float
        Overall ENCE score.
    """

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    sigma = np.asarray(sigma)

    squared_error = (y_true - y_pred) ** 2
    variance = sigma ** 2

    # Sort by predicted uncertainty
    idx = np.argsort(variance)

    squared_error = squared_error[idx]
    variance = variance[idx]
    sigma_sorted = sigma[idx]

    # Create equal-size bins
    bins = np.array_split(np.arange(len(sigma)), n_bins)

    rmv = []
    rmse = []
    bin_centers = []
    counts = []

    for b in bins:
        if len(b) == 0:
            continue

        rmv.append(np.sqrt(np.mean(variance[b])))
        rmse.append(np.sqrt(np.mean(squared_error[b])))
        bin_centers.append(np.mean(sigma_sorted[b]))
        counts.append(len(b))

    rmv = np.asarray(rmv)
    rmse = np.asarray(rmse)
    bin_centers = np.asarray(bin_centers)
    counts = np.asarray(counts)

    ence = np.mean(np.abs(rmv - rmse) / rmv)

    return rmv, rmse, bin_centers, counts, ence