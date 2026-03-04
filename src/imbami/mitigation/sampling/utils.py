import numpy as np



def heom_distance(
    x: np.ndarray,
    y: np.ndarray,
    feature_ranges: np.ndarray,
    categorical_mask: np.ndarray,
    numerical_mask: np.ndarray) -> np.ndarray:
    """
    Calculate the Heterogeneous Euclidean-Overlap Metric (HEOM) distance between:
    - two 1D arrays, or
    - a 1D array and each row of a 2D array.

    Parameters
    ----------
    x : np.ndarray
        Array of shape (a,) representing one data point.
    y : np.ndarray
        Array of shape (a,) or (n, a) representing one or multiple data points.
    feature_ranges : np.ndarray
        Range (max - min) of each numerical feature.
    categorical_mask : np.ndarray
        Boolean mask indicating categorical features.
    numerical_mask : np.ndarray
        Boolean mask indicating numerical features.

    Returns
    -------
    np.ndarray
        Array of HEOM distances. Shape (1,) if y is 1D, else (n,).
    """
    if y.ndim == 1 or y.ndim == 0:
        y = y.reshape(1, -1)
        single_input = True
    else:
        single_input = False

    x = x.reshape(1, -1)  # Shape (1, a)
    feature_ranges = feature_ranges.reshape(-1) # for the case Shape(0) (float input)
    categorical_mask = categorical_mask.reshape(-1)
    numerical_mask = numerical_mask.reshape(-1)

    # Masks
    nan_mask = np.isnan(x) | np.isnan(y)  # Shape (n, a)
    range_mask = (feature_ranges != 0)
    num_mask = numerical_mask & range_mask  # Shape (a,)

    # Initialize distances
    distances = np.zeros_like(y, dtype=float)

    # Categorical distance (0 if same, 1 if different)
    distances[:, categorical_mask] = (x[:, categorical_mask] != y[:, categorical_mask])

    # Numerical distance (normalized by feature range)
    valid_x = x[:, num_mask]
    valid_y = y[:, num_mask]
    valid_ranges = feature_ranges[num_mask]
    distances[:, num_mask] = np.abs(valid_x - valid_y) / valid_ranges

    # Missing value handling: set distance to 1
    missing_mask = np.logical_or(nan_mask, ~np.expand_dims(range_mask, axis=0))
    distances[missing_mask] = 1

    # HEOM distance
    heom = np.sqrt(np.sum(distances ** 2, axis=1))

    if single_input:
        return heom.reshape(1)
    return heom