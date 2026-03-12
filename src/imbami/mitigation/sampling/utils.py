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


def interpolate_sample(
    x: np.ndarray, 
    y: np.ndarray, 
    feature_ranges: np.ndarray,
    categorical_mask: np.ndarray,
    numerical_mask: np.ndarray,
    rng: np.random.Generator
    ) -> np.ndarray:
    """
    Generate a new synthetic sample by interpolating between two samples.

    This function is inspired by the SMOTE technique and interpolates both categorical and numeric data
    to create a new synthetic sample. The target column is interpolated based on HEOM distance.

    Parameters:
    ----------
    x : np.ndarray
        The first sample (reference data point).
    y : np.ndarray
        Second sample.
    feature_ranges : np.ndarray
        The range (max-min) of each feature for calculating the HEOM-distance.
    categorical_mask : np.ndarray
        Array of length a containing True if the corresponding data point is categorical, else False.
    numerical_mask : np.ndarray
        Array of length a containing True if the corresponding data point is numerical, else False.
    rng : np.random.Generator
        Random number generator for reproducibility.

    Returns:
    -------
    np.ndarray
        A new synthetic sample interpolated between the two input samples, with the target column
        interpolated based on HEOM distance.
    """
    new_sample = np.zeros_like(x)

    # For categoricals chose between both options
    random_choices = rng.random(categorical_mask.sum()) < 0.5
    # Set values in result based on random choices
    new_sample[categorical_mask] = np.where(random_choices, x[categorical_mask], y[categorical_mask])

    # For numericals interpolate between both options (exclude target column)
    diffs = y[1:][numerical_mask[1:]] - x[1:][numerical_mask[1:]]
    new_sample[1:][numerical_mask[1:]] = x[1:][numerical_mask[1:]] + rng.uniform(size=numerical_mask[1:].sum()) * diffs

    # Calculate distance (exclude target column)
    # heom_distance returns an (1,) array, thus reduce it to float using [0]
    dist_ref = heom_distance(x= x[1:],
                                y=new_sample[1:],
                                feature_ranges=feature_ranges[1:],
                                categorical_mask=categorical_mask[1:],
                                numerical_mask=numerical_mask[1:])[0]
    dist_near = heom_distance(x= y[1:],
                                y=new_sample[1:],
                                feature_ranges=feature_ranges[1:],
                                categorical_mask=categorical_mask[1:],
                                numerical_mask=numerical_mask[1:])[0]

    if dist_near + dist_ref == 0:
        new_sample[0] = (x[0] + y[0])/2
    elif x[0] == y[0]:
        new_sample[0] = y[0]
    else:
        new_sample[0] = (dist_near * x[0] + dist_ref * y[0]) / (dist_near + dist_ref)

    return new_sample


def add_gaussian_noise(
        x: np.ndarray, 
        standard_deviations: np.ndarray, 
        noise_factor: float,
        n_samples: int,
        numerical_mask: np.ndarray,
        categorical_mask: np.ndarray,
        rng: np.random.Generator
    ) -> np.ndarray:
        """
        Add Gaussian noise to a reference sample based on specified standard deviations and a noise factor.

        Parameters:
        ----------
        x : np.ndarray
            The original data array to which noise will be added. Shape should be (n_features,).
        
        standard_deviations : np.ndarray
            An array of standard deviations for each corresponding element in the reference_sample x. Shape should be (n_features,).
        
        noise_factor : float
            The multiplier for the generated Gaussian noise.

        n_samples : int
            Number of samples with Gaussian noise to be returned.

        numerical_mask : np.ndarray
            A boolean array where True indicates numerical features to which noise should be added. Shape should be (n_features,).
        
        categorical_mask : np.ndarray
            A boolean array where True indicates categorical features to which noise should be preserved (no noise added). Shape should be (n_features,).

        Returns:
        -------
        np.ndarray
            A new NumPy array of shape (n_samples, n_features) with Gaussian noise added to the numerical elements of the original reference_sample x.
        """
        # Initialize the output array
        noisy_samples = np.zeros((n_samples, x.shape[0]))
        
        # Generate noise only for numerical features
        noise = rng.normal(0, standard_deviations[numerical_mask] * noise_factor, size= (n_samples, np.sum(numerical_mask)))
        
        noisy_samples[:, numerical_mask] = x[numerical_mask] + noise
        
        # Preserve categorical features (no noise added)
        noisy_samples[:, categorical_mask] = x[categorical_mask]
        
        return noisy_samples