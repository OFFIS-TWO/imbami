import pandas as pd
import numpy as np
from .utils import heom_distance
from abc import ABC, abstractmethod





class SamplingMethodBase(ABC):
    """
    Abstract base class for all sampling methods.

    This class provides the common interface and functionality that all sampling methods
    should implement. It handles the basic data preparation and provides utility methods
    for random number generation.

    Attributes:
    ----------
    data : pd.DataFrame
        The input data containing features and target column.
    relevance_values : pd.Series
        Relevance values corresponding to each row in the data.
    target_column : str
        Name of the target column in the data.
    numeric_columns : list
        List of numeric column names.
    categorical_columns : list
        List of categorical column names.

    Methods:
    -------
    run_sampling(*args, **kwargs) -> pd.DataFrame:
        Abstract method to be implemented by subclasses for performing the sampling.
    _get_random_generator(random_state: int|None) -> np.random.Generator:
        Utility method to create a random number generator with optional seed.
    """
    def __init__(self, data: pd.DataFrame, target_column: str, relevance_values: pd.Series) -> None:
        """
        Initialize the sampling method with data, target column, and relevance values.

        Parameters:
        ----------
        data : pd.DataFrame
            The input data containing features and target column. The target column
            should be present in this DataFrame.
        target_column : str
            Name of the target column in the data. This column will be moved to the
            first position in the DataFrame.
        relevance_values : pd.Series
            Relevance values corresponding to each row in the data. Must have the same
            length and index as the data DataFrame.

        Raises:
        ------
        ValueError
            If relevance_values has a different length or index than the data DataFrame.
        """
        self.data = data
        self.relevance_values = relevance_values
        self.target_column = target_column
        if data.shape[0] != relevance_values.shape[0]:
            raise ValueError("relevance_values must have same length as data.")
        if not data.index.equals(relevance_values.index):
            raise ValueError("relevance_values must have the same index as data.")
        
        # Sort target column to be at the front
        self.data = self.data[[self.target_column] + [c for c in self.data.columns if c != self.target_column]]
        # Identify numeric and categorical columns
        self.numeric_columns = list(self.data.select_dtypes(include=['number']).columns)
        self.categorical_columns = [col for col in self.data.columns if col not in self.numeric_columns]

    @abstractmethod
    def run_sampling(self, *args, **kwargs) -> pd.DataFrame:
        """
        Perform the sampling operation and return the sampled data.

        Must be implemented by subclasses to define the specific sampling method.

        Returns:
        -------
        pd.DataFrame
            The sampled data as a DataFrame.
        """
        raise NotImplementedError
    
    def _get_random_generator(self, random_state: int|None) -> np.random.Generator:
        if random_state is not None:
            rng = np.random.default_rng(random_state)
        else:
            rng = np.random.default_rng()
        return rng




class EmpiricalDomainMitigationMethodBase(SamplingMethodBase):
    """
    This base class extends SamplingMethodBase with functionality specifically designed
    for oversampling techniques that are designed to work with both the empirical distribution
    of the data and the domain distribution.

    Attributes:
    ----------
    data_numpy : np.ndarray
        The data converted to a NumPy array for efficient numerical operations.
    feature_ranges : np.ndarray
        The range (max-min) of each feature, used for distance calculations.
    standard_deviations : np.ndarray
        Standard deviations of each feature, used for Gaussian noise addition.
    categorical_mask : np.ndarray
        Boolean mask indicating which features are categorical.
    numerical_mask : np.ndarray
        Boolean mask indicating which features are numerical.
    _heom_distance : function
        Reference to the HEOM distance function for calculating distances between samples.
    """
    def __init__(self,
                data: pd.DataFrame,
                target_column: str,
                relevance_values: pd.Series):
        """
        Initialize the sampling method with data, target column, and relevance values.

        Parameters:
        ----------
        data : pd.DataFrame
            The input data containing features and target column. The target column
            should be present in this DataFrame.
        target_column : str
            Name of the target column in the data. This column will be moved to the
            first position in the DataFrame.
        relevance_values : pd.Series
            Relevance values corresponding to each row in the data. Must have the same
            length and index as the data DataFrame.

        Raises:
        ------
        ValueError
            If relevance_values has a different length or index than the data DataFrame.
        """
        super().__init__(data=data, target_column=target_column, relevance_values=relevance_values) 

        # Prepare for oversampling
        self.data_numpy = self.data.to_numpy()
        self.feature_ranges = self.data_numpy.max(axis = 0) - self.data_numpy.min(axis = 0)
        self.standard_deviations = self.data_numpy.std(axis=0) # For Gaussian noise addition
        self.categorical_mask = np.array([True if col in self.categorical_columns else False for col in self.data.columns])
        self.numerical_mask = ~self.categorical_mask

        self._heom_distance = heom_distance


    def _discretize_dataset(self,
        data: pd.DataFrame, 
        num_bins: int, 
        numeric_columns: list, 
        categorical_columns: list, 
        ignore_categoricals: bool
    ) -> pd.DataFrame:
        """
        Discretize numeric and non-numeric columns in a dataset.

        This function applies discretization to specified numeric columns by dividing them into a given number of bins.
        It also converts non-numeric categorical columns to numeric codes.

        Parameters:
        ----------
        data : pd.DataFrame
            The input DataFrame containing the data to be discretized.
        
        num_bins : int
            The number of bins to use for discretizing the numeric columns.
        
        numeric_columns : list
            A list of column names in the DataFrame that contain numeric data to be discretized.
        
        categorical_columns : list
            A list of column names in the DataFrame that contain non-numeric categorical data to be converted to numeric codes.
        
        ignore_categoricals : bool
            Whether to ignore categorical columns during discretization.

        Returns:
        -------
        pd.DataFrame
            A new DataFrame with the numeric columns discretized into bins and non-numeric columns converted to numeric codes.
        """
        binned_data = data.copy()
        binned_data[numeric_columns] = data[numeric_columns].apply(lambda x: pd.cut(x, bins=num_bins, labels=False))
        
        if not ignore_categoricals:
            for col in categorical_columns:
                category_mapping = {category: idx for idx, category in enumerate(data[col].cat.categories)}
                binned_data[col] = data[col].map(category_mapping)
        
        return binned_data
        
    def _interpolate_sample(self,
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
        dist_ref = self._heom_distance(x= x[1:],
                                    y=new_sample[1:],
                                    feature_ranges=feature_ranges[1:],
                                    categorical_mask=categorical_mask[1:],
                                    numerical_mask=numerical_mask[1:])[0]
        dist_near = self._heom_distance(x= y[1:],
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




    def _add_gaussian_noise(self,
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




    def _get_similar_samples(self,
        index: int, 
        binned_data: np.ndarray, 
        numerical_mask: np.ndarray, 
        categorical_mask: np.ndarray, 
        allowed_bin_deviation: int, 
        ignore_categoricals: bool
    ) -> np.ndarray:
        """
        Retrieve the indices of rows in a NumPy array that are similar to a specified row based on categorical and numeric columns.

        Parameters:
        ----------
        index : int
            The index of the row to compare against.
        binned_data : np.ndarray
            The array containing the data. It should be pre-binned if numeric columns are used.
        numerical_mask : np.ndarray
            A boolean mask indicating which columns are numeric.
        categorical_mask : np.ndarray
            A boolean mask indicating which columns are categorical.
        allowed_bin_deviation : int
            The allowed deviation in bin values for numeric columns to consider rows as similar.
        ignore_categoricals : bool
            If True, the function will ignore categorical columns when determining similarity.

        Returns:
        -------
        np.ndarray
            Array of indices of rows that are similar to the specified row, excluding the row at the given index.
        """
        # Get target row from index if not provided
        target_row = binned_data[index]
        
        # Initialize mask for similarity (all True initially)
        mask = np.ones(binned_data.shape[0], dtype=bool)
        mask[index] = 0
        
        # Apply numerical similarity check using the numerical mask
        if numerical_mask.any():  # Check if there are any numeric columns
            numeric_data = binned_data[:, numerical_mask]
            target_numeric = target_row[numerical_mask]
            mask &= np.all(
                (numeric_data >= (target_numeric - allowed_bin_deviation)) &
                (numeric_data <= (target_numeric + allowed_bin_deviation)),
                axis=1
            )
        
        # Apply categorical similarity check using the categorical mask if not ignored
        if not ignore_categoricals and categorical_mask.any():  # Check if there are any categorical columns
            categorical_data = binned_data[:, categorical_mask]
            target_categorical = target_row[categorical_mask]
            mask &= np.all(categorical_data == target_categorical, axis=1)
        
        # Return the indices of rows that match the mask
        return np.where(mask)[0]

