import pandas as pd
import numpy as np
from .base_mitigation_method import SamplingMethodBase
import logging
from .utils import heom_distance,interpolate_sample

class WSMOTER(SamplingMethodBase):
    """
    WSMOTER (Weighting SMOTE for Regression) class.


    This class performs oversampling on imbalanced regression datasets.
    Based on 'WSMOTER: a novel approach for imbalanced regression' by Camacho and Bacao (2024).

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset containing features and target variable.
    target_column : str
        Name of the target column in the dataset.
    relevance_values : pd.Series
        Relevance values for each sample in the dataset, ranging from 0 to 1.
        Higher values indicate more relevant/rare samples that should be oversampled.

        
    Important Note on Data Types:
    ----------------------------
    The input DataFrame must have correct data types assigned:
    - Numeric columns should be of numeric types (int, float)
    - Categorical columns should be explicitly marked as 'category' dtype
    - Discrete numeric states (e.g., 1, 2, 3) must be of type 'category' to be treated as categorical;
      otherwise they will be considered continuous numeric features.
    """
    def __init__(self,
                 data: pd.DataFrame,
                 target_column: str,
                 relevance_values: pd.Series):
        """
        Initialize the WSMOTER sampler.

        Parameters
        ----------
        data : pd.DataFrame
            Input dataset containing features and target variable.
        target_column : str
            Name of the target column in the dataset.
        relevance_values : pd.Series
            Relevance values for each sample in the dataset, ranging from 0 to 1.
            Higher values indicate more relevant/rare samples that should be oversampled.
        """
        super().__init__(data=data, target_column= target_column, relevance_values=relevance_values)

        # Validate relevance
        if not ((relevance_values >= 0) & (relevance_values <= 1)).all():
            # Normalize sample relevance
            self.relevance_values = self._normalize_with_clipping(self.relevance_values)
        
        # Sort data and relevance
        self.data = self.data.sort_values(by=target_column, ascending=True)
        self.relevance_values = self.relevance_values.loc[self.data.index]

        self._heom_distance = heom_distance
        self._interpolate_sample = interpolate_sample
        

    def run_sampling(self,
                     knns: int = 5,
                     k_position_shifts: int = 10,
                     oversample_rate: float = 0.5,
                     random_state: int|None = 0) -> pd.DataFrame:
        """
        Execute the WSMOTER oversampling algorithm.

        Parameters
        ----------
        knns : int, default=5
            Number of nearest neighbors to consider for interpolation.
        k_position_shifts : int, default=10
            Number of neighboring samples to consider in the target space.
            Must be greater than knns.
        oversample_rate : float, default=0.5
            Rate at which to oversample the dataset (e.g., 0.5 means 50% more samples).
        random_state : int or None, default=0
            Random seed for reproducibility.

        Returns
        -------
        pd.DataFrame
            Oversampled dataset containing both original and synthetic samples.

        Raises
        ------
        ValueError
            If knns is greater than k_position_shifts.
        """
        if knns > k_position_shifts:
            raise ValueError(f"{knns=} has to smaller than {k_position_shifts=}.")

        rng = self._get_random_generator(random_state)
        self.data_numpy = self.data.to_numpy()
        self.feature_ranges = self.data_numpy.max(axis = 0) - self.data_numpy.min(axis = 0)
        self.categorical_mask = np.array([True if col in self.categorical_columns else False for col in self.data.columns])
        self.numerical_mask = ~self.categorical_mask

        weights = np.array(self.relevance_values/self.relevance_values.sum()) # Why do this? Kind of useless. (but stated in the paper)

        max_weight = weights.max()
        min_weight = weights.min()
        target_median = self.data[self.target_column].median()

        total_oversample_count = int(self.data_numpy.shape[0]*oversample_rate)
        oversampled_data = np.zeros((total_oversample_count, self.data_numpy.shape[1]))
        


        self.num_oversampled_samples = 0
        max_idx = self.data_numpy.shape[0]
        while self.num_oversampled_samples < total_oversample_count:
            random_index = int(rng.integers(0, weights.shape[0]))
            sample_weight = weights[random_index]
   
            if sample_weight > rng.uniform(low=min_weight, high=max_weight):
                # rare samples have a high weight, thus over-sample whenever larger then random
                first_seed = self.data_numpy[random_index]

                # select the neighboring samples
                if first_seed[0] < target_median: # target value < target median -> sample is in first half of data
                    if random_index >= k_position_shifts:
                        # sample is far enough from first sample of data (idx=0), just take the k samples to the left.
                        neighbor_indices = np.arange(random_index - k_position_shifts, random_index)
                    else:
                        # there are no k samples to the left (too close to idx=0), use the first k+1 samples of the dataset and drop the seed sample from it.
                        neighbor_indices = np.arange(0, k_position_shifts + 1)
                        neighbor_indices = neighbor_indices[neighbor_indices != random_index]
                else: # target value > target median -> sample is in the second half of data.
                    if random_index <= max_idx - k_position_shifts - 1:
                        # sample is far enough from last sample to just use the k samples right of it.
                        neighbor_indices = np.arange(random_index + 1, random_index + k_position_shifts + 1)
                    else:
                        # there are no k samples right of it (too close to idx=data.shape[0]), use last k+1 samples of data and drop the seed sample from it.
                        neighbor_indices = np.arange(max_idx - (k_position_shifts + 1), max_idx)
                        neighbor_indices = neighbor_indices[neighbor_indices != random_index]
                
                neighbor_samples = self.data_numpy[neighbor_indices]

                distances = self._heom_distance(x=first_seed,
                                    y=neighbor_samples,
                                    feature_ranges= self.feature_ranges,
                                    categorical_mask=self.categorical_mask,
                                    numerical_mask=self.numerical_mask)
                nearest_neighbors = neighbor_samples[np.argsort(distances)[:knns]]
                second_seed = nearest_neighbors[rng.choice(nearest_neighbors.shape[0], replace=False)]

                new_sample = self._interpolate_sample(x=first_seed,
                                                    y=second_seed,
                                                    feature_ranges= self.feature_ranges,
                                                    categorical_mask= self.categorical_mask,
                                                    numerical_mask=self.numerical_mask,
                                                    rng=rng)


                # Store the new samples
                oversampled_data[self.num_oversampled_samples] = new_sample
                self.num_oversampled_samples += 1


        oversampled_data = pd.DataFrame(
            data=oversampled_data,
            columns=self.data.columns,
            index=range(self.data.index.max() + 1, self.data.index.max() + 1 + oversampled_data.shape[0])
        )
        for col in self.data.columns:
            oversampled_data[col] = oversampled_data[col].astype(self.data[col].dtype)
        new_data = pd.concat([self.data, oversampled_data], axis=0)
        
        return new_data




def apply_wsmoter(data: pd.DataFrame,
            target_column: str,
            relevance_values: pd.Series,
            k_position_shifts: int = 10,
            oversample_rate: float = 0.5,
            knns: int = 5,
            random_state: int | None = 0) -> pd.DataFrame:
    """
    Apply WSMOTER oversampling to a dataset.

    Convenience function that creates a WSMOTER instance and runs the oversampling.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset containing features and target variable.
    target_column : str
        Name of the target column in the dataset.
    relevance_values : pd.Series
        Relevance values for each sample in the dataset, ranging from 0 to 1.
        Higher values indicate more relevant/rare samples that should be oversampled.
    k_position_shifts : int, default=10
        Number of neighboring samples to consider in the target space.
    oversample_rate : float, default=0.5
        Rate at which to oversample the dataset (e.g., 0.5 means 50% more samples).
    knns : int, default=5
        Number of nearest neighbors to consider for interpolation.
    random_state : int or None, default=0
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        Oversampled dataset containing both original and synthetic samples.

    """
    logging.debug('Begin WSMOTER...')
    sampler = WSMOTER(data = data,
                    target_column = target_column,
                    relevance_values = relevance_values)
    new_data = sampler.run_sampling(oversample_rate=oversample_rate,
                                    k_position_shifts= k_position_shifts,
                                    knns=knns,
                                    random_state=random_state)    
    logging.debug('Finished WSMOTER...')
    return new_data 