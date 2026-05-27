import pandas as pd
import numpy as np
from .base_mitigation_method import SamplingMethodBase
import logging
from .utils import heom_distance,interpolate_sample, add_gaussian_noise

class SMOGN(SamplingMethodBase):
    """
    SMOGN (Synthetic Minority Over-sampling Technique for Regression with Gaussian Noise) class.


    This class performs both oversampling and undersampling on imbalanced regression datasets.
    It generates synthetic samples for minority cases using interpolation and Gaussian noise,
    and optionally removes majority cases through undersampling.

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
        Initialize the SMOGN sampler.

        Parameters
        ----------
        data : pd.DataFrame
            Input dataset containing features and target variable.
        target_column : str
            Name of the target column in the dataset.
        relevance_values : pd.Series
            Relevance values for each sample in the dataset, ranging from 0 to 1. (otherwise are normalized)
            Higher values indicate more relevant/rare samples that should be oversampled.
        """
        super().__init__(data=data, target_column= target_column, relevance_values=relevance_values)

        # Validate sample relevance
        if not ((relevance_values >= 0) & (relevance_values <= 1)).all():
            # Normalize sample relevance
            self.relevance_values = self._normalize_with_clipping(self.relevance_values)
        
        # Sort data and relevance
        self.data = self.data.sort_values(by=target_column, ascending=True)
        self.relevance_values = self.relevance_values.loc[self.data.index]

        self._heom_distance = heom_distance
        self._interpolate_sample = interpolate_sample
        self._add_gaussian_noise = add_gaussian_noise

        

    def run_sampling(self,
                     knns: int = 5,
                     noise_factor: float = 0.01,
                     oversample_rate: float = 0.5,
                     undersample_rate: float = 0.5,
                     relevance_threshold: float = 0.8,
                     enable_undersampling: bool = True,
                     random_state: int|None = 0) -> pd.DataFrame:
        """
        Execute the SMOGN sampling procedure to balance the dataset.

        This method performs both oversampling of rare cases and optional undersampling
        of common cases to create a more balanced dataset for regression problems.

        Parameters
        ----------
        knns : int, default=5
            Number of nearest neighbors to consider when generating synthetic samples.
        noise_factor : float, default=0.01
            Scaling factor for Gaussian noise added to synthetic samples.
        oversample_rate : float, default=0.5
            Rate at which to oversample the rare cases (relevant samples above threshold).
            e.g. 0.5 generates 50% more rare cases.
        undersample_rate : float, default=0.5
            Rate at which to undersample the common cases (relevant samples below threshold).
            This determines what fraction of the common samples to keep.
        relevance_threshold : float, default=0.8
            Threshold for determining which samples are considered rare (above threshold)
            and which are considered common (below threshold).
        enable_undersampling : bool, default=True
            Whether to perform undersampling of common cases.
        random_state : int or None, default=0
            Random seed for reproducibility.

        Returns
        -------
        pd.DataFrame
            The balanced dataset after applying SMOGN sampling.
        """
        rng = self._get_random_generator(random_state)
        relevance_values_numpy = self.relevance_values.to_numpy()
        self.data_numpy = self.data.to_numpy()
        self.feature_ranges = self.data_numpy.max(axis = 0) - self.data_numpy.min(axis = 0)
        self.standard_deviations = self.data_numpy.std(axis=0) # For Gaussian noise addition
        self.categorical_mask = np.array([True if col in self.categorical_columns else False for col in self.data.columns])
        self.numerical_mask = ~self.categorical_mask

        # Determine bins/partitions with samples above/below the relevance threshold.
        mask = relevance_values_numpy >= relevance_threshold

        # indices where the mask changes (True->False or False->True)
        change_points = np.where(np.diff(mask.astype(int)) != 0)[0] + 1

        # split indices
        index_groups = np.split(np.arange(len(relevance_values_numpy)), change_points)

        partition_indice_above = []
        partition_indice_below = []

        for idx in index_groups:
            if mask[idx[0]]:
                partition_indice_above.append(idx)
            else:
                partition_indice_below.append(idx)

        self.num_gaussian_samples = 0
        self.num_interpolated_samples = 0
        self.num_oversampled_samples = 0

        # Oversampling
        oversampled_rows = []
        for part_indice in partition_indice_above:
            # calculate number of samples to oversample
            n_create = int(np.ceil(len(part_indice) * oversample_rate))                 
            partition_data = self.data_numpy[part_indice] # get rows of the full dataset that correspond to the indices in the partition.
            n_samples_partition = partition_data.shape[0]
            # compute distance matrix inside bump
            distance_matrix = np.zeros((n_samples_partition, n_samples_partition))
            for i in range(n_samples_partition):
                distance_matrix[i] = self._heom_distance(
                    x=partition_data[i],
                    y=partition_data,
                    feature_ranges=self.feature_ranges,
                    categorical_mask=self.categorical_mask,
                    numerical_mask=self.numerical_mask
                )
            maxDM = np.median(distance_matrix, axis=1) / 2 # array of maximum allowed distance per sample
            # nearest neighbors
            effective_knns = min(knns, n_samples_partition - 1)
            neighbor_indices = np.argsort(distance_matrix, axis=1)[:, 1:effective_knns+1]
                     
            # main generation loop
            for i in range(n_create):
                random_index = int(rng.integers(0, partition_data.shape[0]))
                seed_sample = partition_data[random_index]

                use_gaussian_noise = False

                if effective_knns == 0:
                    use_gaussian_noise = True
                    rnd_selected_nn_idx = None # just for the code analyser to show now warnings
                else:
                    rnd_selected_nn_idx = rng.choice(neighbor_indices[random_index])
                    distance = distance_matrix[random_index, rnd_selected_nn_idx]

                    if distance > maxDM[random_index]:
                        use_gaussian_noise = True

                # Add gaussian Noise
                if use_gaussian_noise:
                    new_samples = self._add_gaussian_noise(x=seed_sample,
                                                            standard_deviations= self.standard_deviations,
                                                            noise_factor= noise_factor,
                                                            n_samples= 1,
                                                            numerical_mask= self.numerical_mask,
                                                            categorical_mask=self.categorical_mask,
                                                            rng=rng)
                    oversampled_rows.append(new_samples)
                    self.num_gaussian_samples += 1
                else:
                    seed_sample2 = partition_data[rnd_selected_nn_idx]
                    new_sample = self._interpolate_sample(x=seed_sample,
                                                        y=seed_sample2,
                                                        feature_ranges= self.feature_ranges,
                                                        categorical_mask= self.categorical_mask,
                                                        numerical_mask=self.numerical_mask,
                                                        rng=rng)
                    oversampled_rows.append(new_sample)
                    self.num_interpolated_samples += 1


        # Postprocessing: convert back to pandas DataFrame
        if len(oversampled_rows) == 0: # this is the case if no relevance values were above the threshold -> no oversampling
            oversampled_array = np.empty((0, self.data.shape[1]))
        else:
            oversampled_array = np.vstack(oversampled_rows)
        self.num_oversampled_samples = oversampled_array.shape[0]
        oversampled_data = pd.DataFrame(
            data=oversampled_array,
            columns=self.data.columns,
            index=range(self.data.index.max() + 1, self.data.index.max() + 1 + oversampled_array.shape[0])
        )
        for col in self.data.columns:
            oversampled_data[col] = oversampled_data[col].astype(self.data[col].dtype)

        # Undersampling
        if enable_undersampling:
            drop = [] # stores indices to keep
            for idx in partition_indice_below:
                n_drop = int(np.ceil(len(idx) * undersample_rate))

                selected = rng.choice(idx, size=n_drop, replace=False)
                drop.append(selected)

            if len(drop) == 0:
                dropped_data_indices = pd.Index([])
            else:
                dropped_indices = np.concatenate(drop)
                dropped_data_indices = self.data.iloc[dropped_indices].index # transform from numpy indices to pandas

            undersampled_data = self.data.drop(index=dropped_data_indices)

            new_data = pd.concat([undersampled_data, oversampled_data], axis=0)
        else:
            new_data = pd.concat([self.data, oversampled_data], axis=0)

        return new_data
    


def apply_smogn(data: pd.DataFrame,
            target_column: str,
            relevance_values: pd.Series,
            relevance_threshold: float = 0.8,
            enable_undersampling: bool = True,
            oversample_rate: float = 0.5,
            undersample_rate: float = 0.5,
            noise_factor: float = 0.01,
            knns: int = 5,
            random_state: int | None = 0) -> pd.DataFrame:
    """
    Apply SMOGN (Synthetic Minority Over-sampling Technique for Regression with Gaussian Noise)
    to balance an imbalanced regression dataset.

    This is a convenience function that creates a SMOGN sampler instance and executes
    the sampling procedure in one step. It handles both oversampling of rare cases
    and optional undersampling of common cases to create a more balanced dataset.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataset containing features and target variable.
    target_column : str
        Name of the target column in the dataset.
    relevance_values : pd.Series
        Relevance values for each sample in the dataset, ranging from 0 to 1.
        Higher values indicate more relevant/rare samples that should be oversampled.
    relevance_threshold : float, default=0.8
        Threshold for determining which samples are considered rare (above threshold)
        and which are considered common (below threshold).
    enable_undersampling : bool, default=True
        Whether to perform undersampling of common cases.
    oversample_rate : float, default=0.5
        Rate at which to oversample the rare cases (relevant samples above threshold).
        For each sample, this determines how many synthetic samples to create.
    undersample_rate : float, default=0.5
        Rate at which to undersample the common cases (relevant samples below threshold).
        This determines what fraction of the common samples to keep.
    noise_factor : float, default=0.01
        Scaling factor for Gaussian noise added to synthetic samples.
    knns : int, default=5
        Number of nearest neighbors to consider when generating synthetic samples.
    random_state : int or None, default=0
        Random seed for reproducibility.

    Returns
    -------
    pd.DataFrame
        The balanced dataset after applying SMOGN sampling.

    Notes
    -----
    The input DataFrame must have correct data types assigned:
    - Numeric columns should be of numeric types (int, float)
    - Categorical columns should be explicitly marked as 'category' dtype
    - Discrete numeric states (e.g., 1, 2, 3) must be of type 'category' to be treated as categorical;
      otherwise they will be considered continuous numeric features.
    """
    logging.debug('Begin SMOGN...')
    sampler = SMOGN(data = data,
                    target_column = target_column,
                    relevance_values = relevance_values)
    new_data = sampler.run_sampling(oversample_rate=oversample_rate,
                                    undersample_rate=undersample_rate,
                                    relevance_threshold=relevance_threshold,
                                    knns=knns,
                                    noise_factor=noise_factor,
                                    enable_undersampling=enable_undersampling,
                                    random_state=random_state)    
    logging.debug('Finished SMOGN...')
    return new_data 