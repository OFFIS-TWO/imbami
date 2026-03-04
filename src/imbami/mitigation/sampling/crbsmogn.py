from .base_mitigation_method import EmpiricalDomainMitigationMethodBase
import pandas as pd
import numpy as np
import logging

class crbSMOGN(EmpiricalDomainMitigationMethodBase):
    """
    crbSMOGN: Continuous ratio-based Synthetic Oversampling for Regression.

    Performs oversampling and optional undersampling based on sample relevance,
    controlling which samples are replicated, retained, or dropped.

    Attributes
    ----------
    data : pd.DataFrame
        The dataset to be sampled.
    target_column : str
        Name of the target variable.
    relevance_values : pd.Series
        Relevance values for each sample, in [0, 1].
    binned_data : np.ndarray
        Dataset discretized for similarity calculations.
    sample_treatment : pd.Series
        Indicates for each sample whether it is kept (1), dropped (0), or oversampled (>1).
    oversample_indices : np.ndarray
        Indices of samples selected for oversampling.
    num_oversampled_samples : int
        Number of samples added via oversampling.
    num_gaussian_samples : int
        Number of samples generated using Gaussian noise.
    num_knn_samples : int
        Number of samples generated using nearest-neighbor interpolation.
    num_dropped_samples : int
        Number of samples removed via undersampling.
    min_acceptable_relevance : float
        Minimum relevance threshold for retaining samples.
    max_acceptable_relevance : float
        Maximum relevance threshold for retaining samples.
    allowed_bin_deviation : int
        Maximum deviation in binning for similarity calculations.
    noise_factor : float
        Multiplier applied to Gaussian noise.
    ignore_categorical_similarity : bool
        Whether to ignore categorical features when computing similarity.
    num_bins : int
        Number of bins used for discretizing numeric columns.
    """
    def __init__(self,
                data: pd.DataFrame,
                target_column: str,
                relevance_values: pd.Series):
        """
        Parameters
        ----------
        data : pd.DataFrame
            Dataframe containing the data to be sampled.
        target_column : str
            Name of the target variable.
        relevance_values : pd.Series
            Series containing the respective relevance values for each row in `data`.
        """
        super().__init__(data, target_column, relevance_values)
        
    def run_sampling(self,
                    min_acceptable_relevance: float = 1.0,
                    max_acceptable_relevance: float = 1.0,
                    num_bins: int = 10,
                    allowed_bin_deviation: int = 1,
                    noise_factor: float = 0.01,
                    ignore_categorical_similarity: bool = False,
                    enable_undersampling: bool = True) -> pd.DataFrame:
        """
        Applies crbSMOGN to the given dataset, performing oversampling and optional undersampling.

        Parameters
        ----------
        min_acceptable_relevance : float, optional
            Minimum relevance to retain a sample as-is. Default 1.0.
        max_acceptable_relevance : float, optional
            Maximum relevance to retain a sample as-is. Default 1.0.
        num_bins : int, optional
            Number of bins for discretizing numeric features. Default 10.
        allowed_bin_deviation : int, optional
            Maximum deviation in binning for similarity. Default 1.
        noise_factor : float, optional
            Gaussian noise factor for numeric features. Default 0.01.
        ignore_categorical_similarity : bool, optional
            If True, ignore categorical features for similarity. Default False.
        enable_undersampling : bool, optional
            If True, remove low-relevance samples. Default True.

        Returns
        -------
        pd.DataFrame
            New dataset with oversampling and optional undersampling applied.
        Notes:
        -----
            - num_gaussian_samples: int, the number of samples generated using Gaussian noise.
            - num_knn_samples: int, the number of samples generated through interpolation with nearest neighbors.
            - num_dropped_samples: int, the number of samples dropped due to undersampling.
        """
        self.min_acceptable_relevance = min_acceptable_relevance
        self.max_acceptable_relevance = max_acceptable_relevance
        self.allowed_bin_deviation = allowed_bin_deviation
        self.ignore_categorical_similarity = ignore_categorical_similarity
        self.noise_factor = noise_factor
        self.num_bins = num_bins

        # Discretize the dataset for similarity calculations
        self.binned_data = self._discretize_dataset(
            self.data, self.num_bins, self.numeric_columns, self.categorical_columns, self.ignore_categorical_similarity).to_numpy()

        # Decide which samples to retain, oversample, or undersample
        self.sample_treatment = pd.Series(index=self.data.index, dtype=object)
        self.sample_treatment = self.relevance_values.apply(
            lambda relv: self._check_treatment(relv))

        # Identify samples that require oversampling
        self.oversample_indices = self.data.index.get_indexer(self.sample_treatment[self.sample_treatment > 1].index)
        total_oversample_count = (self.sample_treatment[self.sample_treatment > 1] - 1).sum()
        oversampled_data = np.zeros((total_oversample_count, self.data_numpy.shape[1]))

        # Counters for the number of samples generated by each method
        self.num_oversampled_samples = 0
        self.num_gaussian_samples = 0
        self.num_knn_samples = 0

        # Perform oversampling
        for idx in self.oversample_indices:
            num_replications = int(self.sample_treatment.iloc[idx]) - 1  # Adjust for existing sample

            # Generate synthetic samples using interpolation and Gaussian noise
            new_samples, gaussian_count, knn_count = self._oversample(reference_sample_index= int(idx),
                                                                num_replications=num_replications)


            # Store the new samples
            oversampled_data[self.num_oversampled_samples:self.num_oversampled_samples + num_replications, :] = new_samples
            self.num_oversampled_samples += num_replications
            self.num_gaussian_samples += gaussian_count
            self.num_knn_samples += knn_count

        # Postprocessing. Back to Pandas.
        oversampled_data = pd.DataFrame(data = oversampled_data, columns= self.data.columns, index= range(self.data.index.max(), self.data.index.max() + oversampled_data.shape[0]))
        oversampled_data = oversampled_data.astype(self.data.dtypes)
        new_dataset = pd.concat([self.data, oversampled_data], axis=0)

        # Perform undersampling if enabled
        if enable_undersampling:
            undersample_indices = self.sample_treatment[self.sample_treatment == 0].index
            new_dataset = new_dataset.drop(undersample_indices, axis=0)
            self.num_dropped_samples = len(undersample_indices)
        else:
            self.num_dropped_samples = 0

        return new_dataset



    def _check_treatment(self, relevance: float) -> int:
        """
        Determine the sampling treatment for a sample based on its relevance.

        This function decides the treatment for a sample based on its relevance. The possible outcomes are:
        - `1`: Keep the sample as is.
        - `0`: Drop the sample (undersample).
        - `x`: Replicate the sample `x-1` times (oversample).

        Parameters:
        ----------
        relevance : float
            The ratio based relevance of the sample.

        Returns:
        -------
        int
            A value indicating the treatment of the sample:
            - `1` if the sampling rate is between `min_acceptable_relevance` and `max_acceptable_relevance`.
            - `0` or a positive integer based on the rate and a random value if outside the acceptable range.
        """
        if self.min_acceptable_relevance < relevance < self.max_acceptable_relevance:
            return 1
        else:
            remainder = relevance % 1
            random_value = np.random.uniform()  # Generates a random number between [0, 1]
            if remainder > random_value:
                return int(np.ceil(relevance))
            else:
                return int(np.floor(relevance))
        


    def _oversample(self, reference_sample_index: int, 
                    num_replications: int) -> tuple[np.ndarray, int, int]:
        """
        Generate synthetic samples based on a reference sample using interpolation or Gaussian noise.

        Parameters
        ----------
        reference_sample_index : int
            Index of the reference sample in the dataset.
        num_replications : int
            Number of replications to be created of the sample.

        Returns
        -------
        Tuple[np.ndarray, bool]
            new_sample : np.ndarray
                The generated synthetic samples.
            interpolated : bool
                True if generated by interpolation, False if generated by Gaussian noise.
        """
        # get similar samples indice (rows) based on the discretized data
        similar_rows = self._get_similar_samples(
                index = reference_sample_index, 
                binned_data = self.binned_data, 
                numerical_mask =self.numerical_mask, 
                categorical_mask =self.categorical_mask, 
                allowed_bin_deviation = self.allowed_bin_deviation, 
                ignore_categoricals = self.ignore_categorical_similarity
            ) 
        # get the samples to the corresponding indices 
        similar_samples = self.data_numpy[similar_rows]

        # check if enough similar samples where found. If yes, sort them by distance. If not, use all for oversampling
        if similar_samples.shape[0] > num_replications:
            num_gaussian_samples = 0
            distances = self._heom_distance(x=self.data_numpy[reference_sample_index],
                                y=similar_samples,
                                feature_ranges= self.feature_ranges,
                                categorical_mask=self.categorical_mask,
                                numerical_mask=self.numerical_mask)
            nearest_neighbors = similar_samples[np.argsort(distances)[:num_replications]]
            num_knn_samples = num_replications
        else:
            nearest_neighbors = similar_samples
            num_knn_samples = nearest_neighbors.shape[0]
            num_gaussian_samples = num_replications - num_knn_samples
            
            

        new_samples = np.zeros(shape=(num_replications, self.data_numpy.shape[1])) # is initialized as object
        replication_counter = 0

        for neighbor_sample in nearest_neighbors:
            new_samples[replication_counter, :] = self._interpolate_sample(x=self.data_numpy[reference_sample_index],
                                                                    y=neighbor_sample,
                                                                    feature_ranges= self.feature_ranges,
                                                                    categorical_mask= self.categorical_mask,
                                                                    numerical_mask=self.numerical_mask)
            replication_counter += 1

        new_samples[replication_counter:, :] = self._add_gaussian_noise(x=self.data_numpy[reference_sample_index],
                                                                standard_deviations= self.standard_deviations,
                                                                noise_factor= self.noise_factor,
                                                                n_samples= num_gaussian_samples,
                                                                numerical_mask= self.numerical_mask,
                                                                categorical_mask=self.categorical_mask
                                                                )

        replication_counter += num_gaussian_samples

        return new_samples, num_gaussian_samples, num_knn_samples


def apply_crbsmogn(data: pd.DataFrame,
                   target_column: str,
                   relevance_values: pd.Series,
                   enable_undersampling: bool = True,
                   min_acceptable_relevance: float = 1.0,
                   max_acceptable_relevance: float = 1.0,
                   num_bins: int = 10,
                   allowed_bin_deviation: int = 1,
                   noise_factor: float = 0.01,
                   ignore_categorical_similarity: bool = False
                   ) -> pd.DataFrame:
    """
    Apply the crbSMOGN sampling technique to balance an imbalanced regression dataset.

    This function creates a new dataset by generating synthetic samples for minority
    cases using interpolation and Gaussian noise, and optionally removing majority
    cases through undersampling based on ratio-based relevance thresholds.

    Parameters
    ----------
    data : pd.DataFrame
        The input dataset to be balanced.
    target_column : str
        Name of the target variable column.
    relevance_values : pd.Series
        Relevance scores for each sample in the dataset, ranging from 0 to 1.
        Samples with relevance outside the [min_acceptable_relevance, max_acceptable_relevance]
        range will be considered for oversampling or undersampling.
    enable_undersampling : bool, optional
        If True, applies undersampling to remove low-relevance samples. Default is True.
    min_acceptable_relevance : float, optional
        Minimum relevance threshold for retaining samples as-is. Default is 1.0.
    max_acceptable_relevance : float, optional
        Maximum relevance threshold for retaining samples as-is. Default is 1.0.
    num_bins : int, optional
        Number of bins used for discretizing numeric features during similarity calculations. Default is 10.
    allowed_bin_deviation : int, optional
        Maximum bin difference allowed for considering two samples similar. Default is 1.
    noise_factor : float, optional
        Factor controlling the magnitude of Gaussian noise added to numeric features. Default is 0.01.
    ignore_categorical_similarity : bool, optional
        If True, ignores categorical feature differences when finding similar samples. Default is False.

    Returns
    -------
    pd.DataFrame
        A new balanced dataset with synthetic samples added and/or majority samples removed.

    Notes
    -----
    - The function internally uses the crbSMOGN class to perform the actual sampling.
    - Synthetic samples are generated either through interpolation with nearest neighbors
      or by adding Gaussian noise to existing samples.
    - The sampling process is controlled by the relevance thresholds provided.
    """
    logging.debug('Begin crbSMOGN...')
    sampler = crbSMOGN(data=data,
                      target_column=target_column,
                      relevance_values=relevance_values)
    new_data = sampler.run_sampling(min_acceptable_relevance=min_acceptable_relevance,
                                   max_acceptable_relevance=max_acceptable_relevance,
                                   num_bins=num_bins,
                                   allowed_bin_deviation=allowed_bin_deviation,
                                   noise_factor=noise_factor,
                                   ignore_categorical_similarity=ignore_categorical_similarity,
                                   enable_undersampling=enable_undersampling)
    logging.debug('Finished crbSMOGN...')
    return new_data