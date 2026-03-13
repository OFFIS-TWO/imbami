import pandas as pd
import numpy as np
import logging
from .base_mitigation_method import SamplingMethodBase

class WERCS(SamplingMethodBase):
    """
    WEighted Relevance-based Combination Strategy (WERCS) for imbalanced regression datasets.

    This class performs both oversampling and undersampling based on sample weights/relevance values.
    Samples with higher weights (more relevant) are more likely to be oversampled, while samples
    with lower weights (less relevant) are more likely to be undersampled.

    Attributes
    ----------
    data : pd.DataFrame
        The dataset to be sampled.
    relevance_values : pd.Series
        Weights/relevance values for each sample, typically in [0, 1].
    """

    def __init__(self,
                 data: pd.DataFrame,
                 target_column: str,
                 relevance_values: pd.Series):
        """
        Initialize the WERCS sampler.

        Parameters
        ----------
        data : pd.DataFrame
            Dataframe containing the data to be sampled.
        target_column : str
            Name of the target variable. Not used for sampling but required per sampler template.
        relevance_values : pd.Series
            Series containing the respective weights/relevance values for each row in `data`.
            Higher values indicate more relevant samples.
        """
        super().__init__(data=data, target_column= target_column, relevance_values=relevance_values)

        # Validate sample relevance
        if not ((relevance_values >= 0) & (relevance_values <= 1)).all():
            # Normalize sample relevance
            self.relevance_values = self._normalize_with_clipping(self.relevance_values)

    def run_sampling(self,
                    oversample_rate: float = 0.5,
                    undersample_rate: float = 0.5,
                    enable_undersampling: bool = True,
                    random_state: int|None = 0) -> pd.DataFrame:
        """
        Apply WERCS to the dataset, performing oversampling and optional undersampling.

        Parameters
        ----------
        oversample_rate : float, optional
            Fraction of dataset size to generate via oversampling. Default is 0.5.
        undersample_rate : float, optional
            Fraction of dataset size to remove via undersampling. Default is 0.5.
        enable_undersampling : bool, optional
            If True, perform undersampling. Default is True.
        random_state : int | None, optional
            Random seed for reproducibility. Default is 0.

        Returns
        -------
        pd.DataFrame
            New dataset with samples added via oversampling and removed via undersampling (if enabled).

        Notes
        -----
        - The number of samples added/removed is determined by the rates relative to the original dataset size.
        - Oversampling simply duplicates existing samples based on their weights.
        - Undersampling removes existing samples based on their weights.
        """
        rng = self._get_random_generator(random_state)
        data_numpy = self.data.to_numpy()
        weights = self.relevance_values.to_numpy()

        # Oversampling
        n_oversample = int(data_numpy.shape[0] * oversample_rate)
        oversampled_data = np.zeros((n_oversample, data_numpy.shape[1]))
        num_oversampled_samples = 0

        while num_oversampled_samples < n_oversample:
            # Pick random index and the respective weight
            random_index = rng.integers(0, weights.shape[0])
            random_weight = weights[random_index]  # High value indicates high relevance

            # Oversample if weight is larger than random
            if random_weight > rng.uniform(low=0, high=1):
                oversampled_data[num_oversampled_samples:num_oversampled_samples + 1, :] = data_numpy[random_index]
                num_oversampled_samples += 1

        # Postprocessing: convert back to pandas DataFrame
        oversampled_data = pd.DataFrame(
            data=oversampled_data,
            columns=self.data.columns,
            index=range(self.data.index.max() + 1, self.data.index.max() + 1 + oversampled_data.shape[0])
        )
        for col in self.data.columns:
            oversampled_data[col] = oversampled_data[col].astype(self.data[col].dtype)

        # Undersampling
        if enable_undersampling:
            n_undersample = int(data_numpy.shape[0] * undersample_rate)
            num_undersampled_samples = 0
            undersample_indice = -np.ones(n_undersample, dtype=int)

            while num_undersampled_samples < n_undersample:
                # Pick random index and the respective weight
                random_index = rng.integers(0, weights.shape[0])
                random_weight = weights[random_index]  # Low value indicates low relevance

                # Undersample if weight is lower than random
                if random_weight < rng.uniform(low=0, high=1):
                    if random_index not in undersample_indice:
                        undersample_indice[num_undersampled_samples] = random_index
                        num_undersampled_samples += 1

            undersampled_data = self.data.drop(index=self.data.iloc[undersample_indice].index.tolist())
            new_data = pd.concat([undersampled_data, oversampled_data], axis=0)
        else:
            new_data = pd.concat([self.data, oversampled_data], axis=0)

        return new_data
    

def apply_wercs(data: pd.DataFrame,
                target_column: str,
               relevance_values: pd.Series,
               enable_undersampling: bool = True,
               oversample_rate: float = 0.5,
               undersample_rate: float = 0.5,
               random_state: int|None = 0) -> pd.DataFrame:
    """
    Apply the WERCS (WEighted Relevance-based Combination Strategy) sampling technique
    to balance an imbalanced regression dataset.

    This function creates a new dataset by duplicating existing samples based on their
    relevance weights and optionally removing samples with low relevance weights.

    Parameters
    ----------
    data : pd.DataFrame
        The input dataset to be balanced.
    relevance_values : pd.Series
        Relevance weights for each sample in the dataset, ranging from 0 to 1.
        Samples with higher weights are more likely to be duplicated, while samples
        with lower weights are more likely to be removed.
    enable_undersampling : bool, optional
        If True, applies undersampling to remove low-relevance samples. Default is True.
    oversample_rate : float, optional
        Proportion of the original dataset size to generate via oversampling. Default is 0.5.
    undersample_rate : float, optional
        Proportion of the original dataset size to remove via undersampling. Default is 0.5.
    random_state : int | None, optional
        Random seed for reproducibility. Default is 0.

    Returns
    -------
    pd.DataFrame
        A new balanced dataset with samples added via oversampling and/or removed via undersampling.

    Notes
    -----
    - The function internally uses the WERCS class to perform the actual sampling.
    - Oversampling simply duplicates existing samples based on their relevance weights.
    - Undersampling removes existing samples based on their relevance weights.
    - Samples with higher relevance weights are more likely to be duplicated.
    - Samples with lower relevance weights are more likely to be removed.
    """
    logging.debug('Begin WERCS...')
    sampler = WERCS(data=data,
                   relevance_values=relevance_values, target_column=target_column)
    new_data = sampler.run_sampling(oversample_rate=oversample_rate,
                                   undersample_rate=undersample_rate,
                                   enable_undersampling=enable_undersampling,
                                   random_state= random_state)
    logging.debug('Finished WERCS...')
    return new_data