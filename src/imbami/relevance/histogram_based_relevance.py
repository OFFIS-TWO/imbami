import numpy as np
from .base_relevance_function import RelevanceFunctionBase

class HistogramBasedRelevance(RelevanceFunctionBase):
    '''
    Calculate the Histogram-based relevance as defined in the article "Histogram Approaches for Imbalanced Data Streams Regression"
    by Ehsan Aminian, Rita P. Ribeiro and Joao Gama 
    '''
    def __init__(self) -> None:
        self.is_fitted = False
        self.type = "HistogramBasedRelevance"

    def fit(self, data: np.ndarray, bins: int = 10) -> None:
        self.data = data
        self.bins = bins
        self.bin_counts, self.bin_boundaries = np.histogram(data, bins = bins, density=False)
        self.max_bin_count = np.max(self.bin_counts)

        # normalize bin counts
        self.relative_bin_counts = self.bin_counts / self.max_bin_count
        # invert such that rare bins are most relevant.
        self.bin_relevance = np.maximum(1- self.relative_bin_counts, 1E-6)
        self.is_fitted = True

    def eval(self, y: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("eval() can not be called before fit().")
        # determine bin indices
        indices = np.digitize(y, bins=self.bin_boundaries) - 1 # subtract 1 so first bin is index 0
        # Create a valid mask: mask samples that are out of distribution; only indices within [0, len(bin_counts)-1]
        valid_mask = (indices >= 0) & (indices < len(self.bin_counts))
        # Assign relevance
        relevance = np.zeros_like(y, dtype=float)
        relevance[valid_mask] = self.bin_relevance[indices[valid_mask]]
        return relevance