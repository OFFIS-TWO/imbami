import numpy as np
from imbami.utils.base_relevance_function import BaseRelevanceFunction

class DensityDistanceRelevance(BaseRelevanceFunction):
    """
    Relevance function based on the difference (distance) of empirical and relevance KDEs.
    """
    def __init__(self):
        """
        Initializes DensityDistanceRelevance.
        Inherits from BaseRelevanceFunction. No arguments are required.
        """
        super().__init__()
    def fit(self,
            emp_data: np.ndarray, 
            rel_data: np.ndarray | None,
            emp_bandwidth_type: str = 'silverman',
            rel_bandwidth_type: str = 'uniform',
            rel_bandwidth_factor: float = 1.0):
        """
        Fits KDEs to both empirical and relevance datasets.

        Parameters:
            emp_data (np.ndarray): Empirical data array.
            emp_bandwidth_type (str): Bandwidth selection method for empirical KDE ('silverman' or 'ISJ').
            rel_data (np.ndarray or None): Relevance data array, required if rel_bandwidth_type is not 'uniform'.
            rel_bandwidth_type (str): Bandwidth type for relevance KDE ('silverman', 'ISJ', or 'uniform').
            rel_bandwidth_factor (float): Scaling factor for relevance KDE bandwidth.
        """
        self.fit_to_data(emp_data, emp_bandwidth_type, rel_data, rel_bandwidth_type, rel_bandwidth_factor)
        # Get normalization
        density_dist = self.get_distance(self.emp_data)
        self.min_dist = np.min(density_dist)
        self.max_dist = np.max(density_dist)

    def eval(self, y: np.ndarray, centered: bool = True) -> np.ndarray:
        """
        Evaluate relevance scores.

        Parameters:
            y (np.ndarray): Input values.
            centered (bool): Whether to center the relevance around 0.5.

        Returns:
            np.ndarray: Relevance scores.
        """
        lamb = self.get_distance(y)
        if centered:
            relevance = self.prob_dist_to_centered_relevance(lamb)
        else:
            relevance = self.prob_dist_to_relevance(lamb)
        return relevance

    def get_distance(self, y: np.ndarray) -> np.ndarray:
        emp_density, rel_density = self.get_densities(y)
        prob_dist = emp_density - rel_density
        return prob_dist

    def prob_dist_to_relevance(self, lamb: np.ndarray) -> np.ndarray:
        """Calculate relevance linearly normalized between 0...1"""
        # Normalizing the series
        relv = (lamb - self.min_dist) / (self.max_dist - self.min_dist)
        # Invert and avoid zeros
        relv = np.maximum(1 - relv, 1E-6)
        # 1 = very rare
        # 0 = very frequent
        return relv

    def prob_dist_to_centered_relevance(self, lamb: np.ndarray) -> np.ndarray:
        """Calculate the relevnace between 0...1 centered around 0.5."""       
        relv = np.empty_like(lamb)
        relv[lamb < 0] = 0.5 - 0.5 * (lamb[lamb < 0] / self.min_dist)
        relv[lamb >= 0] = 0.5 + 0.5 * (lamb[lamb >= 0] / self.max_dist)
        # Invert and avoid zeros
        relv = np.maximum(1 - relv, 1E-6)
        # 1 = very rare
        # 0 = very frequent
        return relv


