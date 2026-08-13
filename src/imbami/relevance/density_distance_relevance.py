import numpy as np
from .base_relevance_function import EmpiricalDomainRelevanceFunctionBase
from typing import Literal
from typing import Callable

class DensityDistanceRelevance(EmpiricalDomainRelevanceFunctionBase):
    """
    Relevance function based on the difference (distance) between empirical and domain KDEs.

    This class calculates relevance scores by measuring the distance between empirical and
    domain density estimates. The scores can be either linearly normalized between 0 and 1,
    or centered around 0.5 to distinguish between rare and common values.

    Public Interface:
        - __init__(): Initialize the relevance function
        - fit(): Fit the density estimators to data
        - eval(): Evaluate relevance scores for new data
        - __call__(): Alias for eval() for direct calling
    """
    def __init__(self,
                 emp_density_mode: Literal['fit_kde', 'provide_pdf'],
                 domain_density_mode: Literal['fit_kde', 'provide_pdf'],
                 centered: bool = True):
        """
        Initialize DensityDistanceRelevance.

        Args:
            - emp_density_mode: Method for empirical density estimation ('fit_kde' or 'provide_pdf').
            - domain_density_mode: Method for domain density estimation ('fit_kde' or 'provide_pdf').
            - centered: If True, relevance scores are centered around 0.5 (values < 0.5 are common,
                     values > 0.5 are rare). If False, scores are linearly normalized between 0 and 1.

        Note:
            Global centering and normalization to [0,1] of relevance only works if emp_density_mode is 'fit_kde'.
            With 'provide_pdf' centering and normalization is done for each .eval() call individually.
        """
        super().__init__(emp_density_mode = emp_density_mode,
                         domain_density_mode= domain_density_mode)
        
        self.centered = centered
        self.type = "DensityDistanceRelevance"
        
    def fit(self,
            data: None | np.ndarray,
            emp_bw_type: None | str = 'silverman',
            emp_bw_factor: None | float = 1.0,
            emp_kernel_type: None | str = 'gaussian',
            emp_pdf: None | Callable = None,

            domain_data: None | np.ndarray = None,
            domain_bw_type: None | str = 'uniform',
            domain_bw_factor: None | float = 1.0,
            domain_kernel_type: None | str = 'gaussian',
            domain_pdf: None | Callable = None,
            
            grid_points: int = 4096)  -> None:
        """
        Fit density estimators to both empirical and domain datasets.

        Configures either KDEs or provided PDFs for both empirical and domain distributions
        based on the specified density modes. Also calculates normalization parameters for
        relevance scoring.

        Args:
            - data: Empirical data samples. Required if emp_density_mode is 'fit_kde'.
            - emp_bw_type: Bandwidth selection method for empirical KDE ('silverman', 'ISJ', or 'uniform').
            - emp_bw_factor: Scaling factor for empirical KDE bandwidth.
            - emp_kernel_type: Kernel type for empirical KDE (e.g., 'gaussian').
            - emp_pdf: Callable PDF for empirical distribution. Required if emp_density_mode is 'provide_pdf'.

            - domain_data: Domain data samples. Required if domain_density_mode is 'fit_kde'.
            - domain_bw_type: Bandwidth selection method for domain KDE ('silverman', 'ISJ', or 'uniform').
            - domain_bw_factor: Scaling factor for domain KDE bandwidth.
            - domain_kernel_type: Kernel type for domain KDE (e.g., 'gaussian').
            - domain_pdf: Callable PDF for domain distribution. Required if domain_density_mode is 'provide_pdf'.

        Note:
            When domain_bw_type is 'uniform', domain_data is still required to determine upper and lower domain bounds. The distribution itself is not used.
            Set domain_data = data.
        """
        self._fit_to_data(emp_data= data, 
                          emp_bw_type = emp_bw_type, 
                          emp_bw_factor= emp_bw_factor,
                          emp_kernel_type = emp_kernel_type,
                          domain_data = domain_data, 
                          domain_bw_type = domain_bw_type, 
                          domain_bw_factor = domain_bw_factor,
                          domain_kernel_type= domain_kernel_type,
                          emp_pdf= emp_pdf,
                          domain_pdf= domain_pdf,
                          grid_points=grid_points)
        super().fit()
        # calibrate normalization
        if self.emp_density_mode == 'fit_kde':
            min_support = self.emp_grid_x.min()
            max_support = self.emp_grid_x.max()
            range_support = max_support - min_support
            extended_value = np.linspace(min_support - 0.1*range_support, max_support + 0.1*range_support, 1001) # get evenly spaced points between the extended ends of the target distribution
            lamb = self._get_distance(extended_value) 
            self.min_dist = np.min(lamb)
            self.max_dist = np.max(lamb)

    def eval(self, y: np.ndarray) -> np.ndarray:
        """
        Evaluate relevance scores for input values.

        The relevance score measures how different the empirical density is from the domain density.
        Scores can be either linearly normalized between 0 and 1, or centered around 0.5.

        Args:
            y: Input values for which to compute relevance scores.

        Returns:
            Computed relevance scores for the input values.

        Raises:
            RuntimeError: If called before fit().
            ValueError: If input y is invalid.
        """
        if not self.is_fitted:
            raise RuntimeError("eval() can not be called before fit().")
        lamb = self._get_distance(y)
        if self.emp_density_mode == 'provide_pdf':
            self.min_dist = np.min(lamb)
            self.max_dist = np.max(lamb)
        if self.centered:
            relevance = self._prob_dist_to_centered_relevance(lamb)
        else:
            relevance = self._prob_dist_to_relevance(lamb)
        return relevance

    def _get_distance(self, y: np.ndarray) -> np.ndarray:
        """
        Calculate density distance for input values.

        Args:
            y: Input values for which to compute density distance.

        Returns:
            Density distance (empirical density - domain density).

        Note:
            This is an internal method used by eval().
        """
        emp_density, rel_density = self._get_densities_from_grid(y)
        prob_dist = emp_density - rel_density
        return prob_dist

    def _prob_dist_to_relevance(self, lamb: np.ndarray) -> np.ndarray:
        """
        Convert density distance to linearly normalized relevance scores.

        The relevance scores are normalized between 0 and 1, where:
        - 1 represents very rare values
        - 0 represents very frequent values

        Args:
            lamb: Density distance values to convert.

        Returns:
            Linearly normalized relevance scores between 0 and 1.
        """
        # Normalizing the series
        relv = (lamb - self.min_dist) / (self.max_dist - self.min_dist)
        # Invert and avoid zeros
        relv = np.maximum(1 - relv, 1E-6)
        # 1 = very rare
        # 0 = very frequent
        return relv

    def _prob_dist_to_centered_relevance(self, lamb: np.ndarray) -> np.ndarray:
        """
        Convert density distance to centered relevance scores.

        The relevance scores are centered around 0.5, where:
        - Values < 0.5 represent common values
        - Values > 0.5 represent rare values
        - 0.5 represents neutral relevance

        Args:
            lamb: Density distance values to convert.

        Returns:
            Centered relevance scores between 0 and 1.
        """      
        relv = np.empty_like(lamb)
        relv[lamb < 0] = 0.5 - 0.5 * (lamb[lamb < 0] / self.min_dist)
        relv[lamb >= 0] = 0.5 + 0.5 * (lamb[lamb >= 0] / self.max_dist)
        # Invert and avoid zeros
        relv = np.maximum(1 - relv, 1E-6)
        # 1 = very rare
        # 0 = very frequent
        return relv


