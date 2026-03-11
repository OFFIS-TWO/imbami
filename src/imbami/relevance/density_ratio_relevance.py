import numpy as np
from .base_relevance_function import EmpiricalDomainRelevanceFunctionBase
from typing import Literal
from typing import Callable

class DensityRatioRelevance(EmpiricalDomainRelevanceFunctionBase):
    """
    Relevance function based on density ratios (empirical / domain).

    This class calculates relevance scores by comparing the density of empirical data
    to the density of domain/relevance data. The relevance score is the inverse of the
    density ratio (domain density / empirical density).

    Public Interface:
        - __init__(): Initialize the relevance function
        - fit(): Fit the density estimators to data
        - eval(): Evaluate relevance scores for new data
        - __call__(): Alias for eval() for direct calling
    """
    def __init__(self,
                 emp_density_mode: Literal['fit_kde', 'provide_pdf'],
                 domain_density_mode: Literal['fit_kde', 'provide_pdf']):
        """
        Initialize DensityRatioRelevance.

        Args:
            emp_density_mode: Method for empirical density estimation ('fit_kde' or 'provide_pdf').
            domain_density_mode: Method for domain density estimation ('fit_kde' or 'provide_pdf').
        """
        super().__init__(emp_density_mode = emp_density_mode,
                         domain_density_mode= domain_density_mode)
        self.type = "DensityRatioRelevance"
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
            
            grid_points: int = 4096):
        """
        Fit density estimators to both empirical and domain datasets.

        Configures either KDEs or provided PDFs for both empirical and domain distributions
        based on the specified density modes.

        Args:
            data: Empirical data samples. Required if emp_density_mode is 'fit_kde'.
            emp_bw_type: Bandwidth selection method for empirical KDE ('silverman', 'ISJ', or 'uniform').
            emp_bw_factor: Scaling factor for empirical KDE bandwidth.
            emp_kernel_type: Kernel type for empirical KDE (e.g., 'gaussian').
            emp_pdf: Callable PDF for empirical distribution. Required if emp_density_mode is 'provide_pdf'.

            domain_data: Domain data samples. Required if domain_density_mode is 'fit_kde'.
            domain_bw_type: Bandwidth selection method for domain KDE ('silverman', 'ISJ', or 'uniform').
            domain_bw_factor: Scaling factor for domain KDE bandwidth.
            domain_kernel_type: Kernel type for domain KDE (e.g., 'gaussian').
            domain_pdf: Callable PDF for domain distribution. Required if domain_density_mode is 'provide_pdf'.

        Note:
            When domain_bw_type is 'uniform', domain_data is not required as a uniform distribution
            will be used for the domain density.
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

    def eval(self, y: np.ndarray) -> np.ndarray:
        """
        Evaluate relevance scores for input values.

        The relevance score is calculated as the inverse of the density ratio
        (domain density / empirical density).

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
        lamb = self._get_ratio(y)
        relevance = 1/ lamb
        return relevance

    def _get_ratio(self, y: np.ndarray) -> np.ndarray:
        """
        Calculate density ratio for input values.

        Args:
            y: Input values for which to compute density ratio.

        Returns:
            Density ratio (empirical density / domain density).

        Note:
            This is an internal method used by eval().
        """
        emp_density, rel_density = self._get_densities_from_grid(y)
        lamb = np.divide(emp_density, rel_density)
        return lamb