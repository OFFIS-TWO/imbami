import numpy as np
from typing import Callable
import pandas as pd
from typing import Literal
from imbami import DensityRatioRelevance

def get_density_ratio_lambda(
            emp_density_mode: Literal['fit_kde', 'provide_pdf'],
            domain_density_mode: Literal['fit_kde', 'provide_pdf'],
            data: np.ndarray,
            emp_bw_type: None | str = 'silverman',
            emp_bw_factor: None | float = 1.0,
            emp_kernel_type: None | str = 'gaussian',
            emp_pdf: None | Callable = None,

            domain_data: None | np.ndarray = None,
            domain_bw_type: None | str = 'uniform',
            domain_bw_factor: None | float = 1.0,
            domain_kernel_type: None | str = 'gaussian',
            domain_pdf: None | Callable = None,
            
            grid_points: int = 4096) -> np.ndarray:
    
    """
    Calculate density ratio (Lambda) between empirical and domain distributions.

    Computes the density ratio (empirical density / domain density) for the input data
    using either Kernel Density Estimation (KDE) or provided probability density functions.

    Parameters
    ----------
    emp_density_mode : {'fit_kde', 'provide_pdf'}
        Method for empirical density estimation.
    domain_density_mode : {'fit_kde', 'provide_pdf'}
        Method for domain density estimation.
    data : np.ndarray
        Empirical data samples for which to calculate density ratios.
    emp_bw_type : str or None, default='silverman'
        Bandwidth selection method for empirical KDE ('silverman', 'ISJ', or 'uniform').
    emp_bw_factor : float or None, default=1.0
        Scaling factor for empirical KDE bandwidth.
    emp_kernel_type : str or None, default='gaussian'
        Kernel type for empirical KDE (e.g., 'gaussian').
    emp_pdf : callable or None, default=None
        Callable PDF for empirical distribution. Required if emp_density_mode is 'provide_pdf'.
    domain_data : np.ndarray or None, default=None
        Domain data samples. Required if domain_density_mode is 'fit_kde' and domain_bw_type is not 'uniform'.
    domain_bw_type : str or None, default='uniform'
        Bandwidth selection method for domain KDE ('silverman', 'ISJ', or 'uniform').
    domain_bw_factor : float or None, default=1.0
        Scaling factor for domain KDE bandwidth.
    domain_kernel_type : str or None, default='gaussian'
        Kernel type for domain KDE (e.g., 'gaussian').
    domain_pdf : callable or None, default=None
        Callable PDF for domain distribution. Required if domain_density_mode is 'provide_pdf'.
    grid_points : int, default=4096
        Number of points in the evaluation grid for KDE.

    Returns
    -------
    np.ndarray
        Density ratio (Lambda) values for each input data point.

    Notes
    -----
    When domain_bw_type is 'uniform', domain_data is not required as a uniform
    distribution will be used for the domain density.
    """

    drr = DensityRatioRelevance(emp_density_mode=emp_density_mode,
                                domain_density_mode= domain_density_mode)
    
    drr.fit(data= data, 
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
    lamb = drr._get_ratio(data)
    return lamb




def imbalance_ratio(lamb: pd.Series) -> pd.Series:
    '''
    Calculates the imbalance ratio (IR) from the probability ratio (Lambda)

    Parameters:
    - lamb (pd.Series): Probability ratio values.

    Returns:
    - pd.Series: Imbalance ratio for each value in the input.
    '''
    ir = (lamb.apply(np.log).abs()).apply(np.exp)
    return ir