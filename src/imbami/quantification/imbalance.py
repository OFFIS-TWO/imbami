import numpy as np
from typing import Callable
import pandas as pd
from typing import Literal
from imbami import DensityRatioRelevance

def sample_density_ratio(
        data: pd.Series,
        emp_density_mode: Literal['fit_kde', 'provide_pdf'] = 'fit_kde',
        domain_density_mode: Literal['fit_kde', 'provide_pdf'] = 'fit_kde',
        emp_bw_type: None | str = 'silverman',
        emp_bw_factor: None | float = 1.0,
        emp_kernel_type: None | str = 'gaussian',
        emp_pdf: None | Callable = None,
        domain_data: None | np.ndarray | pd.Series = None,
        domain_bw_type: None | str = 'uniform',
        domain_bw_factor: None | float = 1.0,
        domain_kernel_type: None | str = 'gaussian',
        domain_pdf: None | Callable = None,
        grid_points: int = 4096) -> pd.Series:
    """
    Per-sample density ratio Lambda between empirical and domain distributions, indexed like `data`.
  
    Computes the density ratio (empirical density / domain density) for the input data
    using either Kernel Density Estimation (KDE) or provided probability density functions.
    Lambda(x_i) = f_x(x_i) / f_r(x_i). Lambda < 1 indicates the sample is
    underrepresented relative to the relevance distribution, Lambda > 1
    indicates it is overrepresented.
 
    Parameters
    ----------
    data : pd.Series
        Target values for which to calculate the density ratio.
    emp_density_mode : {'fit_kde', 'provide_pdf'}, default='fit_kde'
        Method for empirical density estimation.
    domain_density_mode : {'fit_kde', 'provide_pdf'}, default='fit_kde'
        Method for domain density estimation.
    emp_bw_type : str or None, default='silverman'
        Bandwidth selection method for empirical KDE.
    emp_bw_factor : float or None, default=1.0
        Scaling factor for empirical KDE bandwidth.
    emp_kernel_type : str or None, default='gaussian'
        Kernel type for empirical KDE.
    emp_pdf : callable or None, default=None
        Callable PDF for empirical distribution.
    domain_data : np.ndarray, pd.Series, or None, default=None
        Domain data samples.
    domain_bw_type : str or None, default='uniform'
        Bandwidth selection method for domain KDE.
    domain_bw_factor : float or None, default=1.0
        Scaling factor for domain KDE bandwidth.
    domain_kernel_type : str or None, default='gaussian'
        Kernel type for domain KDE.
    domain_pdf : callable or None, default=None
        Callable PDF for domain distribution.
    grid_points : int, default=4096
        Number of points in the evaluation grid for KDE.
 
    Returns
    -------
    pd.Series
        Density ratio (Lambda) for each sample in `data`, indexed like `data`.
 
    Raises
    ------
    TypeError
        If data is not a pandas Series.
 
    Notes
    -----
    When domain_bw_type is 'uniform', domain_data is not required as a uniform
    distribution will be used for the domain density.
    """
    if not isinstance(data, pd.Series):
        raise TypeError("The argument 'data' is not of type pd.Series.")
 
    if domain_density_mode == 'fit_kde' and domain_data is None and domain_bw_type == 'uniform':
        domain_data = data
    if isinstance(domain_data, pd.Series):
        domain_data = domain_data.to_numpy()
 
    drr = DensityRatioRelevance(emp_density_mode=emp_density_mode,
                                domain_density_mode= domain_density_mode)
    
    drr.fit(data= data.to_numpy(), 
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
    lamb = drr._get_ratio(data.to_numpy())
 
    return pd.Series(lamb, index=data.index)





def sample_imbalance_ratio(
        data: pd.Series,
        emp_density_mode: Literal['fit_kde', 'provide_pdf'] = 'fit_kde',
        domain_density_mode: Literal['fit_kde', 'provide_pdf'] = 'fit_kde',
        emp_bw_type: None | str = 'silverman',
        emp_bw_factor: None | float = 1.0,
        emp_kernel_type: None | str = 'gaussian',
        emp_pdf: None | Callable = None,
        domain_data: None | np.ndarray | pd.Series = None,
        domain_bw_type: None | str = 'uniform',
        domain_bw_factor: None | float = 1.0,
        domain_kernel_type: None | str = 'gaussian',
        domain_pdf: None | Callable = None,
        grid_points: int = 4096,
        directional: bool = False) -> pd.Series:
    """
    Per-sample imbalance ratio (IR), indexed like `data`.

    IR(x_i) = exp(|log(Lambda(x_i))|), IR in [1, inf).

    Set `directional=True` to get a signed IR: positive for over-represented
    samples, negative for under-represented samples, magnitude unchanged.
    Use `directional=False` (default) for the classic absolute IR.
    """
    lamb = sample_density_ratio(
        data=data,
        emp_density_mode=emp_density_mode,
        domain_density_mode=domain_density_mode,
        emp_bw_type=emp_bw_type,
        emp_bw_factor=emp_bw_factor,
        emp_kernel_type=emp_kernel_type,
        emp_pdf=emp_pdf,
        domain_data=domain_data,
        domain_bw_type=domain_bw_type,
        domain_bw_factor=domain_bw_factor,
        domain_kernel_type=domain_kernel_type,
        domain_pdf=domain_pdf,
        grid_points=grid_points)
 
    return imbalance_ratio(lamb, directional=directional)





def imbalance_ratio(lamb: pd.Series, directional: bool = False) -> pd.Series:
    '''
    Calculates the imbalance ratio (IR) from the probability ratio (Lambda).

    Parameters:
    - lamb (pd.Series): Probability ratio values.
    - directional (bool): If False (default), returns the undirected IR,
      exp(|log(lambda)|) >= 1, same magnitude regardless of over-/under-representation.
      If True, returns a signed IR: positive when the sample is over-represented
      (lambda > 1), negative when under-represented (lambda < 1), i.e.
      lambda if lambda >= 1, else -1/lambda. Magnitude matches the undirected IR.

    Returns:
    - pd.Series: (Signed or unsigned) imbalance ratio for each value in the input.
    '''
    log_lamb = lamb.apply(np.log)
    magnitude = log_lamb.abs().apply(np.exp)
    if directional:
        return magnitude * np.sign(log_lamb)
    return magnitude