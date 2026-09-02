import pandas as pd
import numpy as np
from typing import Callable
from typing import Literal

from .imbalance import sample_imbalance_ratio, sample_density_ratio


def imbalanced_sample_percentage(
        emp_density_mode: Literal['fit_kde', 'provide_pdf'],
        domain_density_mode: Literal['fit_kde', 'provide_pdf'],
        data: pd.Series,
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
        ir_bound: None| float = None,
        lower_bound: None| float = None,
        upper_bound: None| float = None) -> float:


    """
    Calculate percentage of imbalanced samples in a dataset (ISP).

    Computes the percentage of samples that are considered imbalanced based on
    either a direct density ratio threshold or an imbalance ratio threshold.

    Parameters
    ----------
    emp_density_mode : {'fit_kde', 'provide_pdf'}
        Method for empirical density estimation.
    domain_density_mode : {'fit_kde', 'provide_pdf'}
        Method for domain density estimation.
    data : pd.Series
        Target values for which to calculate imbalance percentage.
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
    ir_bound : float or None, default=None
        Imbalance ratio threshold. If provided, samples with IR > ir_bound are considered imbalanced.
    lower_bound : float or None, default=None
        Lower density ratio threshold. Samples with density ratio < lower_bound are considered imbalanced.
    upper_bound : float or None, default=None
        Upper density ratio threshold. Samples with density ratio > upper_bound are considered imbalanced.

    Returns
    -------
    float
        Percentage of samples in the dataset that are considered imbalanced.


    Notes
    -----
    Exactly one of ir_bound, lower_bound, or upper_bound must be provided.
    When domain_bw_type is 'uniform', domain_data is not required.
    """

    if ir_bound is None and lower_bound is None and upper_bound is None:
        raise ValueError("At least one of ir_bound, lower_bound, or upper_bound must be provided.")
    if ir_bound is not None and (lower_bound is not None or upper_bound is not None):
        raise ValueError("lower_bound and upper_bound must be None if ir_bound is defined.")


    if ir_bound is None:
        # Eq. 6/7/8: thresholds are defined on the raw density ratio Lambda.
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
 
        isp_l = 0.0
        isp_u = 0.0
        if lower_bound is not None:
            isp_l = (lamb < lower_bound).sum() / len(lamb) * 100
        if upper_bound is not None:
            isp_u = (lamb > upper_bound).sum() / len(lamb) * 100
        isp = isp_l + isp_u
 
    else:
        # Eq. 10: threshold is defined on the imbalance ratio IR.
        ir = sample_imbalance_ratio(
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
        isp = (ir > ir_bound).sum() / len(ir) * 100
 
    return isp


def mean_imbalance_ratio(
        data: pd.Series,
        emp_density_mode: Literal['fit_kde', 'provide_pdf'] ='fit_kde',
        domain_density_mode: Literal['fit_kde', 'provide_pdf']= 'fit_kde',
        emp_bw_type: None | str = 'silverman',
        emp_bw_factor: None | float = 1.0,
        emp_kernel_type: None | str = 'gaussian',
        emp_pdf: None | Callable = None,
        domain_data: None | np.ndarray | pd.Series = None,
        domain_bw_type: None | str = 'uniform',
        domain_bw_factor: None | float = 1.0,
        domain_kernel_type: None | str = 'gaussian',
        domain_pdf: None | Callable = None,
        grid_points: int = 4096,) -> float:
    """
    Calculate mean imbalance ratio (mIR) for a dataset.

    Computes the average imbalance ratio across all samples in the dataset,
    where imbalance ratio is defined as exp(|log(density_ratio)|).

    Parameters
    ----------
    data : pd.Series
        Target values for which to calculate mean imbalance ratio.
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
    float
        Mean imbalance ratio across all samples in the dataset.

    Raises
    ------
    TypeError
        If data is not a pandas Series.

    Notes
    -----
    When domain_bw_type is 'uniform', domain_data is not required as a uniform
    distribution will be used for the domain density.
    """

    ir = sample_imbalance_ratio(
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
 
    return float(np.mean(ir))










