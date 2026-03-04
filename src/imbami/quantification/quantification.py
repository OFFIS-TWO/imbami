import pandas as pd
import numpy as np
from typing import Callable

from .imbalance import imbalance_ratio, probability_ratio


def imbalanced_sample_percentage(
    data: pd.Series | pd.DataFrame,
    relevance_pdf: Callable  | list[Callable] | None = None,
    discrete: bool | list[bool] = False,
    drop_inf_nan: bool = True,  
    ir_bound: None| float = None,
    lower_bound: None| float = None,
    upper_bound: None| float = None,
    kde_bandwidth: str | float = 'silverman',
    kde_kernel: str = 'gaussian',
    kde_grid_points: int = 2**14) -> float:
    """
    Calculates the percentage of imbalanced samples based on the provided criteria.
    The criteria are either
    - ir_bound: According to Equation (4) in the article. All samples are considered to be imbalanced whos Imbalance Ratio is higher than the provided threshold. The minimum achievable Imbalance ratio is 1,
                which indicates no imbalance to be present. An ir_bound of 2 indicates that only samples are counted which occure twice or half as often as the provided relevance_pdf (Uniform if relevance_pdf=None).
    - lower_bound/upper_bound: According to Equation (3) in the article. All samples are considered imbalanced whos probability ratio is smaller/higher then the provided bound. If both bounds are provided the
                the imbalanced sample percentages are summed.

    The Imbalance ratio is the absolute magnitude of the probability ratio lambda. IR = exp(abs(log(lambda(x)))).
    Thus ir_bound = 2 equals the combined behaviour of lower_bound = 0.5 and upper_bound = 2.

    Args:
    - data (pd.Series | pd.DataFrame): Input data for which to compute the probability ratio.
    - relevance_pdf (Callable | list[Callable] | None, optional): Probability density function(s) for relevance probability.
      - If a Series is passed to `data`, `relevance_pdf` should be a single callable function.
      - If a DataFrame is passed to `data`, `relevance_pdf` should be a list of callable functions with one function per column in data.
      - It is assumed that each callable function returns a value of type float.
      - If None, default functions will be used for relevance probability, which assumes uniform distribution of the relevance.
    - discrete (bool | list[bool], optional): Indicates if the data is discrete.
      - If a Series is passed to `data`, `discrete` should be a single boolean value.
      - If a DataFrame is passed to `data`, `discrete` should be a list of boolean values with one value per column.
    - drop_inf_nan (bool, optional): Whether to drop infinite and NaN values from `data` before computation.
      Default is True.
    - ir_bound: Imbalance ratio threshold for considering samples imbalanced. If defined, excludes lower_bound and upper_bound.
    - lower_bound: Lower threshold for probability ratio lamb.
    - upper_bound: Upper threshold for probability ratio lamb.
    - kde_bandwidth (str | float): Bandwidth for KDE. Default is 'silverman'.
    - kde_kernel (str): Kernel type for KDE. Default is 'gaussian'.
    - kde_grid_points (int): Number of points in the KDE evaluation grid. Default is 2^14.

    Returns:
    - (float): Percentage of imbalanced samples based on the given criteria.
    """
    if not isinstance(data, (pd.Series, pd.DataFrame)):
        raise TypeError("The argument 'data' is not of type pd.Series or pd.DataFame.")
    if ir_bound is None and lower_bound is None and upper_bound is None:
        raise ValueError("At least one of ir_bound, lower_bound, or upper_bound must be provided.")
    if ir_bound is not None and (lower_bound is not None or upper_bound is not None):
        raise ValueError("lower_bound and upper_bound must be None if ir_bound is defined.")


    lamb = probability_ratio(data, relevance_pdf, discrete, drop_inf_nan,
                            kde_bandwidth, kde_kernel, kde_grid_points)
    if ir_bound is None:
        isp_l = 0.0
        isp_u = 0.0
        if lower_bound is not None:
            isp_l = (lamb < lower_bound).sum() / len(lamb) * 100
        if upper_bound is not None:
            # Perform actions for when t_upper is provided
            isp_u = (lamb > upper_bound).sum() / len(lamb) * 100
        isp = isp_l + isp_u
    
    else:
        ir = imbalance_ratio(lamb)
        isp = (ir > ir_bound).sum() / len(ir) *100

    return isp


def mean_imbalance_ratio(
    data: pd.Series | pd.DataFrame,
    relevance_pdf: Callable  | list[Callable] | None = None,
    discrete: bool | list[bool] = False,
    drop_inf_nan: bool = True,
    kde_bandwidth: str | float = 'silverman',
    kde_kernel: str = 'gaussian',
    kde_grid_points: int = 2**14) -> float:
    '''
    Computes the mean imbalance ratio (mIR) for given data using empirical and relevance probabilities (According to Equation (5) in the article).
    Args:
    - data (pd.Series | pd.DataFrame): Input data for which to compute the probability ratio.
    - relevance_pdf (Callable | list[Callable] | None, optional): Probability density function(s) for relevance probability.
      - If a Series is passed to `data`, `relevance_pdf` should be a single callable function.
      - If a DataFrame is passed to `data`, `relevance_pdf` should be a list of callable functions with one function per column in data.
      - It is assumed that each callable function returns a value of type float.
      - If None, default functions will be used for relevance probability, which assumes uniform distribution of the relevance.
    - discrete (bool | list[bool], optional): Indicates if the data is discrete.
      - If a Series is passed to `data`, `discrete` should be a single boolean value.
      - If a DataFrame is passed to `data`, `discrete` should be a list of boolean values with one value per column.
    - drop_inf_nan (bool, optional): Whether to drop infinite and NaN values from `data` before computation.
      Default is True.
    - kde_bandwidth (str | float): Bandwidth for KDE. Default is 'silverman'.
    - kde_kernel (str): Kernel type for KDE. Default is 'gaussian'.
    - kde_grid_points (int): Number of points in the KDE evaluation grid. Default is 2^14.
    
    Returns:
    - (float): The mean imbalance ratio mIR) for the input data.
    '''
    if not isinstance(data, (pd.Series, pd.DataFrame)):
        raise TypeError("The argument 'data' is not of type pd.Series or pd.DataFame.")
    lamb = probability_ratio(data, relevance_pdf, discrete, drop_inf_nan,
                            kde_bandwidth, kde_kernel, kde_grid_points)
    ir = imbalance_ratio(lamb)
    mir = float(np.mean(ir))
    return mir








