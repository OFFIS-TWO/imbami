"""
Core density estimation functionality used throughout the package.
This module contains the fundamental density estimation functions that are
used by other modules but not directly exposed to users.
"""
from .helpers import bisect_kde_score, Uniform_kernel
import numpy as np
import KDEpy
import pandas as pd
from typing import Callable
import warnings
from KDEpy.bw_selection import improved_sheather_jones
import logging


def get_densities(data: pd.Series | pd.DataFrame,
                 relevance_pdf: Callable | list[Callable] | None = None,
                 discrete: bool | list[bool] = False,
                 kde_bandwidth: str | float = 'silverman',
                 kde_kernel: str = 'gaussian',
                 kde_grid_points: int = 2**14) -> tuple[pd.Series, pd.Series]:
    """
    Calculate empirical and relevance densities for given data.

    Parameters:
    - data (pd.Series | pd.DataFrame): Input data for which to calculate densities.
    - relevance_pdf (Callable | list[Callable] | None): Probability density function(s) for relevance.
      If None, uniform relevance is assumed.
    - discrete (bool | list[bool]): Whether the data is discrete.
    - kde_bandwidth (str | float): Bandwidth for KDE. Can be 'silverman' for Silverman's rule of thumb
      or a float value. Default is 'silverman'.
    - kde_kernel (str): Kernel type for KDE. Default is 'gaussian'.
    - kde_grid_points (int): Number of points in the KDE evaluation grid. Default is 2^14.

    Returns:
    - tuple[pd.Series, pd.Series]: Tuple containing (empirical_density, relevance_density)
    """
    if isinstance(data, pd.Series):
        # argument sanity checks
        if isinstance(discrete, list):
            if len(discrete) == 1:
                discrete = discrete[0]
            else:
                raise ValueError(f"Too many values are supplied in 'discrete'. Expected list of length 1, received length {len(discrete)}. {discrete=}")
        if isinstance(relevance_pdf, list):
            if len(relevance_pdf) == 1:
                relevance_pdf = relevance_pdf[0]
            else:
                raise ValueError(f"Too many density functions are supplied in 'relevance_pdf'. Expected list of length 1, received length {len(relevance_pdf)}. {relevance_pdf=}")
        prob_emp = density(data = data, 
                            discrete= discrete,
                            kde_bandwidth = kde_bandwidth,
                            bandwidth_factor= 1,
                            kde_kernel= kde_kernel, 
                            kde_grid_points=kde_grid_points)
        prob_rel = density(data = data, 
                            discrete= discrete,
                            pdf= relevance_pdf)
        if (prob_emp < 0).any():
            warnings.warn("Found negative values in the empirical probability density. This should not be. Results might be corrupted.")
        if (prob_rel < 0).any():
            warnings.warn("Found negative values in the relevance probability density. This should not be. Results might be corrupted.")
    
    elif isinstance(data, pd.DataFrame):
        if isinstance(discrete, list):
            if len(discrete) != len(data.columns):
                raise TypeError(f"'discrete' is a list of length {len(discrete)}, but the 'data' pd.DataFrame has {len(data.columns)} columns. " +
                                "They should be equal length")
            if not all(isinstance(item, bool) for item in discrete):
                raise ValueError(f"All values in 'discrete' should be booleans, but they are {discrete=}.")
        elif isinstance(discrete, bool):
            discrete = [discrete for i in data.columns]
        prob_emp = pd.DataFrame(data = None, index= data.index, columns= data.columns)
        prob_rel = pd.DataFrame(data = None, index= data.index, columns= data.columns)
        for i, col in enumerate(data.columns):
            col_data = data[col]
            prob_emp[col] = density(data = col_data, 
                            discrete= discrete[i],
                            kde_bandwidth = kde_bandwidth,
                            bandwidth_factor= 1,
                            kde_kernel= kde_kernel, 
                            kde_grid_points=kde_grid_points)
            if relevance_pdf is None:
                prob_rel[col] = density(data = col_data, 
                                    discrete= discrete[i],
                                    pdf= relevance_pdf)
            else:
                if isinstance(relevance_pdf, list):
                    prob_rel[col] = density(data = col_data, 
                                        pdf= relevance_pdf[i])
                else:
                    raise TypeError(f"'relevance_pdf' is not of type list or None. Should be if 'data' is a pd.DataFrame. Received {type(relevance_pdf).__name__}")
                
        # check for negative values
        for col in prob_emp.columns:
            if (prob_emp[col] < 0).any():
                warnings.warn(f"Found negative values in the empirical probability density of column {col}. This should not be. Results might be corrupted.")
        for col in prob_rel.columns:
            if (prob_rel[col] < 0).any():
                warnings.warn(f"Found negative values in the relevance probability density of column {col}. This should not be. Results might be corrupted.")         

        prob_emp = prob_emp.product(axis = 1)
        prob_rel = prob_rel.product(axis = 1)
    
    else:
        raise TypeError(f"Expected pd.Series or pd.DataFrame, got {type(data).__name__}")



    return prob_emp, prob_rel





def density(data: pd.Series,
                     discrete: bool = False,
                     kde_bandwidth: str | float = 'silverman',
                     bandwidth_factor : float = 1.0,
                     kde_kernel: str = 'gaussian',
                     kde_grid_points: int = 2**14,
                     pdf: None | Callable = None) -> pd.Series:
    """
    Calculate empirical probability density/mass for given data.

    For continuous data, uses KDE to estimate density. For discrete data,
    calculates the probability mass function.

    Parameters:
    - data (pd.Series): Input data series.
    - discrete (bool): Whether the data is discrete. Defaults to False.
    - kde_bandwidth (str | float): Bandwidth for KDE. Default is 'silverman'.
    - kde_kernel (str): Kernel type for KDE. Default is 'gaussian'.
    - kde_grid_points (int): Number of points in the KDE evaluation grid. Default is 2^14.

    Returns:
    - pd.Series: Empirical probability density/mass estimates for each point in the input data.
    """
    if pdf is not None:
        density = data.apply(lambda x: pdf(x))
    if not discrete:
        # continuous case
        kernel = get_kernel(data = data.to_numpy(),
                            bw_type= kde_bandwidth,
                            bw_factor= bandwidth_factor,
                            kernel_type= kde_kernel)
        grid_x, grid_y = kernel.evaluate(kde_grid_points)
        density = np.array([bisect_kde_score(i, grid_x, grid_y) for i in data])
        density = pd.Series(data = density, index = data.index)
    else:
        # calculates the probability using the PMF
        unique_values, counts = np.unique(data, return_counts=True)
        probabilities = counts / len(data)
        replacement_dict = dict(zip(unique_values, probabilities))
        density = np.vectorize(replacement_dict.get)(data)
        
    density = pd.Series(density, index= data.index, name= data.name)
    return density



def get_kernel(data: np.ndarray,
               bw_type: str | float,
               bw_factor: float = 1.0,
               kernel_type: str = 'gaussian') -> KDEpy.FFTKDE | Uniform_kernel:
    """
    Returns a Gaussian KDE kernel using specified bandwidth type or a uniform kernel for 'uniform'.

    Parameters:
        data (np.ndarray): Data array to fit the KDE.
        bw_type (str or float): 'ISJ' for Improved Sheather-Jones, 'silverman', 'uniform', or a float for fixed bandwidth.
        bw_factor (float): Factor to scale the bandwidth (default is 1.0; ignored if bw_type is 'uniform' or float).
        kernel_type (str): Kernel type to use for KDE (default is 'gaussian'). Must be one of KDEpy's supported kernels (ignored if bw_type is 'uniform').

    Returns:
        KDEpy.FFTKDE or Uniform_kernel: The fitted kernel density estimator or uniform kernel.
    """

    usable_bw_type = ['silverman', 'ISJ', 'uniform']    
    if bw_type not in usable_bw_type:
        raise ValueError(f"{bw_type=} is not an acceptable value: {usable_bw_type}")

    if bw_type == 'uniform':
        return Uniform_kernel(data=data)
    elif isinstance(bw_type, (int, float)):
        bandwidth = float(bw_type)
    else:
        match bw_type:
            case 'ISJ':
                try:
                    bandwidth = improved_sheather_jones(data.reshape(-1,1))
                except Exception as error:
                    logging.warning('Failed to compute ISJ Bandwidth. Fall back to Silverman bandwidth as default.\n' + f'Exception error: {error}')
                    bandwidth = (4*data.std(ddof=1)**5 / 3 / len(data))**(1/5)
            case 'silverman':
                bandwidth = (4*data.std(ddof=1)**5 / 3 / len(data))**(1/5)
            case _:
                raise ValueError(f"Unsupported bw_type: {bw_type}")
        
    bandwidth = bandwidth * bw_factor

    kernel = KDEpy.FFTKDE(bw = bandwidth, kernel = kernel_type).fit(data) # type: ignore
    return kernel