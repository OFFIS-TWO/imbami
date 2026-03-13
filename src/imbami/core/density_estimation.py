"""
Core density estimation functionality used throughout the package.
This module contains the fundamental density estimation functions that are
used by other modules but not directly exposed to users.
"""
from .helpers import Uniform_kernel
import numpy as np
import KDEpy
from KDEpy.bw_selection import improved_sheather_jones
import logging



def get_kernel(data: np.ndarray,
               bw_type: str | float,
               bw_factor: float | None = 1.0,
               kernel_type: str | None = 'gaussian') -> KDEpy.FFTKDE | Uniform_kernel:
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
        kernel =  Uniform_kernel(data=data)
    else:
        if kernel_type is None:
            raise ValueError("kernel_type is None. It must be provided if bw_type is not 'uniform'.")
        if isinstance(bw_type, (int, float)):
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
        if isinstance(bw_factor, float):   
            bandwidth = bandwidth * bw_factor
        

        kernel = KDEpy.FFTKDE(bw = bandwidth, kernel = kernel_type).fit(data) # type: ignore
    return kernel