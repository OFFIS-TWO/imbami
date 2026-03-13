from .base_relevance_function import RelevanceFunctionBase
import numpy as np
from denseweight import DenseWeight as DenseWeight_og
from typing import Union


class DenseWeight(RelevanceFunctionBase):
    """
    Wrapper for DenseWeight_og to expose a `fit(data=...)` interface.

    This class provides a model-agnostic interface for density-based weighting
    functions. It wraps the original `DenseWeight_og` class and allows using
    `data` instead of `y` in the `fit` method, making it compatible with
    factory functions or standardized pipelines.

    Parameters
    ----------
    alpha : float
        Smoothing factor for the density weighting.
    bandwidth : float
        Bandwidth parameter for the density estimation.
    eps : float
        Small value to avoid division by zero or numerical instability.

    Attributes
    ----------
    type : str
        Indicates the type of the relevance function ('DenseWeight').
    relevance_function : DenseWeight_og
        Internal instance of the original DenseWeight implementation.
    is_fitted : bool
        Flag indicating whether the relevance function has been fitted.

    Methods
    -------
    fit(data, grid_points=4096)
        Fit the density-based relevance function to the provided data.
    eval(y)
        Evaluate the relevance values for the given data points.
    """
    def __init__(self, alpha: float = 1, bandwidth: Union[float, str, None] = None, eps: float = 1e-6) -> None:
        super().__init__()
        self.type = "DenseWeight"
        self.relevance_function = DenseWeight_og(alpha=alpha, bandwidth=bandwidth, eps=eps)

    def fit(self, data: np.ndarray, grid_points: int = 4096):
        self.relevance_function.fit(y=data, grid_points=grid_points)
        self.is_fitted = True

    def eval(self, y: np.ndarray) -> np.ndarray:
        return self.relevance_function.eval(y=y)
