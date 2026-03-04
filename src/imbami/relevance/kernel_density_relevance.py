import logging
import numpy as np
from ..core.density_estimation import get_kernel, bisect_kde_score
from .base_relevance_function import RelevanceFunctionBase

class KernelDensityRelevance(RelevanceFunctionBase):
    def __init__(self) -> None:
        super().__init__()
        self.usable_bandwidth_type = ['silverman', 'ISJ']
        self._bisect_kde_score = bisect_kde_score
        self.get_kernel = get_kernel
        self.type = "KernelDensityRelevance"

    def fit(self,
            data: np.ndarray,
            bandwidth_type: str = 'silverman',
            bandwidth_factor: float = 1,
            kernel_type : str = 'gaussian',
            grid_points = 2**12) -> None:
        if bandwidth_type not in self.usable_bandwidth_type:
            logging.error(f'{bandwidth_type=} is not an acceptable value: {self.usable_bandwidth_type}')
        
        # Empirical Data
        self.data = data
        self.bandwidth_type = bandwidth_type
        self.kernel = self.get_kernel(data = self.data,
                            bw_type= self.bandwidth_type,
                            bw_factor= bandwidth_factor,
                            kernel_type= kernel_type)
        self.grid_x, self.grid_y = self.kernel.evaluate(grid_points)
        self.max_density = max(self.grid_y)
        self.is_fitted = True

    def eval(self, y: np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("eval() can not be called before fit().")
        density = np.array([bisect_kde_score(i, self.grid_x, self.grid_y) for i in y])
        density = density / self.max_density
        weights = 1 - density
        return weights
    

