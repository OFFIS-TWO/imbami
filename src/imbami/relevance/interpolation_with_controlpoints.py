import numpy as np
import pandas as pd
from smogn.phi_ctrl_pts import phi_ctrl_pts
from smogn.phi import phi
from .base_relevance_function import RelevanceFunctionBase

class InterpolationWithControlPoints(RelevanceFunctionBase):
    def __init__(self) -> None:
        super().__init__()
        self.type = "InterpolationWithControlPoints"

    def fit(self, data: np.ndarray, method = 'auto', extrems = 'both', coef = 1.5, control_points = None) -> None:

        self.data = data
        self.control_params = phi_ctrl_pts(
                        y = data,
                        method = method,
                        xtrm_type = extrems,
                        coef = coef,
                        ctrl_pts = control_points)
        self.is_fitted = True


    def eval(self, y: pd.Series | np.ndarray) -> np.ndarray:
        if not self.is_fitted:
            raise RuntimeError("eval() can not be called before fit().")
        if isinstance(y, np.ndarray): # this is stupid, but phi requires pd.Series. The function would work with an array, but is badly coded.
            y = pd.Series(y)
        weights = phi(y = y, ctrl_pts = self.control_params)
        weights = np.array(weights)
        return weights
