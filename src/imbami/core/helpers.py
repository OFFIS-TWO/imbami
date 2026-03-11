import numpy as np
import bisect


def bisect_kde_score(y, grid_x, grid_y) -> np.ndarray:
    idx = bisect.bisect(grid_x, y)
    try:
        dens = grid_y[idx]
    except IndexError:
        if idx <= -1:
            idx = 0
        elif idx >= len(grid_x):
            idx = len(grid_x) - 1
        dens = grid_y[idx]
    return dens


class Uniform_kernel:
    """
    Simple uniform kernel for relevance estimation. 
    Returns constant density within the range of the training data.
    """
    def __init__(self, data: np.ndarray):
        self.data_min = data.min()
        self.data_max = data.max()
        self.rel = 1 / (max(data) - min(data))

    def __call__(self, y: np.ndarray) -> np.ndarray:
        return np.full(shape = y.shape, fill_value = self.rel)
    
    def evaluate(self, grid_points: int) -> tuple[np.ndarray, np.ndarray]:
        """
        Evaluate the uniform kernel on a grid.
        
        Parameters:
            grid_points (int): Number of points in the evaluation grid.
            
        Returns:
            tuple[np.ndarray, np.ndarray]: Grid points and corresponding density values.
        """
        grid_x = np.linspace(start=self.data_min, stop=self.data_max, num=grid_points)
        grid_y = self.__call__(grid_x)
        return grid_x, grid_y