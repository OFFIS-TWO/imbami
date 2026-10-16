
import numpy as np



def _silverman_bandwidth(sigma: np.ndarray) -> float:
    """Silverman's rule of thumb for the kernel bandwidth on sigma."""
    n = sigma.size
    if n < 2:
        return 1e-8
    std = np.std(sigma, ddof=1)
    iqr = np.subtract(*np.percentile(sigma, [75, 25]))
    scale = min(std, iqr / 1.34)
    bw = 0.9 * scale * n ** (-1 / 5)
    # Fallback if all sigma values are (nearly) identical
    return bw if bw > 0 else 1e-8


def _check_inputs(sigma, values):
    sigma = np.asarray(sigma, dtype=float)
    values = np.asarray(values, dtype=float)
    if sigma.ndim != 1 or values.ndim != 1:
        raise ValueError("sigma and values must be 1D arrays.")
    if sigma.shape != values.shape:
        raise ValueError("sigma and values must have the same length.")
    return sigma, values