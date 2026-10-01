from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, cast

import numpy as np
import pandas as pd
from scipy import stats, optimize, integrate


@dataclass
class CDeltaResult:
    aggregator: Literal["mean", "median"]   # "mean" or "median"
    n_samples: int
    alpha: float

    # Shape diagnostic only; not used to determine the correction
    ks_statistic: float                     # tests the standardized squared z-score against chi2_1
    shape_diagnostic_pvalue: float          # caveat: not a strictly valid p-value, see docstring
    shape_looks_gaussian: bool              # ks_pvalue >= alpha

    # Both constants always computed; pick whichever fits your reporting needs.
    c_delta_theoretical: float              # fixed chi2_1 reference (1.76 mean / 1.156 median)
    c_delta: float                          # derived from this model's own residual distribution


@lru_cache
def _theoretical_mean_abs_log_chi2_1() -> float:
    f = lambda z: np.abs(np.log(z)) * stats.chi2.pdf(z, df=1)
    val, _ = integrate.quad(f, 1e-10, np.inf, limit=200)
    return val


@lru_cache
def _theoretical_median_abs_log_chi2_1() -> float:
    F = stats.chi2.cdf
    g = lambda t: (F(np.exp(t), df=1) - F(np.exp(-t), df=1)) - 0.5
    return cast(float, optimize.brentq(g, 1e-6, 50, full_output=False))


def compute_c_delta(
    y: pd.Series | np.ndarray,
    mu: pd.Series | np.ndarray,
    sigma: pd.Series | np.ndarray,
    aggregator: Literal["mean", "median"] = "median",
    alpha: float = 0.05,
    eps: float = 1e-8,
) -> CDeltaResult:
    """
    Compute the theoretical and empirical correction constants c_Delta for
    realized QQU (q_R).

    The empirical correction constant is computed from the observed
    standardized residual distribution after rescaling it to unit empirical
    variance. This removes the effect of global scale mismatch between the
    residuals and predicted uncertainty, while retaining the shape of the
    standardized residual distribution.

    The corrected realized QQU, q_R - c_Delta, therefore measures the
    deviation of the observed QQU from the discrepancy expected under the
    empirical residual-shape reference.

    Two variants of c_Delta are returned:

    c_delta_theoretical : the fixed chi2_1 reference (~1.76 for the mean,
        ~1.156 for the median). The same number for every model and
        dataset, so corrected q_R values stay directly comparable across
        models. Assumes the z-score (mu - y)/sigma is standard Gaussian.

    c_delta : empirically derived from this model's own standardized residual distribution,
        after removing its scale (rescaled to unit variance) so that any
        real over/underconfidence is not absorbed into the constant. As a
        result it removes the residual scale and therefore makes the constant 
        insensitive to multiplicative scale mismatch; differences from the 
        theoretical constant then reflect differences in the standardized 
        residual distribution's shape. Use this when you want a
        shape-adjusted correction for a single model, or use the gap
        between the two constants as a diagnostic of how much non-Gaussian
        shape (rather than genuine mistracking) contributes to q_R.

    A KS test (`ks_statistic`, `ks_pvalue`) checks whether the z-score
    distribution's shape resembles Gaussian; treat the p-value as
    informative rather than a strict test, since the scale used to
    standardize is estimated from the same sample being tested.

    Parameters
    ----------
    y, mu, sigma : array-like, shape (N,)
        Ground truth, predicted mean, predicted std for ONE model on ONE
        dataset. Compute separately per model/dataset — do not pool, since
        each may have a different residual shape and therefore a different
        c_Delta.
    aggregator : {"mean", "median"}, default "median"
        Must match whatever aggregator is applied to real q_R downstream:
        the mean and median of |log Z| are different reference numbers
        (~1.76 vs ~1.156), and mixing them breaks the zero-centering.
    alpha : float, default 0.05
        Significance threshold for the shape diagnostic.
    eps : float, default 1e-8
        Numerical floor to avoid log(0) / division by zero.

    Returns
    -------
    CDeltaResult
        c_delta_theoretical, c_delta,
        and the shape diagnostic fields described above.
    """
    if aggregator not in ("mean", "median"):
        raise ValueError(f"aggregator must be 'mean' or 'median', got {aggregator!r}")

    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    if not (y.shape == mu.shape == sigma.shape):
        raise ValueError("y, mu, sigma must have the same shape.")
    if np.any(sigma <= 0):
        raise ValueError("sigma must be strictly positive.")

    residual = mu - y
    z_score = residual / sigma
    z_score = z_score[np.isfinite(z_score)]

    n = z_score.size
    if n < 2:
        raise ValueError("At least two finite observations are required.")

    scale = z_score.std(ddof=1)
    if not np.isfinite(scale) or scale <= eps:
        raise ValueError("Standardized residuals have zero or invalid variance.")
    z_score_scaled = z_score / scale   # isolate shape from scale mismatch
    z = np.maximum(z_score_scaled ** 2, eps)

    # Diagnostic shape test on the standardized squared z-score (see docstring)
    ks_stat, shape_diagnostic_pvalue = stats.kstest(z, stats.chi2(df=1).cdf)
    shape_looks_gaussian = shape_diagnostic_pvalue >= alpha

    c_delta_theoretical = (
        _theoretical_mean_abs_log_chi2_1() if aggregator == "mean"
        else _theoretical_median_abs_log_chi2_1()
    )

    q_empirical = np.abs(np.log(z))
    c_delta_empirical = (
        float(q_empirical.mean()) if aggregator == "mean"
        else float(np.median(q_empirical))
    )


    return CDeltaResult(
        aggregator=aggregator,
        n_samples=int(n),
        alpha=alpha,
        ks_statistic=float(ks_stat),
        shape_diagnostic_pvalue=float(shape_diagnostic_pvalue),
        shape_looks_gaussian=bool(shape_looks_gaussian),
        c_delta_theoretical=float(c_delta_theoretical),
        c_delta=float(c_delta_empirical),
    )