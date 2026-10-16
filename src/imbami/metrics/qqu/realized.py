from dataclasses import dataclass
from functools import lru_cache
from typing import Literal, Union, cast

import numpy as np
import pandas as pd
from scipy import stats, optimize, integrate

from .discrepancy_functions import Discrepancy, log_discrepancy, resolve_discrepancy
from .residual_statistics import SQUARED, ResidualStatistic, resolve_statistic


@dataclass
class CDeltaResult:
    aggregator: Literal["mean", "median"]
    n_samples: int
    alpha: float

    # Shape diagnostic only; not used to determine the correction
    ks_statistic: float                     # tests the standardized squared z-score against chi2_1
    shape_diagnostic_pvalue: float          # caveat: not a strictly valid p-value, see docstring
    shape_looks_gaussian: bool              # ks_pvalue >= alpha

    # Both constants always computed; pick whichever fits your reporting needs.
    c_delta_theoretical: float              # fixed Gaussian reference
    c_delta: float                          # derived from this model's own residual distribution


# --------------------------------------------------------------------------
# Exact values for statistic="squared", discrepancy="log" (|log chi2_1|)
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# General theoretical correction constant
# --------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _reference_c_delta(
    statistic: ResidualStatistic,
    discrepancy: Discrepancy,
    aggregator: str,
    n_grid: int,
) -> float:
    # Exact closed-form-ish path for the most common case.
    if statistic is SQUARED and discrepancy is log_discrepancy:
        return (
            _theoretical_mean_abs_log_chi2_1() if aggregator == "mean"
            else _theoretical_median_abs_log_chi2_1()
        )

    # Deterministic quantile grid of z ~ N(0, 1): no Monte Carlo noise.
    # By scale invariance (R1) we may set sigma = 1.
    z = stats.norm.ppf((np.arange(n_grid) + 0.5) / n_grid)
    r = statistic.ratio(z, np.ones_like(z))
    q = discrepancy(r)
    return float(q.mean() if aggregator == "mean" else np.median(q))


def reference_c_delta(
    statistic: Union[str, ResidualStatistic, tuple] = "squared",
    discrepancy: Union[str, Discrepancy] = "squared_ratio",
    aggregator: Literal["mean", "median"] = "mean",
    n_grid: int = 200_000,
) -> float:
    """
    Theoretical c_Delta: the value the chosen aggregate of Delta(r^R) takes
    when the standardized residuals are standard Gaussian, e/sigma ~ N(0, 1).

    Works for any statistic/discrepancy pair by evaluating Delta on a fine,
    deterministic quantile grid of N(0, 1). Reproduces ~1.76 (mean) and
    ~1.156 (median) for (squared, log) and 2 (mean) for (squared, squared_ratio).

    Assumes S_err(sigma*z) / S_unc(sigma) does not depend on sigma
    (guaranteed for scale-invariant pairs, requirement R1).
    """
    if aggregator not in ("mean", "median"):
        raise ValueError(f"aggregator must be 'mean' or 'median', got {aggregator!r}")
    return _reference_c_delta(
        resolve_statistic(statistic), resolve_discrepancy(discrepancy), aggregator, n_grid
    )


# --------------------------------------------------------------------------
# Theoretical + empirical correction constant with diagnostics
# --------------------------------------------------------------------------

def compute_c_delta(
    y: pd.Series | np.ndarray,
    mu: pd.Series | np.ndarray,
    sigma: pd.Series | np.ndarray,
    statistic: Union[str, ResidualStatistic, tuple] = "squared",
    discrepancy: Union[str, Discrepancy] = "squared_ratio",
    aggregator: Literal["mean", "median"] = "mean",
    alpha: float = 0.05,
    eps: float = 1e-12,
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

    c_delta_theoretical : the fixed Gaussian reference (e.g. ~1.76 for the
        mean and ~1.156 for the median of |log r|, 2 for the mean of
        (r-1)^2). The same number for every model and dataset, so corrected
        q_R values stay directly comparable across models. Assumes the
        z-score (y - mu)/sigma is standard Gaussian.

    c_delta : empirically derived from this model's own standardized residual
        distribution, after removing its scale (rescaled to unit variance) so
        that any real over/underconfidence is not absorbed into the constant.
        Differences from the theoretical constant then reflect differences in
        the standardized residual distribution's shape. Use this when you want
        a shape-adjusted correction for a single model, or use the gap between
        the two constants as a diagnostic of how much non-Gaussian shape
        (rather than genuine mistracking) contributes to q_R.

    A KS test of z^2 against chi2_1 checks whether the z-score distribution's
    shape resembles Gaussian (independent of the chosen statistic); treat the
    p-value as informative rather than a strict test, since the scale used to
    standardize is estimated from the same sample being tested.

    Parameters
    ----------
    y, mu, sigma : array-like, shape (N,)
        Ground truth, predicted mean, predicted std for ONE model on ONE
        dataset. Compute separately per model/dataset -- do not pool.
    statistic, discrepancy
        Must match what is used for the real q_R (see QQU).
    aggregator : {"mean", "median"}, default "median"
        Must match whatever aggregator is applied to real q_R downstream.
    alpha : float, default 0.05
        Significance threshold for the shape diagnostic.
    eps : float, default 1e-12
        Numerical floor for the variance check.
    """
    if aggregator not in ("mean", "median"):
        raise ValueError(f"aggregator must be 'mean' or 'median', got {aggregator!r}")
    statistic = resolve_statistic(statistic)
    discrepancy = resolve_discrepancy(discrepancy)

    y = np.asarray(y, dtype=float)
    mu = np.asarray(mu, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    if not (y.shape == mu.shape == sigma.shape):
        raise ValueError("y, mu, sigma must have the same shape.")
    if np.any(sigma <= 0):
        raise ValueError("sigma must be strictly positive.")

    z_score = (y - mu) / sigma
    z_score = z_score[np.isfinite(z_score)]

    n = z_score.size
    if n < 2:
        raise ValueError("At least two finite observations are required.")

    scale = z_score.std(ddof=0)
    if not np.isfinite(scale) or scale <= eps:
        raise ValueError("Standardized residuals have zero or invalid variance.")
    z_scaled = z_score / scale   # isolate shape from scale mismatch

    # Diagnostic shape test on the squared, rescaled z-score (see docstring)
    ks_stat, shape_diagnostic_pvalue = stats.kstest(z_scaled ** 2, stats.chi2(df=1).cdf)

    # Empirical reference: same pipeline as for real data, with sigma = 1.
    q_empirical = discrepancy(statistic.ratio(z_scaled, np.ones_like(z_scaled)))
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
        shape_looks_gaussian=bool(shape_diagnostic_pvalue >= alpha),
        c_delta_theoretical=reference_c_delta(statistic, discrepancy, aggregator),
        c_delta=c_delta_empirical,
    )