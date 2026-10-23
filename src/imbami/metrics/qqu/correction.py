"""
Sample-level reference corrections c_Delta of the QQU framework.

Both variants are zero at r = 1 but positive in expectation under correct
tracking: the realized ratio r^R_i = e_i^2 / sigma_i^2 fluctuates by the
irreducible sampling variability of a single residual, the expected ratio
r^E_i = E_hat[e^2 | sigma_i] / sigma_i^2 by the finite-sample estimation of the
conditional mean. Subtracting the reference value

    c_Delta,i = E_ref[Delta(r_i)]

makes each variant interpretable on its own: Delta(r_i) - c_Delta,i is positive
when the observed discrepancy exceeds the level expected under correct tracking
and negative when it falls short.

Realized QQU
    Under the framework's assumption that the standardized residual is Gaussian,
    z = e / sigma ~ N(0, 1), we have r^R_i = z_i^2 ~ chi2_1 for every sample, so
    the reference value is a constant -- the same for every sample, model and
    dataset (see ``realized_c_delta``).
Expected QQU
    The reference law of r^E_i is that of the kernel average of the per-sample
    reference statistics, whose mean and variance follow from the geometry of the
    kernel (see ``expected.kernel_conditional_mean``), so c^E_Delta,i varies
    across samples (see ``expected_c_delta``). The kernel makes the correction
    data-dependent, the distribution it is compared against is the theoretical
    one.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Callable, Literal

import numpy as np
from scipy import integrate, optimize, stats

from .discrepancy_functions import (
    Discrepancy,
    log_discrepancy,
    ratio_discrepancy,
    resolve_discrepancy,
    squared_log_discrepancy,
    squared_ratio_discrepancy,
)
from .expected import kernel_conditional_mean
from .residual_statistics import (
    ABSOLUTE,
    ABSOLUTE_PLAIN,
    SQUARED,
    ResidualStatistic,
    resolve_statistic,
)
from .utils import as_1d_float, check_choice, round_significant

Mode = Literal["realized", "expected"]
Aggregator = Literal["mean", "median"]
Estimator = Literal["kernel", "ENCE"]

_REFERENCE_GRID = 200_000    # quantile grid for the reference moments of custom statistics
_CHUNK_ELEMENTS = 2_000_000  # cap per (reference law x quantile grid) block

def _identity(r: np.ndarray) -> np.ndarray:
    """Default ratio transform: none."""
    return r

# --------------------------------------------------------------------------
# Realized QQU: a single constant
# --------------------------------------------------------------------------

@lru_cache(maxsize=None)
def realized_c_delta(
    statistic: str | ResidualStatistic | tuple = "squared",
    discrepancy: str | Discrepancy = "squared_ratio",
    aggregator: Aggregator = "mean",
    n_grid: int = 200_000,
) -> float:
    """
    Reference value c^R_Delta = E_ref[Delta(r^R)] of the realized QQU variant.

    With r^R_i ~ chi2_1 the reference value does not depend on the sample, so it is
    one constant that may be subtracted before or after aggregation. Reproduces
    1.758565 (mean) and 1.155668 (median) for (squared, log), 0.967876 for
    (squared, ratio), 2.0 for the mean of (squared, squared_ratio) and -1.270365
    for the mean of (squared, signed_log).

    Parameters
    ----------
    statistic, discrepancy
        Must match the ones used for the realized q.
    aggregator : {"mean", "median"}, default "mean"
        Which functional of the chi2_1 reference distribution is subtracted; must
        match the aggregator applied to the corrected scores downstream.
    n_grid : int, default 200_000
        Quantile grid of z ~ N(0, 1) used when the pair has no closed form; the
        (squared, log) pair is integrated / root-solved exactly instead.
    """
    statistic = resolve_statistic(statistic)
    discrepancy = resolve_discrepancy(discrepancy)
    check_choice(aggregator, ("mean", "median"), "aggregator")

    if statistic is SQUARED and discrepancy is squared_ratio_discrepancy and aggregator == "mean":
        return 2.0                                # E[(z^2-1)^2] = 3 - 2 + 1
    if statistic is SQUARED and discrepancy is log_discrepancy:
        if aggregator == "mean":
            integral, _ = integrate.quad(
                lambda x: abs(np.log(x)) * stats.chi2.pdf(x, df=1), 1e-10, np.inf, limit=200
            )
            return float(integral)
        # median of |log chi2_1|: solve P(|log chi2_1| <= t) = 1/2
        cdf = stats.chi2.cdf
        root = optimize.root_scalar(
            lambda t: cdf(np.exp(t), df=1.0) - cdf(np.exp(-t), df=1.0) - 0.5,
            bracket=(1e-6, 50.0),
            method="brentq",
        )
        return float(root.root)
    z = stats.norm.ppf((np.arange(n_grid) + 0.5) / n_grid)
    penalty = discrepancy(statistic.ratio(z, np.ones_like(z)))
    return float(penalty.mean() if aggregator == "mean" else np.median(penalty))

# --------------------------------------------------------------------------
# Expected QQU: one value per sample
# --------------------------------------------------------------------------

@lru_cache(maxsize=None)
def _reference_moments(statistic: ResidualStatistic) -> tuple[float, float]:
    """
    Mean and variance of W = S_err(z) / S_unc(1) for z ~ N(0, 1).

    W is the per-sample reference statistic whose kernel average is r^E, so these
    moments anchor the reference law. Exact for the registered statistics -- (1, 2)
    for the squared one, since W = z^2 -- and evaluated on a quantile grid for
    custom statistic pairs.
    """
    if statistic is SQUARED:
        return 1.0, 2.0
    if statistic is ABSOLUTE:
        return 1.0, np.pi / 2.0 - 1.0
    if statistic is ABSOLUTE_PLAIN:
        return float(np.sqrt(2.0 / np.pi)), 1.0 - 2.0 / np.pi
    z = stats.norm.ppf((np.arange(_REFERENCE_GRID) + 0.5) / _REFERENCE_GRID)
    w = statistic.s_err(z) / statistic.s_unc(np.ones_like(z))
    return float(w.mean()), float(w.var())

def expected_c_delta(
    discrepancy: str | Discrepancy,
    mean_factor: np.ndarray,
    var_factor: np.ndarray,
    *,
    statistic: str | ResidualStatistic | tuple = "squared",
    aggregator: Aggregator = "mean",
    ratio_transform: Callable[[np.ndarray], np.ndarray] | None = None,
    n_grid: int = 4001,
) -> np.ndarray:
    """
    Sample-level reference value c^E_Delta,i = E_ref[Delta(r^E_i)] of expected QQU.

    The reference law of r^E_i is a Gamma law matched to the first two moments of
    the kernel average,

        mean_i = mu_ref * mean_factor_i,    var_i = Var_ref * var_factor_i,

    with mu_ref and Var_ref the moments of the per-sample reference statistic
    W = S_err(e) / S_unc(sigma) under the Gaussian assumption (1 and 2 for the
    squared statistic, where W = z^2 ~ chi2_1) and mean_factor_i, var_factor_i the
    kernel geometry of the estimator used for q^E. Matching those moments is exact
    in the first two moments of the reference law, is exact in distribution when
    the kernel weights are equal and sigma is locally constant (where the law
    becomes chi2_{n_eff}/n_eff), and reduces to the realized reference chi2_1 when
    a single sample contributes.

    Parameters
    ----------
    discrepancy : str | callable
        Must match the discrepancy used for q^E.
    mean_factor, var_factor : array-like, shape (N,)
        Reference geometry of the estimator, elements 2 and 3 of
        ``expected.kernel_conditional_mean`` (or ``QQU.geometry``). They are
        factors, not moments: the law they describe is
        (mu_ref * mean_factor, Var_ref * var_factor).
    statistic : str | ResidualStatistic | (s_err, s_unc), default "squared"
        Supplies mu_ref and Var_ref.
    aggregator : {"mean", "median"}, default "mean"
        Which functional of the reference law is subtracted; must match the
        aggregator applied to the corrected scores downstream. "mean" is the
        article's c^E_Delta,i, "median" is the analogue of the median reference
        constant of the realized variant.
    ratio_transform : callable, optional
        Transform of the reference ratio applied before Delta, e.g. np.sqrt for the
        ENCE-style estimator, whose ratio is sqrt(E_hat[s_err] / s_unc) rather than
        E_hat[s_err] / s_unc.
    n_grid : int, default 4001
        Midpoint quantile grid of the reference law. The midpoint rule omits the
        mass beyond its extreme nodes and therefore understates fast-growing
        discrepancies by ~0.3% for the smallest pools and ~0.05% for n_eff >= 20,
        falling roughly as 1/n_grid and about ten times less for slowly growing
        ones (log, ratio). Reference laws are deduplicated to six significant
        digits, so the cost is per distinct law (~3 ms each at n_grid = 4001).

    Returns
    -------
    np.ndarray, shape (N,)
        c^E_Delta,i, to be subtracted per sample before any aggregation.
    """
    statistic = resolve_statistic(statistic)
    discrepancy = resolve_discrepancy(discrepancy)
    aggregator = check_choice(aggregator, ("mean", "median"), "aggregator")
    mu_ref, var_ref = _reference_moments(statistic)

    mean_factor = as_1d_float(mean_factor, "mean_factor", positive=True)
    var_factor = as_1d_float(var_factor, "var_factor", mean_factor.size, positive=True)
    ref_mean = mean_factor * mu_ref
    ref_var = var_factor * var_ref

    if n_grid < 2:
        raise ValueError(f"n_grid must be >= 2, got {n_grid}.")
    transform = _identity if ratio_transform is None else ratio_transform

    # One Gamma(shape = mean^2/var, scale = var/mean) per distinct (mean, var).
    uniq, inverse = np.unique(
        np.stack([round_significant(ref_mean), round_significant(ref_var)], axis=1),
        axis=0, return_inverse=True,
    )
    quantiles = (np.arange(n_grid) + 0.5) / n_grid
    chunk = max(1, _CHUNK_ELEMENTS // n_grid)
    c_uniq = np.empty(uniq.shape[0], dtype=float)
    for start in range(0, uniq.shape[0], chunk):
        block = uniq[start:start + chunk]
        shape = (block[:, 0] ** 2 / block[:, 1])[:, None]
        scale = (block[:, 1] / block[:, 0])[:, None]
        ratios = stats.gamma.ppf(np.tile(quantiles, (block.shape[0], 1)), a=shape, scale=scale)
        penalty = discrepancy(transform(ratios))
        c_uniq[start:start + chunk] = (
            penalty.mean(axis=1) if aggregator == "mean" else np.median(penalty, axis=1)
        )
    return c_uniq[inverse]

# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------

def compute_c_delta(
    y: np.ndarray,
    mu: np.ndarray,
    sigma: np.ndarray,
    *,
    statistic: str | ResidualStatistic | tuple = "squared",
    discrepancy: str | Discrepancy = "squared_ratio",
    mode: str = "realized",
    aggregator: Aggregator = "mean",
    estimator: str = "kernel",
    estimator_kwargs: dict | None = None,
    geometry: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
    n_grid: int = 4001,
) -> np.ndarray:
    """
    Sample-level reference correction c_Delta,i for realized or expected QQU.

    Parameters
    ----------
    y, mu, sigma : array-like, shape (N,)
        Ground truth, predicted mean and predicted std of ONE model on ONE
        dataset. Must be finite and of equal length, sigma strictly positive.
    statistic, discrepancy
        Must match the ones used for the QQU scores.
    mode : {"realized", "expected"}, default "realized"
        Determines the reference law: the constant chi2_1 reference for the
        realized variant, the geometry-dependent one for the expected variant.
    aggregator : {"mean", "median"}, default "mean"
        Must match the aggregator applied to the corrected scores downstream.
    estimator : {"kernel", "ENCE"}, default "kernel"
        Estimator used for q^E; fixes the ratio transform of the expected
        reference (np.sqrt for "ENCE") and, together with estimator_kwargs, the
        geometry.
    estimator_kwargs : dict, optional
        Keyword arguments of that estimator, e.g. {"bandwidth_factor": 2.0}. Must
        match the ones used for q^E.
    geometry : tuple, optional
        (n_eff, mean_factor, var_factor) of the estimator, e.g. ``QQU.geometry``.
        Computed from sigma and the estimator configuration if omitted; passing it
        avoids a second traversal of the kernel. Only the last two elements enter
        the reference law, the first is validated and otherwise ignored.
    n_grid : int, default 4001
        Quantile grid of the expected reference law, see ``expected_c_delta``.

    Returns
    -------
    np.ndarray, shape (N,)
        c_Delta,i: constant for the realized variant, geometry-dependent for the
        expected one. Aggregate it with ``aggregation.aggregate_c_delta``.
    """
    mode = check_choice(mode, ("realized", "expected"), "mode")
    aggregator = check_choice(aggregator, ("mean", "median"), "aggregator")
    estimator = check_choice(estimator, ("kernel", "ENCE"), "estimator")
    statistic = resolve_statistic(statistic)
    discrepancy = resolve_discrepancy(discrepancy)

    y = as_1d_float(y, "y")
    n = y.size
    if n < 2:
        raise ValueError("At least two observations are required.")
    mu = as_1d_float(mu, "mu", n)
    sigma = as_1d_float(sigma, "sigma", n, positive=True)

    if mode == "realized":
        return np.full(n, realized_c_delta(statistic, discrepancy, aggregator))

    if geometry is None:
        kernel = kernel_conditional_mean(
            sigma, scale=statistic.s_unc(sigma), **(estimator_kwargs or {})
        )
        geometry = (kernel[1], kernel[2], kernel[3])
    _ = as_1d_float(geometry[0], "geometry[0] (n_eff)", n)   # validated, not used below
    mean_factor = as_1d_float(geometry[1], "geometry[1] (mean_factor)", n)
    var_factor = as_1d_float(geometry[2], "geometry[2] (var_factor)", n)

    return expected_c_delta(
        discrepancy, mean_factor, var_factor,
        statistic=statistic, aggregator=aggregator,
        ratio_transform=np.sqrt if estimator == "ENCE" else None,
        n_grid=n_grid,
    )