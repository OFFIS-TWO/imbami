"""
User-facing QQU object.

The user supplies y, mu, sigma once, plus the choice of residual statistic,
discrepancy, mode and estimator. The per-sample contributions q_i and the
reference geometry of the estimator are computed lazily and cached; all
aggregations reuse them.

    qqu = QQU(y, mu, sigma, mode="expected")
    qqu.avg(corrected=True)          # QQU_avg,corr
    qqu.c_delta()                    # dataset-wide correction c_Delta
    qqu.c_delta_i()                  # per-sample correction c_Delta,i
    t, avg, n = qqu.avg_profile()    # regional profile
"""
from functools import cached_property
from typing import Literal, Union

import numpy as np

from .aggregation import (
    QQU_avg,
    QQU_avg_profile,
    QQU_median,
    QQU_sum_profile,
    aggregate_c_delta,
)
from .correction import Aggregator, compute_c_delta
from .discrepancy_functions import Discrepancy, resolve_discrepancy
from .expected import kernel_conditional_mean
from .residual_statistics import ResidualStatistic, resolve_statistic
from .utils import as_1d_float, check_choice

class QQU:
    """
    Quality of Uncertainty (QQU) evaluation of a probabilistic regression model.

    Parameters
    ----------
    y, mu, sigma : array-like, shape (N,)
        Ground truth, predicted mean and predicted standard deviation. Must be
        finite and of equal length; sigma strictly positive.
    statistic : str | ResidualStatistic | (s_err, s_unc), default "squared"
        Residual statistic. Registered: "squared" (e^2 vs sigma^2), "absolute"
        (|e| vs sqrt(2/pi)*sigma, assumes Gaussian residuals), "absolute_plain"
        (|e| vs sigma). Custom: a tuple of two callables (s_err(e), s_unc(sigma)).
    discrepancy : str | callable, default "squared_ratio"
        Delta(r). Registered: "ratio", "log", "squared_ratio", "squared_log",
        "signed_log". Custom: any callable on the ratio array r, e.g.
        ``make_asymmetric_squared(0.5)``.
    mode : {"realized", "expected"}, default "realized"
        "realized": r_i = S_err(e_i) / S_unc(sigma_i)
        "expected": r_i = E[S_err(e) | sigma_i] / S_unc(sigma_i), estimated from
        neighbouring samples in sigma.
    estimator : {"kernel", "ENCE"}, default "kernel"
        Conditional-mean estimator, expected mode only. "ENCE" uses the same
        kernel but square-roots numerator and denominator, aligning r with the
        RMSE/RMV ratio of ENCE.
    estimator_kwargs : dict, optional
        Passed to the estimator, e.g. {"bandwidth_factor": 2.0} or
        {"kernel_type": "uniform"} (see expected.py).
    w : array-like, shape (N,), optional
        Sample weights w(y_i) for the regional mean and profiles. Default: uniform.
    g : array-like, shape (N,), optional
        Domain position g(y_i) for regional aggregation. Default: y itself.

    Notes
    -----
    Corrected scores are always formed per sample, q_corr,i = q_i - c_Delta,i, and
    only then aggregated. For the realized variant c_Delta,i is a constant, so
    subtracting before or after aggregation is equivalent; for the expected variant
    it is not, and only the per-sample route reproduces the framework.
    """

    def __init__(
        self,
        y,
        mu,
        sigma,
        *,
        statistic: Union[str, ResidualStatistic, tuple] = "squared",
        discrepancy: Union[str, Discrepancy] = "squared_ratio",
        mode: Literal["realized", "expected"] = "realized",
        estimator: Literal["kernel", "ENCE"] = "kernel",
        estimator_kwargs: dict | None = None,
        w=None,
        g=None,
    ):
        self.y = as_1d_float(y, "y")
        n = self.y.size
        self.mu = as_1d_float(mu, "mu", n)
        self.sigma = as_1d_float(sigma, "sigma", n, positive=True)
        self.w = None if w is None else as_1d_float(w, "w", n, non_negative=True)
        self.g = self.y if g is None else as_1d_float(g, "g", n)

        self.mode = check_choice(mode, ("realized", "expected"), "mode")
        self.estimator = check_choice(estimator, ("kernel", "ENCE"), "estimator")
        if self.estimator == "ENCE" and self.mode != "expected":
            raise ValueError("estimator='ENCE' is only available in mode='expected'.")
        self.statistic = resolve_statistic(statistic)
        self.discrepancy = resolve_discrepancy(discrepancy)
        self.estimator_kwargs = dict(estimator_kwargs or {})
        self._c_delta_cache: dict[tuple, np.ndarray] = {}

    # ------------------------------------------------------------------
    # Per-sample quantities (lazy, cached)
    # ------------------------------------------------------------------
    @cached_property
    def e(self) -> np.ndarray:
        """Residuals e_i = y_i - mu_i."""
        return self.y - self.mu

    @cached_property
    def s_err(self) -> np.ndarray:
        """S_err(e_i): the residual statistic of the model (before any smoothing)."""
        return self.statistic.s_err(self.e)

    @cached_property
    def s_unc(self) -> np.ndarray:
        """S_unc(sigma_i): the reference value of the residual statistic."""
        return self.statistic.s_unc(self.sigma)

    @cached_property
    def r(self) -> np.ndarray:
        """
        Ratio r_i = S_err / S_unc: realized (e_i^2/sigma_i^2) or expected, i.e. the
        kernel estimate of E[S_err(e) | sigma_i] divided by S_unc(sigma_i).

        Computing it also caches ``geometry``, the reference geometry of the very
        same kernel weights, so the correction can never be based on a different
        bandwidth, kernel or window than the estimate.
        """
        s_err, s_unc = self.s_err, self.s_unc
        if self.mode == "realized":
            return s_err / s_unc
        estimate, n_eff, mean_factor, var_factor = kernel_conditional_mean(
            self.sigma, s_err, scale=s_unc, **self.estimator_kwargs
        )
        assert estimate is not None, "values were given, so an estimate is returned"
        self.geometry = (n_eff, mean_factor, var_factor)
        if self.estimator == "ENCE":
            return np.sqrt(estimate) / np.sqrt(s_unc)
        return estimate / s_unc

    @cached_property
    def q(self) -> np.ndarray:
        """Per-sample QQU contributions q_i = Delta(r_i), uncorrected."""
        return self.discrepancy(self.r)

    @cached_property
    def geometry(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        (n_eff, mean_factor, var_factor) of the estimator under the reference,
        expected mode only:

            mean(r^E_i) = mu_ref * mean_factor_i, var(r^E_i) = Var_ref * var_factor_i

        with mean_factor_i = sum_j kappa_ij a_ij and var_factor_i = sum_j kappa_ij^2
        a_ij^2 for a_ij = S_unc(sigma_j)/S_unc(sigma_i). Set as a side effect of `r`
        (same kernel rows, one traversal); computed on demand if `r` has not been
        accessed yet.
        """
        if self.mode != "expected":
            raise ValueError("geometry is only defined for mode='expected'.")
        kernel = kernel_conditional_mean(
            self.sigma, scale=self.s_unc, **self.estimator_kwargs
        )
        return kernel[1], kernel[2], kernel[3]

    @cached_property
    def n_eff(self) -> np.ndarray:
        """Effective pool size n_eff,i = 1 / sum_j kappa_ij^2 (expected mode only)."""
        if self.mode != "expected":
            raise ValueError("n_eff is only defined for mode='expected'.")
        return self.geometry[0]

    # ------------------------------------------------------------------
    # Reference correction c_Delta
    # ------------------------------------------------------------------
    def c_delta_i(self, *,
                  aggregator: Aggregator = "mean",
                  n_grid: int = 4001) -> np.ndarray:
        """
        Per-sample reference values c_Delta,i, cached per configuration.

        Constant across samples for the realized variant, geometry-dependent for
        the expected one. ``aggregator`` selects which functional of the reference
        distribution is subtracted and must match the aggregator of the corrected
        score; ``method`` chooses between the closed form of the article
        ("analytic") and the numeric reference law ("distribution").
        """
        key = (aggregator, n_grid)
        if key not in self._c_delta_cache:
            geometry = self.geometry if self.mode == "expected" else None
            self._c_delta_cache[key] = compute_c_delta(
                self.y, self.mu, self.sigma,
                statistic=self.statistic, discrepancy=self.discrepancy,
                mode=self.mode, aggregator=aggregator,
                estimator=self.estimator, estimator_kwargs=self.estimator_kwargs,
                geometry=geometry, n_grid=n_grid,
            )
        return self._c_delta_cache[key]

    def c_delta(self,
                aggregator: Aggregator = "mean",
                *,
                t: float | None = None,
                g=None,
                n_grid: int = 4001) -> float:
        """
        Dataset-wide (t=None) or regional aggregate of the per-sample corrections,
        using the same region and weights as ``avg``:

            c_delta(t) = sum_i w_i phi_t(y_i) c_Delta,i / sum_i w_i phi_t(y_i),

        so that ``avg(t=t, corrected=True) == avg(t=t) - c_delta(t=t)``.
        """
        g_arr = None
        if t is not None:
            g_arr = self.g if g is None else as_1d_float(g, "g", self.y.size)
        return aggregate_c_delta(
            self.c_delta_i(aggregator=aggregator, n_grid=n_grid),
            aggregator=aggregator, g=g_arr, t=t, w=self.w,
        )

    def q_corrected(self, *,
                    aggregator: Aggregator = "mean",
                    n_grid: int = 4001) -> np.ndarray:
        """Per-sample corrected scores q_corr,i = q_i - c_Delta,i."""
        return self.q - self.c_delta_i(aggregator=aggregator, n_grid=n_grid)

    # ------------------------------------------------------------------
    # Aggregations
    # ------------------------------------------------------------------
    def avg(self, t: float | None = None, *,
            g=None,
            corrected: bool = False,
            n_grid: int = 4001) -> float:
        """
        QQU_avg: dataset-wide (t=None) or over the region {g(y_i) >= t}.

        With corrected=True the mean-aggregator reference values are subtracted per
        sample before averaging, giving QQU_avg,corr.
        """
        g_arr = None
        if t is not None:
            g_arr = self.g if g is None else as_1d_float(g, "g", self.y.size)
        q = (self.q_corrected(aggregator="mean", n_grid=n_grid)
             if corrected else self.q)
        return QQU_avg(q, g=g_arr, t=t, w=self.w)

    def median(self, *, corrected: bool = False,
               n_grid: int = 4001) -> float:
        """
        Dataset-wide median of q (unweighted).

        With corrected=True this is the median of the corrected scores,
        median_i(q_i - c_Delta,i), using the median functional of the reference
        distribution. For the realized variant c_Delta,i is constant, so it equals
        median(q) - c_Delta; for the expected variant the two routes differ and only
        the per-sample one is consistent with the scores.
        """
        q = (self.q_corrected(aggregator="median", n_grid=n_grid)
             if corrected else self.q)
        return QQU_median(q)

    def avg_profile(self, g=None, *, corrected: bool = False,
                    n_grid: int = 4001):
        """
        Regional profile QQU_avg(t) at every threshold induced by g.

        Returns
        -------
        t_values (descending), qqu_avg_values, n_values (samples contributing at
        each t; the profile gets noisier as this shrinks).

        With corrected=True the per-sample corrections are subtracted before the
        cumulative sums, so the profile is that of the corrected scores.
        """
        g_arr = self.g if g is None else as_1d_float(g, "g", self.y.size)
        q = (self.q_corrected(aggregator="mean", n_grid=n_grid)
             if corrected else self.q)
        return QQU_avg_profile(q, g_arr, self.w)

    def sum_profile(self, g=None):
        """Unnormalized QQU_sum(t) at every threshold induced by g."""
        g_arr = self.g if g is None else as_1d_float(g, "g", self.y.size)
        return QQU_sum_profile(self.q, g_arr, self.w)

    def __repr__(self) -> str:
        disc = getattr(self.discrepancy, "__name__", repr(self.discrepancy))
        return (f"QQU(n={self.y.size}, mode={self.mode!r}, "
                f"statistic={self.statistic.name!r}, discrepancy={disc!r})")