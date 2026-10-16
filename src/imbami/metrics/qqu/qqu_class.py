"""
User-facing QQU object.

The user supplies y, mu, sigma once, plus the choice of residual statistic,
discrepancy and mode. The per-sample contributions q_i are computed lazily
and cached; all aggregations reuse them.

    qqu = QQU(y, mu, sigma)                       # framework defaults
    qqu.avg(corrected=True)
    t, avg, n = qqu.profile(g=y)
"""
from functools import cached_property
from typing import Literal, Union

import numpy as np

from .aggregation import QQU_avg, QQU_avg_profile, QQU_median, QQU_sum_profile
from .discrepancy_functions import Discrepancy, resolve_discrepancy
from .expected import kernel_conditional_mean, rolling_conditional_mean
from .realized import compute_c_delta, reference_c_delta
from .residual_statistics import ResidualStatistic, resolve_statistic


def _as_1d_float(x, name: str, n: int | None = None) -> np.ndarray:
    arr = np.array(x, dtype=float)  # copy, so later changes by the caller cannot corrupt the cache
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")
    if n is not None and arr.size != n:
        raise ValueError(f"{name} must have the same length as y ({n}), got {arr.size}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    arr.flags.writeable = False
    return arr


class QQU:
    """
    Quality of Uncertainty (QQU) evaluation of a probabilistic regression model.

    Parameters
    ----------
    y, mu, sigma : array-like, shape (N,)
        Ground truth, predicted mean, predicted standard deviation.
    statistic : str | ResidualStatistic | (s_err, s_unc), default "squared"
        Residual statistic. Registered: "squared" (e^2 vs sigma^2),
        "absolute" (|e| vs sqrt(2/pi)*sigma, assumes Gaussian residuals),
        "absolute_plain" (|e| vs sigma).
        Custom: a tuple of two callables (s_err(e), s_unc(sigma)).
    discrepancy : str | callable, default "squared_ratio"
        Delta(r). Registered: "ratio", "log", "squared_ratio", "squared_log".
        Custom: any callable on the ratio array r, e.g.
        ``make_asymmetric_squared(0.5)``.
    mode : {"realized", "expected"}, default "realized"
        "realized": r_i = S_err(e_i) / S_unc(sigma_i)
        "expected": r_i = E[S_err(e) | sigma_i] / S_unc(sigma_i), estimated
        from neighbouring samples in sigma.
    estimator : {"kernel", "rolling", "ENCE"}, default "kernel"
        Conditional-mean estimator (expected mode only).
        'ENCE' uses a kernel-based estimator, but square-roots s_unc and e_err
        to align with RMSE and RMV from ENCE.
    estimator_kwargs : dict, optional
        Passed on to the estimator, e.g. ``{"bandwidth_factor": 2.0}`` or
        ``{"window_size": 50}`` (see expected.py).
    w : array-like, shape (N,), optional
        Sample weights w(y_i) for the regional mean / profiles. Default: uniform.
    g : array-like, shape (N,), optional
        Domain position g(y_i) for regional aggregation. Default: y itself.
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
        estimator: Literal["kernel", "rolling", "ENCE"] = "kernel",
        estimator_kwargs: dict | None = None,
        w=None,
        g=None,
    ):
        self.y = _as_1d_float(y, "y")
        n = self.y.size
        self.mu = _as_1d_float(mu, "mu", n)
        self.sigma = _as_1d_float(sigma, "sigma", n)
        if np.any(self.sigma <= 0):
            raise ValueError("sigma must be strictly positive.")

        self.w = None if w is None else _as_1d_float(w, "w", n)
        if self.w is not None and np.any(self.w < 0):
            raise ValueError("w must be non-negative.")
        self.g = self.y if g is None else _as_1d_float(g, "g", n)

        if mode not in ("realized", "expected"):
            raise ValueError(f"mode must be 'realized' or 'expected', got {mode!r}")
        if estimator not in ("kernel", "rolling", "ENCE"):
            raise ValueError(f"estimator must be 'kernel', 'rolling' or 'ENCE', got {estimator!r}")

        self.statistic = resolve_statistic(statistic)
        self.discrepancy = resolve_discrepancy(discrepancy)
        self.mode = mode
        self.estimator = estimator
        self.estimator_kwargs = dict(estimator_kwargs or {})

    # ------------------------------------------------------------------
    # Per-sample quantities (lazy, cached)
    # ------------------------------------------------------------------
    @cached_property
    def e(self) -> np.ndarray:
        """Residuals e_i = y_i - mu_i."""
        return self.y - self.mu

    @cached_property
    def r(self) -> np.ndarray:
        """Ratio r_i: realized or conditionally expected, depending on `mode`."""
        self.s_unc = self.statistic.s_unc(self.sigma)
        self.s_err = self.statistic.s_err(self.e)
        if self.mode == "realized":
            return self.s_err / self.s_unc
        if self.estimator == "rolling":
            self.s_err = rolling_conditional_mean(self.sigma, self.s_err, **self.estimator_kwargs)
        elif self.estimator == "ENCE":
            self.s_err = np.sqrt(kernel_conditional_mean(self.sigma, self.s_err, **self.estimator_kwargs))
            self.s_unc = np.sqrt(self.s_unc)
        else:
            self.s_err = kernel_conditional_mean(self.sigma, self.s_err, **self.estimator_kwargs)
        return self.s_err / self.s_unc

    @cached_property
    def q(self) -> np.ndarray:
        """Per-sample QQU contributions q_i = Delta(r_i) (uncorrected)."""
        return self.discrepancy(self.r)

    # ------------------------------------------------------------------
    # Correction constant (realized mode only)
    # ------------------------------------------------------------------
    def c_delta(
        self,
        aggregator: Literal["mean", "median"] = "mean",
        reference: Literal["theoretical", "empirical"] = "empirical",
    ) -> float:
        """
        Correction constant c_Delta for the chosen aggregator.

        reference="theoretical": value under Gaussian standardized residuals
            (identical for every model, so corrected scores are comparable).
        reference="empirical": derived from this model's own residual shape
            (rescaled to unit variance), see ``compute_c_delta``.

        Only defined for mode="realized" (the framework requires no
        correction for expected QQU).
        """
        if self.mode != "realized":
            raise ValueError("c_delta is only defined for mode='realized'.")
        if reference == "theoretical":
            return reference_c_delta(self.statistic, self.discrepancy, aggregator)
        if reference == "empirical":
            return compute_c_delta(
                self.y, self.mu, self.sigma,
                statistic=self.statistic, discrepancy=self.discrepancy,
                aggregator=aggregator,
            ).c_delta
        raise ValueError(f"reference must be 'theoretical' or 'empirical', got {reference!r}")

    def _offset(self, corrected: bool, aggregator: Literal["mean", "median"], reference: Literal["theoretical", "empirical"]) -> float:
        return self.c_delta(aggregator, reference) if corrected else 0.0

    # ------------------------------------------------------------------
    # Aggregations
    # ------------------------------------------------------------------
    def avg(self, t: float | None = None, *, g=None, corrected: bool = False,
            reference: Literal["theoretical", "empirical"] = "theoretical") -> float:
        """
        QQU_avg: dataset-wide (t=None) or over the region {g(y_i) >= t}.

        With corrected=True the Gaussian-reference constant for the mean is
        subtracted (realized mode only). For regional averages this assumes
        the same reference distribution in every region.
        """
        g_arr = None
        if t is not None:
            g_arr = self.g if g is None else _as_1d_float(g, "g", self.y.size)
        value = QQU_avg(self.q, g=g_arr, t=t, w=self.w)
        return value - self._offset(corrected, "mean", reference)

    def median(self, *, corrected: bool = False,
               reference: Literal["theoretical", "empirical"] = "theoretical") -> float:
        """Dataset-wide median of q (unweighted)."""
        return QQU_median(self.q) - self._offset(corrected, "median", reference)

    def avg_profile(self, g=None, *, corrected: bool = False,
                reference: Literal["theoretical", "empirical"] = "theoretical"):
        """
        Regional profile QQU_avg(t) at every threshold induced by g.

        Returns
        -------
        t_values (descending), qqu_avg_values, n_values (samples contributing
        at each t; the profile gets noisier as this shrinks).
        """
        g_arr = self.g if g is None else _as_1d_float(g, "g", self.y.size)
        t, avg, n = QQU_avg_profile(self.q, g_arr, self.w)
        return t, avg - self._offset(corrected, "mean", reference), n

    def sum_profile(self, g=None):
        """Unnormalized QQU_sum(t) at every threshold induced by g."""
        g_arr = self.g if g is None else _as_1d_float(g, "g", self.y.size)
        return QQU_sum_profile(self.q, g_arr, self.w)

    def __repr__(self) -> str:
        disc = getattr(self.discrepancy, "__name__", repr(self.discrepancy))
        return (f"QQU(n={self.y.size}, mode={self.mode!r}, "
                f"statistic={self.statistic.name!r}, discrepancy={disc!r})")