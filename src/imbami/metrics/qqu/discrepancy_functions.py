"""
Discrepancy functions Delta(r) of the QQU framework.

Each discrepancy maps the ratio r = S_err / S_unc to a non-negative penalty
with Delta(1) = 0. Because r is dimensionless, scale invariance (R1) holds
by construction. A special case is 'signed_log' which allows negative penalty.

A discrepancy is any callable ``np.ndarray -> np.ndarray``. The most common
ones are registered under a string name in ``DISCREPANCIES``.
"""
from typing import Callable, Union

import numpy as np

Discrepancy = Callable[[np.ndarray], np.ndarray]

# Floor for r inside logarithms (r == 0 happens when a residual is exactly 0).
EPS = 1e-12


def _safe_log(r: np.ndarray) -> np.ndarray:
    return np.log(np.maximum(r, EPS))


def ratio_discrepancy(r: np.ndarray) -> np.ndarray:
    """|r - 1|. Asymmetric (O1), but not tolerant to small deviations (O2)."""
    return np.abs(r - 1.0)


def log_discrepancy(r: np.ndarray) -> np.ndarray:
    """|log r|. Symmetric under r -> 1/r (O1), not tolerant to small deviations."""
    return np.abs(_safe_log(r))

def signed_log_discrepancy(r):
    """log r (negative for r < 1). Not a penalty; for mean-aggregated diagnostics like log_z_r."""
    return _safe_log(r)

def squared_ratio_discrepancy(r: np.ndarray) -> np.ndarray:
    """(r - 1)^2. General-purpose default: asymmetric and tolerant (O1, O2)."""
    return (r - 1.0) ** 2


def squared_log_discrepancy(r: np.ndarray) -> np.ndarray:
    """(log r)^2. Symmetric under r -> 1/r and tolerant (O1, O2)."""
    return _safe_log(r) ** 2


def make_asymmetric_squared(lam: float) -> Discrepancy:
    """
    Explicitly weighted asymmetric discrepancy

        Delta(r) = max(r-1, 0)^2 + lam * max(1-r, 0)^2,   lam in (0, 1)

    r > 1 means the error exceeds the predicted uncertainty (overconfidence);
    lam < 1 down-weights underconfidence (r < 1).
    """
    if not 0.0 < lam < 1.0:
        raise ValueError(f"lam must be in (0, 1), got {lam}.")

    def asymmetric_squared(r: np.ndarray) -> np.ndarray:
        return np.maximum(r - 1.0, 0.0) ** 2 + lam * np.maximum(1.0 - r, 0.0) ** 2

    asymmetric_squared.__name__ = f"asymmetric_squared(lam={lam})"
    return asymmetric_squared


DISCREPANCIES: dict[str, Discrepancy] = {
    "ratio": ratio_discrepancy,
    "log": log_discrepancy,
    "squared_ratio": squared_ratio_discrepancy,
    "squared_log": squared_log_discrepancy,
    "signed_log": signed_log_discrepancy
}


def resolve_discrepancy(discrepancy: Union[str, Discrepancy]) -> Discrepancy:
    """Turn a registered name or a callable into a discrepancy callable."""
    if callable(discrepancy):
        return discrepancy
    if isinstance(discrepancy, str):
        try:
            return DISCREPANCIES[discrepancy]
        except KeyError:
            raise ValueError(
                f"Unknown discrepancy {discrepancy!r}. "
                f"Choose one of {sorted(DISCREPANCIES)} or pass a callable."
            ) from None
    raise TypeError("discrepancy must be a string or a callable.")