"""Shared helpers: input validation, bandwidth selection and rounding."""
import numpy as np
from typing import TypeVar
from collections.abc import Iterable


def as_1d_float(
    x,
    name: str = "array",
    n: int | None = None,
    *,
    positive: bool = False,
    non_negative: bool = False,
) -> np.ndarray:
    """
    Convert ``x`` to a read-only 1D float array and validate it.

    Parameters
    ----------
    x : array-like
        Values to check.
    name : str, default "array"
        Name used in the error messages.
    n : int, optional
        Required length; checked if given.
    positive : bool, default False
        Require all values > 0 (e.g. sigma, bandwidth scales).
    non_negative : bool, default False
        Require all values >= 0 (e.g. sample weights).

    Returns
    -------
    np.ndarray, shape (n,)
        A copy of the input, so later changes by the caller cannot corrupt cached
        quantities.
    """
    arr = np.array(x, dtype=float)
    if arr.ndim != 1:
        raise ValueError(f"{name} must be a 1D array.")
    if n is not None and arr.size != n:
        raise ValueError(f"{name} must have length {n}, got {arr.size}.")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} contains NaN or infinite values.")
    if positive and np.any(arr <= 0):
        raise ValueError(f"{name} must be strictly positive.")
    if non_negative and np.any(arr < 0):
        raise ValueError(f"{name} must be non-negative.")
    arr.flags.writeable = False
    return arr





TChoice = TypeVar("TChoice", bound=str)

def check_choice(value: TChoice, allowed: Iterable[str], name: str) -> TChoice:
    """
    Return ``value`` if it is one of ``allowed``, else raise a ValueError.

    Generic in the type of ``value``, so a parameter declared as
    ``Literal["realized", "expected"]`` keeps that type instead of widening to
    ``str`` when it is passed through this check.
    """
    if value not in allowed:
        raise ValueError(f"{name} must be one of {tuple(allowed)}, got {value!r}.")
    return value

def round_significant(x: np.ndarray, sig: int = 6) -> np.ndarray:
    """
    Round to ``sig`` significant digits, elementwise.

    Used to merge near-identical values (e.g. pool sizes) into one lookup key.
    """
    x = np.asarray(x, dtype=float)
    scale = 10.0 ** (sig - 1 - np.floor(np.log10(np.maximum(np.abs(x), 1e-300))))
    return np.round(x * scale) / scale

def silverman_bandwidth(sigma: np.ndarray) -> float:
    """Silverman's rule of thumb for a kernel bandwidth on ``sigma``."""
    n = sigma.size
    if n < 2:
        return 1e-8
    std = np.std(sigma, ddof=1)
    iqr = np.subtract(*np.percentile(sigma, [75, 25]))
    scale = min(std, iqr / 1.34)
    bw = 0.9 * scale * n ** (-1 / 5)
    # Fallback if all sigma values are (nearly) identical
    return bw if bw > 0 else 1e-8