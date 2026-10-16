"""
Residual statistics of the QQU framework.

A residual statistic is a PAIR of functions:

    s_err(e)      statistic of the residual e = y - mu
    s_unc(sigma)  reference value s_err attains under the assumed
                  uncertainty-error relationship

The ratio r = s_err(e) / s_unc(sigma) is what the discrepancy acts on.
(The module is not called ``statistics`` to avoid shadowing the stdlib.)
"""
from dataclasses import dataclass
from typing import Callable, Union

import numpy as np

ArrayFn = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class ResidualStatistic:
    name: str
    s_err: ArrayFn
    s_unc: ArrayFn

    def ratio(self, e: np.ndarray, sigma: np.ndarray) -> np.ndarray:
        """Realized ratio r = S_err(e) / S_unc(sigma)."""
        return self.s_err(e) / self.s_unc(sigma)


# Needs only second-moment consistency E[e^2 | sigma] = sigma^2.
SQUARED = ResidualStatistic(
    name="squared",
    s_err=lambda e: e ** 2,
    s_unc=lambda sigma: sigma ** 2,
)

# Needs a distributional assumption: here e | sigma ~ N(0, sigma^2),
# so that E|e| = sqrt(2/pi) * sigma.
ABSOLUTE = ResidualStatistic(
    name="absolute",
    s_err=lambda e: np.abs(e),
    s_unc=lambda sigma: np.sqrt(2.0 / np.pi) * sigma,
)

# Needs a distributional assumption: here e | sigma ~ N(0, sigma^2),
# so that E|e| = sqrt(2/pi) * sigma.
ABSOLUTE_PLAIN = ResidualStatistic(
    name="absolute_plain",
    s_err=lambda e: np.abs(e),
    s_unc=lambda sigma: sigma,
)



STATISTICS: dict[str, ResidualStatistic] = {
    "squared": SQUARED,
    "absolute": ABSOLUTE,
    "absolute_plain" : ABSOLUTE_PLAIN
}


def resolve_statistic(
    statistic: Union[str, ResidualStatistic, tuple[ArrayFn, ArrayFn]],
) -> ResidualStatistic:
    """Accept a registered name, a ResidualStatistic, or a (s_err, s_unc) tuple."""
    if isinstance(statistic, ResidualStatistic):
        return statistic
    if isinstance(statistic, str):
        try:
            return STATISTICS[statistic]
        except KeyError:
            raise ValueError(
                f"Unknown statistic {statistic!r}. Choose one of "
                f"{sorted(STATISTICS)}, a (s_err, s_unc) tuple, or a ResidualStatistic."
            ) from None
    if isinstance(statistic, tuple) and len(statistic) == 2 and all(map(callable, statistic)):
        return ResidualStatistic("custom", statistic[0], statistic[1])
    raise TypeError(
        "statistic must be a string, a ResidualStatistic, or a tuple (s_err, s_unc) of callables."
    )