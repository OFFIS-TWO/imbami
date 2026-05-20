import pandas as pd
from typing import Union
import logging


def binned_loss(
    y: pd.Series,
    err: pd.Series,
    bins: Union[pd.IntervalIndex, int]
) -> tuple[pd.Series, pd.Series, pd.IntervalIndex]:
    """
    Calculate the mean loss per bin over the 'y' domain and the number of
    instances in each bin.

    Parameters
    ----------
    y : pd.Series
        Target variable used for binning.

    err : pd.Series
        Error or loss values.

    bins : int or pd.IntervalIndex
        If int:
            Number of equal-width bins over y.
            The function automatically constructs bins such that:
            - minimum y is included in the first bin
            - maximum y is included in the last bin
        If pd.IntervalIndex:
            Predefined bin structure used directly without modification.

            Example (recommended construction; includes lowest and highest bounds):
                _, interval_bins = pd.cut(y, bins=5, retbins=True, include_lowest=True)
                bins = pd.IntervalIndex.from_breaks(interval_bins)
            

    Returns
    -------
    tuple[pd.Series, pd.Series, pd.IntervalIndex]
        - Mean loss per ranked bin.
        - Number of instances per ranked bin.
        - IntervalIndex used for binning.
    """

    # Create IntervalIndex if number of bins is provided
    if isinstance(bins, int):
        if bins <= 0:
            raise ValueError("`bins` must be a positive integer.")

        _, interval_bins = pd.cut(y, bins=bins, retbins=True, include_lowest=True)
        bins = pd.IntervalIndex.from_breaks(interval_bins)

    elif not isinstance(bins, pd.IntervalIndex):
        raise TypeError(
            "`bins` must be either a pd.IntervalIndex or an integer."
        )

    # Assign samples to bins
    binned = pd.cut(y, bins=bins, include_lowest=False)

    # ----------------------------------------
    # CHECK: values not assigned to any bin
    # ----------------------------------------
    n_unbinned = binned.isna().sum()

    if n_unbinned > 0:
        logging.warning(f"{n_unbinned} samples in 'y' are outside the bin ranges and were not assigned to any bin.")

    # Count instances per bin
    bin_counts = binned.value_counts().sort_index()

    # Rank bins by frequency
    bin_ranks = bin_counts.rank(ascending=True, method="first").astype(int) #'first' is used to decide between bins with equal bin_count.

    # Map interval bins to ranked bins
    ranked_bins = binned.map(bin_ranks)

    # Count ranked bin occurrences
    ranked_bin_counts = ranked_bins.value_counts().sort_index()

    # Mean loss per ranked bin
    mean_loss_per_bin = err.groupby(ranked_bins, observed=False).mean().sort_index()

    return mean_loss_per_bin, ranked_bin_counts, bins