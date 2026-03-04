import numpy as np
import pandas as pd



def bin_loss(y: pd.Series, err: pd.Series, intervals: pd.IntervalIndex) -> tuple[pd.Series, pd.Series]:
    """
    Calculate the mean loss per bin and the number of instances in each bin.

    Parameters:
    y (pd.Series): The target variable. 
        The function uses this variable to determine which bin each error value corresponds to.
    err (pd.Series): The error or loss variable.
    intervals (pd.IntervalIndex): The intervals to bin the target variable. 
        The function uses these intervals to partition the target variable into bins.

    Returns:
    tuple[pd.Series, pd.Series]: A tuple containing the mean loss per bin and the number of instances in each bin.
    """

    # bin computation
        # Step 1: Calculate bins
    bins = pd.cut(y, bins=intervals, include_lowest=False, retbins = False)
        # Step 2: Count the number of instances in each bin
    bin_counts = bins.value_counts().sort_index()
        # Step 3: Rank the bins based on the number of instances
    bin_ranks = bin_counts.rank(ascending=True, method='first').astype(int) #'first' is used to decide between bins with equal bin_count.
        # Map the ranks back to the bins
    ranked_bins = bins.map(bin_ranks)
    ranked_bin_counts = ranked_bins.value_counts().sort_index()
        # calc mean loss per bin 
    mean_loss_per_bin = err.groupby(ranked_bins, observed = False).mean().sort_index()

    return mean_loss_per_bin, ranked_bin_counts