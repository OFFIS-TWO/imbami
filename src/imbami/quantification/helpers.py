'''
This script contains basic utility and helper functions.
'''

import bisect
import numpy as np
import pandas as pd

def drop_inf_and_nan(data: pd.Series | pd.DataFrame, drop_inf_nan: bool = True) -> pd.Series | pd.DataFrame:
    """
    Remove infinite and NaN values from a pandas Series or DataFrame.

    This function replaces infinite values (both positive and negative) with NaN,
    then either drops rows/columns containing NaN values or raises an error based
    on the drop_inf_nan parameter.

    Parameters:
    -----------
    data : pd.Series or pd.DataFrame
        The input data to clean
    drop_inf_nan : bool, default True
        If True, drops rows/columns with NaN/infinite values
        If False, raises a ValueError when NaN/infinite values are found

    Returns:
    --------
    pd.Series or pd.DataFrame
        Cleaned data with infinite and NaN values removed or handled according to
        the drop_inf_nan parameter

    Raises:
    -------
    ValueError
        If drop_inf_nan is False and NaN/infinite values are present in the data
    """
    # Replace inf values with NaN to handle them together
    cleaned = data.replace([np.inf, -np.inf], np.nan)

    # Count total number of NaN values
    if isinstance(cleaned, pd.Series):
        counter = cleaned.isna().sum()
    else:
        counter = cleaned.isna().sum().sum()

    # If any NaN/infinite values were found
    if counter != 0:
        if drop_inf_nan == True:
            # Drop rows with NaN values
            cleaned = cleaned.dropna(axis = 0)
        else:
            raise ValueError(f"Found {counter} inf/NaN values in the input. Either set 'drop_inf_nan' to True or drop them manually.")
    return cleaned