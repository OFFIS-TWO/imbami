from typing import Literal
import pandas as pd
from .sampling.crbsmogn import apply_crbsmogn
from .sampling.csmogn import apply_csmogn
from .sampling.wercs import apply_wercs
from .sampling.smogn import apply_smogn
from .sampling.wsmoter import apply_wsmoter
from ..utils.validation import extract_explicit_parameters

SAMPLING_METHODS = {
        "crbsmogn": apply_crbsmogn,
        "csmogn": apply_csmogn,
        "wercs": apply_wercs,
        "smogn": apply_smogn,
        "wsmoter": apply_wsmoter
    }

def sampling_factory(
                sampler_type: Literal["crbsmogn", "csmogn", "wercs"],
                data: pd.DataFrame,
                target_column: str | None,
                relevance_values: pd.Series,
                **kwargs) -> pd.DataFrame:
    """
    Factory function that applies the specified sampling technique to the dataset.

    STRICT: all parameters must be explicitly provided — defaults are ignored.

    Args:
        sampler_type: String identifier for the sampling technique
        data: Input dataset to be sampled
        target_column: Name of the target variable column (required for crbsmogn and csmogn)
        relevance_values: Relevance scores for each sample
        **kwargs: Parameters for the underlying sampler

    Returns:
        Oversampled dataset

    Raises:
        ValueError: Missing required parameters
        TypeError: Parameter type mismatch
    """

    if sampler_type not in SAMPLING_METHODS:
        available_types = ", ".join(SAMPLING_METHODS.keys())
        raise ValueError(f"Unknown sampler type '{sampler_type}'. Available types: {available_types}")

    sampler_func = SAMPLING_METHODS[sampler_type]

    # target_column required for smogn-based samplers
    if sampler_type in {"crbsmogn", "csmogn"} and target_column is None:
        raise ValueError(f"target_column is required for {sampler_type} sampler")

    # Extract the parameters strictly required by the sampler function
    validated_params, unused_params = extract_explicit_parameters(
        sampler_func,
        data=data,
        target_column=target_column,
        relevance_values=relevance_values,
        **kwargs
    )

    return sampler_func(**validated_params)