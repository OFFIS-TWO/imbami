from typing import Literal, Any, Type, Dict
import pandas as pd
from .sampling.crbsmogn import apply_crbsmogn
from .sampling.csmogn import apply_csmogn
from .sampling.wercs import apply_wercs
from ..utils.validation import _validate_parameters



def apply_sampler(
    sampler_type: Literal["crbsmogn", "csmogn", "wercs"],
    data: pd.DataFrame,
    target_column: str,
    relevance_values: pd.Series,
    **kwargs
) -> pd.DataFrame:
    """
    Factory function that applies the specified sampling technique to the dataset.
    
    This function STRICTLY requires ALL parameters to be explicitly provided - 
    NO DEFAULTS ARE APPLIED, even for parameters that have defaults in the 
    underlying sampler functions.
    
    Args:
        sampler_type: String identifier for the sampling technique
        data: Input dataset to be sampled
        target_column: Name of the target variable column (required for crbsmogn and csmogn)
        relevance_values: Relevance scores for each sample
        **kwargs: ALL parameters must be explicitly provided
        
    Returns:
        Oversampled dataset
        
    Raises:
        ValueError: If any required parameter is missing
        TypeError: If any parameter has incorrect type
    """
    # Map string identifiers to apply functions
    sampler_functions = {
        "crbsmogn": apply_crbsmogn,
        "csmogn": apply_csmogn,
        "wercs": apply_wercs
    }
    
    # Check if sampler type exists
    if sampler_type not in sampler_functions:
        available_types = ", ".join(sampler_functions.keys())
        raise ValueError(f"Unknown sampler type: '{sampler_type}'. Available types: {available_types}")
    
    # Handle each sampler type with strict parameter validation
    if sampler_type == "crbsmogn":
        if target_column is None:
            raise ValueError("target_column is required for crbsmogn sampler")
        
        # ALL parameters for crbSMOGN (including those with defaults in the underlying function)
        required_params = {
            "enable_undersampling": bool,
            "min_acceptable_relevance": float,
            "max_acceptable_relevance": float,
            "num_bins": int,
            "allowed_bin_deviation": int,
            "noise_factor": float,
            "ignore_categorical_similarity": bool
        }
        
        # Validate parameters
        validated_params = _validate_parameters(required_params, kwargs, "crbsmogn")
        
        return apply_crbsmogn(
            data=data,
            target_column=target_column,
            relevance_values=relevance_values,
            enable_undersampling=validated_params["enable_undersampling"],
            min_acceptable_relevance=validated_params["min_acceptable_relevance"],
            max_acceptable_relevance=validated_params["max_acceptable_relevance"],
            num_bins=validated_params["num_bins"],
            allowed_bin_deviation=validated_params["allowed_bin_deviation"],
            noise_factor=validated_params["noise_factor"],
            ignore_categorical_similarity=validated_params["ignore_categorical_similarity"]
        )
    
    elif sampler_type == "csmogn":
        if target_column is None:
            raise ValueError("target_column is required for csmogn sampler")
        
        # ALL parameters for cSMOGN (including those with defaults in the underlying function)
        required_params = {
            "enable_undersampling": bool,
            "oversample_rate": float,
            "undersample_rate": float,
            "num_bins": int,
            "allowed_bin_deviation": int,
            "noise_factor": float,
            "ignore_categorical_similarity": bool,
            "knns": int
        }
        
        # Validate parameters
        validated_params = _validate_parameters(required_params, kwargs, "csmogn")
        
        return apply_csmogn(
            data=data,
            target_column=target_column,
            relevance_values=relevance_values,
            enable_undersampling=validated_params["enable_undersampling"],
            oversample_rate=validated_params["oversample_rate"],
            undersample_rate=validated_params["undersample_rate"],
            num_bins=validated_params["num_bins"],
            allowed_bin_deviation=validated_params["allowed_bin_deviation"],
            noise_factor=validated_params["noise_factor"],
            ignore_categorical_similarity=validated_params["ignore_categorical_similarity"],
            knns=validated_params["knns"]
        )
    
    elif sampler_type == "wercs":
        # ALL parameters for WERCS (including those with defaults in the underlying function)
        required_params = {
            "enable_undersampling": bool,
            "oversampling_rate": float,
            "undersampling_rate": float
        }
        
        # Validate parameters
        validated_params = _validate_parameters(required_params, kwargs, "wercs")
        
        return apply_wercs(
            data=data,
            relevance_values=relevance_values,
            enable_undersampling=validated_params["enable_undersampling"],
            oversampling_rate=validated_params["oversampling_rate"],
            undersampling_rate=validated_params["undersampling_rate"]
        )






