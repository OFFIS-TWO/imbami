from typing import Literal
from .base_relevance_function import RelevanceFunctionBase
from .density_distance_relevance import DensityDistanceRelevance
from .density_ratio_relevance import DensityRatioRelevance
from .histogram_based_relevance import HistogramBasedRelevance
from .interpolation_with_controlpoints import InterpolationWithControlPoints
from .label_distribution_smoothing import LabelDistributionSmoothing
from .kernel_density_relevance import KernelDensityRelevance
from .denseweight import DenseWeight
from ..utils.validation import extract_explicit_parameters

RELEVANCE_FUNCTIONS = {
    "density_distance": DensityDistanceRelevance,
    "density_ratio": DensityRatioRelevance,
    "histogram": HistogramBasedRelevance,
    "interpolation": InterpolationWithControlPoints,
    "kdr": KernelDensityRelevance,
    "lds": LabelDistributionSmoothing,
    "denseweight": DenseWeight
}



def relevance_function_factory(
    relevance_type: Literal["density_distance", "density_ratio", "histogram", "interpolation", "kdr", "lds", "denseweight"],
    **kwargs
) -> RelevanceFunctionBase:
    """
    Factory function that creates and fits a relevance function of the specified type.

    STRICT: all parameters must be explicitly provided for both __init__ and fit.

    Args:
        relevance_type: String identifier for the relevance function type
        **kwargs: Parameters needed for initialization and fitting

    Returns:
        A fitted relevance function object ready for evaluation
    """


    if relevance_type not in RELEVANCE_FUNCTIONS:
        available_types = ", ".join(RELEVANCE_FUNCTIONS.keys())
        raise ValueError(f"Unknown relevance type '{relevance_type}'. Available types: {available_types}")

    RelevanceClass = RELEVANCE_FUNCTIONS[relevance_type]

    # Extract parameters strictly required for __init__
    init_params, unused_params = extract_explicit_parameters(RelevanceClass, **kwargs)

    # Instantiate the relevance function
    relevance_func = RelevanceClass(**init_params)

    # Extract parameters strictly required for fit
    fit_params, unused_params = extract_explicit_parameters(relevance_func.fit, **kwargs)

    # Fit the relevance function
    relevance_func.fit(**fit_params)

    return relevance_func
    
