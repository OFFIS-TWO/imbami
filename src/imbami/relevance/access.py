from typing import Literal
from .base_relevance_function import RelevanceFunctionBase
from .density_distance_relevance import DensityDistanceRelevance
from .density_ratio_relevance import DensityRatioRelevance
from .histogram_based_relevance import HistogramBasedRelevance
from .interpolation_with_controlpoints import InterpolationWithControlPoints
from .label_distribution_smoothing import LabelDistributionSmoothing
from .kernel_density_relevance import KernelDensityRelevance




def get_relevance_function(relevance_type: Literal["density_distance", "density_ratio", "histogram", "interpolation", "kde", "lds"],
                            **kwargs) -> RelevanceFunctionBase:
    """
    Factory function that creates and fits a relevance function of the specified type.
    
    Assumes all parameters have been validated at the configuration level.
    
    Args:
        relevance_type: String identifier for the relevance function type
        **kwargs: Parameters needed for initialization and fitting
        
    Returns:
        A fitted relevance function object ready for evaluation
    """
    # Map string identifiers to relevance function classes
    relevance_classes = {
        "density_distance": DensityDistanceRelevance,
        "density_ratio": DensityRatioRelevance,
        "histogram": HistogramBasedRelevance,
        "interpolation": InterpolationWithControlPoints,
        "kde": KernelDensityRelevance,
        "lds": LabelDistributionSmoothing
    }
    
    # Check if relevance type exists.
    if relevance_type not in relevance_classes:
        available_types = ", ".join(relevance_classes.keys())
        raise ValueError(f"Unknown relevance type: '{relevance_type}'. Available types: {available_types}")
    
    # Extract initialization parameters using match-case
    init_params = {}
    match relevance_type:
        case "density_distance" | "density_ratio":
            # These parameters are needed for __init__ but not for fit
            init_params["emp_density_mode"] = kwargs.get("emp_density_mode")
            init_params["domain_density_mode"] = kwargs.get("domain_density_mode")
            
            # DensityDistance-specific parameter
            if relevance_type == "density_distance":
                init_params["centered"] = kwargs.get("centered", True)
        
        # Other relevance types don't need special init parameters
        case _:
            pass
    
    # Create and fit the relevance function
    RelevanceClass = relevance_classes[relevance_type]
    relevance_func = RelevanceClass(**init_params)
    relevance_func.fit(**kwargs)
    
    return relevance_func
    
