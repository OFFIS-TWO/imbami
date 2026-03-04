from typing import Literal, Any, Type, Dict

def _validate_parameters(
    required_params: Dict[str, Type],
    provided_params: Dict[str, Any],
    identifier: str
) -> Dict[str, Any]:
    """
    Validates that all required parameters are present with correct types.
    
    Args:
        required_params: Dictionary mapping parameter names to expected types.
        provided_params: Dictionary of parameters provided by the user.
        identifier: Identifier for error messages.
        
    Returns:
        Dictionary of validated parameters.
        
    Raises:
        ValueError: If required parameters are missing.
        TypeError: If parameters have incorrect types.
    """
    # Validate ALL parameters are present
    missing_params = [param for param in required_params.keys() if param not in provided_params]
    if missing_params:
        raise ValueError(f"Missing required parameters for {identifier}: {', '.join(missing_params)}")
    
    # Validate parameter types
    for param, param_type in required_params.items():
        value = provided_params[param]
        # Allow ints where floats are expected
        if not isinstance(value, param_type) and not (param_type is float and isinstance(value, int)):
            raise TypeError(
                f"Parameter '{param}' for {identifier} must be of type {param_type.__name__}, "
                f"got {type(value).__name__}"
            )
    
    # Check for unexpected parameters
    unexpected_params = set(provided_params.keys()) - set(required_params.keys())
    if unexpected_params:
        raise ValueError(f"Unexpected parameters for {identifier}: {', '.join(unexpected_params)}")
    
    return {k: v for k, v in provided_params.items() if k in required_params}