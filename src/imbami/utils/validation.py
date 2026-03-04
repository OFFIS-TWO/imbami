from typing import get_args, Union, Any, Dict
import inspect

def extract_explicit_parameters(callable_obj, **kwargs) -> Dict[str, Any]:
    """
    Returns a dictionary of parameters strictly required by the callable (including defaults).
    Raises ValueError if any parameter is missing and TypeError if types mismatch.
    """
    sig = inspect.signature(callable_obj)
    validated_params = {}

    for name, param in sig.parameters.items():
        if param.kind in (param.VAR_POSITIONAL, param.VAR_KEYWORD):
            continue

        if name not in kwargs:
            raise ValueError(
                f"Missing required explicit parameter '{name}' for {callable_obj.__name__}"
            )

        value = kwargs[name]

        if param.annotation is not inspect.Parameter.empty:
            expected_type = param.annotation

            # Handle union types (Python 3.10+ syntax e.g., int | None)
            types_to_check = get_args(expected_type) or (expected_type,)

            if not any(isinstance(value, t) or (t is float and isinstance(value, int)) for t in types_to_check):
                type_names = ", ".join(t.__name__ if hasattr(t, "__name__") else str(t) for t in types_to_check)
                raise TypeError(
                    f"Parameter '{name}' for {callable_obj.__name__} must be of type "
                    f"{type_names}, got {type(value).__name__}"
                )

        validated_params[name] = value

    return validated_params