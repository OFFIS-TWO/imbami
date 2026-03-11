import inspect
from typing import Dict, Any, Callable, get_args, get_origin, Union, Literal
from types import NoneType


def get_callable_parameters(callable_obj: Callable[..., Any]) -> Dict[str, inspect.Parameter]:
    """
    Return all explicit parameters of a callable excluding *args and **kwargs.
    """
    sig = inspect.signature(callable_obj)

    # Skip 'self' if this is a bound method
    parameters = {
        name: param
        for name, param in sig.parameters.items()
        if name != "self" and param.kind not in (param.VAR_POSITIONAL, param.VAR_KEYWORD)
        and not (name == "self" and hasattr(callable_obj, "__self__"))
    }
    return parameters


def validate_parameter_type(
    name: str,
    value: Any,
    annotation: Any,
    callable_name: str
) -> None:
    """
    Validate a single parameter value against its annotation.
    """

    if annotation is inspect.Parameter.empty:
        return

    origin = get_origin(annotation)

    # ---------- Literal ----------
    if origin is Literal:
        allowed_values = get_args(annotation)

        if value not in allowed_values:
            raise TypeError(
                f"Parameter '{name}' for {callable_name} must be one of "
                f"{allowed_values}, got {value}"
            )
        return

    # ---------- Union ----------
    if origin is Union:
        union_types = [
            NoneType if t is type(None) else t
            for t in get_args(annotation)
            if isinstance(t, type) or t is type(None)
        ]

        if union_types and not any(
            isinstance(value, t) or (t is float and isinstance(value, int))
            for t in union_types
        ):
            raise TypeError(
                f"Parameter '{name}' for {callable_name} must be one of "
                f"{[t.__name__ for t in union_types]}, got {type(value).__name__}"
            )
        return

    # ---------- Normal type ----------
    if isinstance(annotation, type):

        if not isinstance(value, annotation):
            raise TypeError(
                f"Parameter '{name}' for {callable_name} must be "
                f"{annotation.__name__}, got {type(value).__name__}"
            )


def extract_explicit_parameters(
    callable_obj: Callable[..., Any],
    **kwargs: Any
) -> tuple[Dict[str, Any], set[str]]:
    """
    Validate and extract explicitly declared parameters for a callable.

    Only considers parameters with fixed names and type annotations.
    Automatically skips:
      - 'self' for bound methods
      - *args and **kwargs

    Performs static validation only; the callable is not executed.

    Parameters
    ----------
    callable_obj : Callable[..., Any]
        Function, method, or class constructor to validate parameters for.
    **kwargs : Any
        Keyword arguments intended for the callable.

    Returns
    -------
    validated_params : Dict[str, Any]
        Dictionary containing parameters that were both provided and valid.
    unused_params : set[str]
        Set of extra kwargs that were not used by the callable.

    Raises
    ------
    ValueError
        If one or more required parameters are missing.
    TypeError
        If a parameter value does not match its annotated type.
    """
    callable_name = getattr(callable_obj, "__name__", str(callable_obj))
  
    parameters = get_callable_parameters(callable_obj=callable_obj)

    validated_params: Dict[str, Any] = {}
    missing_params: list[str] = []

    # First pass: collect missing parameters
    for name, param in parameters.items():
        if name not in kwargs:
            missing_params.append(name)

    if missing_params:
        raise ValueError(
            f"Missing required explicit parameter(s) for {callable_name}: {missing_params}"
        )

    # Second pass: validate types
    for name, param in parameters.items():
        value = kwargs[name]
        validate_parameter_type(name, value, param.annotation, callable_name)
        validated_params[name] = value

    # Log extra kwargs
    unused_params = set(kwargs) - set(parameters)

    return validated_params, unused_params