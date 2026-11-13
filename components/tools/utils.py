"""Shared decorators."""

import functools
import logging

import mlflow

_logger = logging.getLogger(__name__)

_tracing = {"ready": False}  # set by setup_tracing() in components/tracing/config.py


def set_tracing_ready(ready: bool) -> None:
    """True once MLflow tracing is configured for this project (tracking URI + experiment)."""
    _tracing["ready"] = ready


def warn_on_error(default=None):
    """Log a warning and return `default` instead of raising."""

    def decorator(fn):
        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as exc:  # pylint: disable=broad-exception-caught
                _logger.warning("%s failed: %s", fn.__name__, exc)
                return default

        return wrapper

    return decorator


def traced(raise_if_not_setup: bool = False):
    """Trace the function with MLflow if setup_tracing() succeeded.

    Not set up: raise if raise_if_not_setup, else warn and run untraced (nothing is written).
    """

    def decorator(fn):
        traced_fn = mlflow.trace(fn)

        @functools.wraps(fn)
        def wrapper(*args, **kwargs):
            if _tracing["ready"]:
                return traced_fn(*args, **kwargs)
            if raise_if_not_setup:
                raise RuntimeError(f"{fn.__name__}: MLflow tracing is not set up; call setup_tracing() first")
            _logger.warning("%s: MLflow tracing is not set up; running untraced", fn.__name__)
            return fn(*args, **kwargs)

        return wrapper

    return decorator
