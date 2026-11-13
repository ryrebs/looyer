"""MLflow tracing setup"""

import logging
import os
from pathlib import Path

import mlflow
import mlflow.llama_index
import mlflow.tracing
from mlflow.telemetry import set_telemetry_client

from ..tools.utils import set_tracing_ready, warn_on_error

_logger = logging.getLogger(__name__)

MLFLOW_DIR = Path(".cache/mlflow")  # holds mlflow.db (MLFLOW_TRACKING_URI) and artifacts
ARTIFACTS_DIR = MLFLOW_DIR / "artifacts"

EXPERIMENT_MAIN = "looyer-main"  # questions from main.py
EXPERIMENT_EVAL = "looyer-eval"  # evaluation runs


def disable_telemetry() -> None:
    """Turn MLflow telemetry off for this process, even if .env lacks MLFLOW_DISABLE_TELEMETRY."""
    os.environ["MLFLOW_DISABLE_TELEMETRY"] = "true"
    set_telemetry_client()  # re-reads the setting; no client means nothing is sent


@warn_on_error(default=False)
def _tracing_off(reason: str) -> bool:
    """Warn and turn tracing off, so @mlflow.trace doesn't write to MLflow's default location."""
    set_tracing_ready(False)
    _logger.warning("MLflow tracing off: %s", reason)
    mlflow.tracing.disable()
    return False


def setup_tracing(experiment: str) -> bool:
    """Trace every LlamaIndex step to MLflow under `experiment`.

    Returns False with a warning if MLflow can't start; the pipeline then runs untraced.
    """
    disable_telemetry()
    uri = os.getenv("MLFLOW_TRACKING_URI")
    if not uri:
        return _tracing_off("MLFLOW_TRACKING_URI is not set")
    try:
        MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
        mlflow.set_tracking_uri(uri)
        if mlflow.get_experiment_by_name(experiment) is None:
            mlflow.create_experiment(experiment, artifact_location=ARTIFACTS_DIR.resolve().as_uri())
        mlflow.set_experiment(experiment)
        mlflow.tracing.enable()
        mlflow.llama_index.autolog()
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return _tracing_off(str(exc))
    set_tracing_ready(True)
    return True
