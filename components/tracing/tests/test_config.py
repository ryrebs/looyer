"""Tests for tracing/config.py: telemetry stays off, and an MLflow failure never crashes the caller.

Run from the repo root:
    pipenv run test components/tracing/tests -v
"""

import pytest
from mlflow.environment_variables import MLFLOW_DISABLE_TELEMETRY

from components.tracing import config


@pytest.fixture(autouse=True)
def temp_store(tmp_path, monkeypatch):
    """Keep test files out of .cache/mlflow."""
    monkeypatch.setattr(config, "MLFLOW_DIR", tmp_path)
    monkeypatch.setattr(config, "ARTIFACTS_DIR", tmp_path / "artifacts")


def test_telemetry_forced_off_even_if_env_turns_it_on(monkeypatch):
    monkeypatch.setenv("MLFLOW_DISABLE_TELEMETRY", "false")
    config.disable_telemetry()
    # MLflow's own reading of the setting; it also turns telemetry off under pytest,
    # so checking for "no telemetry client" would pass even without our code
    assert MLFLOW_DISABLE_TELEMETRY.get() is True


@pytest.mark.parametrize("uri", [None, "nope://nowhere"], ids=["missing", "broken"])
def test_bad_tracking_uri_returns_false_without_raising(monkeypatch, uri):
    if uri is None:
        monkeypatch.delenv("MLFLOW_TRACKING_URI", raising=False)
    else:
        monkeypatch.setenv("MLFLOW_TRACKING_URI", uri)
    assert config.setup_tracing("test-exp") is False
