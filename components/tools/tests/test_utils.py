"""Tests for tools/utils.py. No MLflow store: mlflow.trace is replaced by a fake.

Run from the repo root:
    pipenv run test components/tools/tests -v
"""

import logging

import pytest

from components.tools import utils


@pytest.fixture(autouse=True)
def tracing_not_ready():
    utils.set_tracing_ready(False)
    yield
    utils.set_tracing_ready(False)


@pytest.fixture
def fake_mlflow_trace(monkeypatch):
    """Replace mlflow.trace; returns the names of functions that went through it."""
    calls = []

    def fake_trace(fn):
        def wrapper(*args, **kwargs):
            calls.append(fn.__name__)
            return fn(*args, **kwargs)

        return wrapper

    monkeypatch.setattr(utils.mlflow, "trace", fake_trace)
    return calls


# --- warn_on_error ---


def test_warn_on_error_returns_the_result_when_nothing_fails():
    @utils.warn_on_error(default=False)
    def ok():
        return True

    assert ok() is True


def test_warn_on_error_returns_default_and_warns_instead_of_raising(caplog):
    @utils.warn_on_error(default=False)
    def broken():
        raise ValueError("boom")

    with caplog.at_level(logging.WARNING):
        assert broken() is False
    assert "broken failed: boom" in caplog.text


# --- traced ---


def test_traced_uses_mlflow_when_tracing_is_ready(fake_mlflow_trace):
    @utils.traced()
    def answer(q):
        return f"answer to {q}"

    utils.set_tracing_ready(True)
    assert answer("bail") == "answer to bail"
    assert fake_mlflow_trace == ["answer"]


def test_traced_warns_and_runs_untraced_when_not_set_up(fake_mlflow_trace, caplog):
    @utils.traced()
    def answer(q):
        return f"answer to {q}"

    with caplog.at_level(logging.WARNING):
        assert answer("bail") == "answer to bail"
    assert fake_mlflow_trace == []  # nothing written to MLflow
    assert "answer: MLflow tracing is not set up" in caplog.text


def test_traced_raises_when_not_set_up_and_asked_to(fake_mlflow_trace):
    @utils.traced(raise_if_not_setup=True)
    def answer(q):
        return f"answer to {q}"

    with pytest.raises(RuntimeError, match="MLflow tracing is not set up"):
        answer("bail")
    assert fake_mlflow_trace == []
