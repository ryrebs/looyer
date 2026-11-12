"""Tests for judge.py and the judge fields in the summary. No LLM: the judge is a fake object.

Run from the repo root:
    pipenv run test components/evaluator/tests -v
"""

from types import SimpleNamespace

import pytest

from components.evaluator import judge
from components.evaluator.evaluate import summarize


class FakeJudge:
    """Returns a fixed reply (or raises) instead of calling a model."""

    def __init__(self, reply=None, error=None):
        self.reply, self.error = reply, error

    def chat(self, messages):
        if self.error:
            raise self.error
        return SimpleNamespace(message=SimpleNamespace(content=self.reply))


# --- strict verdict: exactly YES or NO, anything else is an error ---


@pytest.mark.parametrize("reply, verdict", [("YES", "YES"), ("NO", "NO"), ("  YES\n", "YES")])
def test_exact_verdicts_are_accepted(reply, verdict):
    assert judge.parse_verdict(reply) == verdict


@pytest.mark.parametrize(
    "reply",
    [
        "yes",  # lower case
        "YES.",  # punctuation
        "No, although one could say yes to part of it.",  # LlamaIndex would count this as a pass
        "NO. It ignores the eyes of the law.",  # contains "yes" inside "eyes"
        "",
        None,
    ],
)
def test_anything_else_is_not_a_verdict(reply):
    assert judge.parse_verdict(reply) is None


def test_ask_scores_yes_and_no():
    assert judge.ask(FakeJudge("YES"), "prompt") == (1.0, None)
    assert judge.ask(FakeJudge("NO"), "prompt") == (0.0, None)


def test_ask_reports_a_loose_reply_as_an_error_not_a_pass():
    score, error = judge.ask(FakeJudge("Yes, mostly."), "prompt")
    assert score is None
    assert error.startswith("not YES/NO")


def test_ask_reports_a_failed_call_as_an_error():
    score, error = judge.ask(FakeJudge(error=TimeoutError("slow")), "prompt")
    assert score is None
    assert error == "TimeoutError: slow"


# --- summary: refused / faithful / correct are averaged; text fields and judge errors are not ---


def test_summary_averages_judge_fields_and_counts_judge_errors():
    rows = [
        {"refused": 0.0, "refused_by": None, "faithful": 1.0, "correct": 1.0, "judge_errors": {}},
        {"refused": 1.0, "refused_by": "gate", "faithful": None, "correct": 0.0, "judge_errors": {}},
        {
            "refused": 0.0,
            "refused_by": "model",
            "faithful": None,
            "correct": None,
            "judge_errors": {"faithful": "not YES/NO: 'maybe'"},
        },
    ]
    summary = summarize(rows)  # must not try to average the text field refused_by
    assert summary["refused"] == pytest.approx(1 / 3)
    assert summary["faithful"] == 1.0  # None skipped
    assert summary["correct"] == 0.5
    assert summary["judge_errors"] == 1
    assert "refused_by" not in summary
