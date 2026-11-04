"""Tests for check_index.py (item 4 in todo.md).

No Qdrant needed: the index is a fake list of chunk metadata, and the output files go to a
temporary folder that pytest deletes afterwards.

Run from the repo root:
    pipenv run test components/evaluator/tests -v
"""
import json

import pytest

from components.evaluator import check_index as ci

CONST = "1987_const.html"
CIVIL = "civil_code.pdf"

# The fake index: metadata of three chunks, in the shape index_metadata() returns
INDEX = [
    {"source": CONST, "article": "Article III", "section": "Section 12"},
    {"source": CIVIL, "article": "Article 1318", "section": "Section 3"},  # Civil Code "SECTION n" heading
    {"source": CIVIL, "article": "Article 1352", "section": None},
]
INDEX_INFO = {"collection": "test_collection", "index_points": len(INDEX)}


def entry(entry_id: str, *expected: dict) -> dict:
    """A verified golden entry with only the fields the check reads."""
    return {"id": entry_id, "expected_metadata": list(expected)}


def const(article: str, section: str) -> dict:
    return {"source": CONST, "article": article, "section": section}


def civil(article: str) -> dict:
    return {"source": CIVIL, "article": article}


@pytest.fixture
def index_reads(monkeypatch, tmp_path) -> list:
    """Point check_index at the fake index and a temporary output folder (no Qdrant).

    Returns a list that gets one item each time the index is read, i.e. each time a check really runs.
    """
    reads = []

    def read_fake_index():
        reads.append("read")
        return INDEX

    monkeypatch.setattr(ci, "index_metadata", read_fake_index)
    monkeypatch.setattr(ci, "index_version", lambda: INDEX_INFO)
    monkeypatch.setattr(ci, "INDEX_CHECKS_DIR", tmp_path)
    return reads


def saved_check(folder) -> dict:
    """The one check file saved in the folder."""
    saved_files = list(folder.glob("*.json"))
    assert len(saved_files) == 1
    return json.loads(saved_files[0].read_text(encoding="utf-8"))


# --- check_entries + missing_provisions: one expected provision at a time ---


def test_provision_in_index_is_found():
    entries = ci.check_entries({"direct_lookup": [entry("dl-01", const("Article III", "Section 12"))]}, INDEX)
    assert entries[0]["in_index"] == [True]
    assert ci.missing_provisions(entries) == []


def test_wrong_section_is_missing():
    entries = ci.check_entries({"direct_lookup": [entry("dl-99", const("Article III", "Section 99"))]}, INDEX)
    assert entries[0]["in_index"] == [False]
    assert ci.missing_provisions(entries) == [{"id": "dl-99", "expected": const("Article III", "Section 99")}]


def test_label_must_be_written_like_the_index():
    # "Art. 1352" is the same provision to a human, but not the label the parser wrote
    entries = ci.check_entries({"direct_lookup": [entry("dl-98", civil("Art. 1352"))]}, INDEX)
    assert entries[0]["in_index"] == [False]


def test_civil_code_section_is_ignored():
    # Civil Code entries list source + article only, so the chunk's section doesn't matter
    entries = ci.check_entries({"direct_lookup": [entry("dl-10", civil("Article 1318"))]}, INDEX)
    assert entries[0]["in_index"] == [True]


def test_each_provision_is_checked_separately():
    # A question that needs two provisions, one of them missing: only that one is listed
    entries = ci.check_entries({"inference": [entry("inf-10", civil("Article 1318"), civil("Article 9999"))]}, INDEX)
    assert entries[0]["in_index"] == [True, False]
    assert ci.missing_provisions(entries) == [{"id": "inf-10", "expected": civil("Article 9999")}]


def test_negatives_are_left_out():
    # Negatives expect nothing, so there is nothing to check
    entries = ci.check_entries({"negative": [entry("neg-01")]}, INDEX)
    assert entries == []


# --- check_index: the whole check, with the fake index and a temporary output folder ---


def test_check_passes_when_every_provision_is_in_the_index(index_reads, tmp_path):
    verified = {"direct_lookup": [entry("dl-01", const("Article III", "Section 12")), entry("dl-10", civil("Article 1318"))]}
    assert ci.check_index(verified) is True
    saved = saved_check(tmp_path)
    assert saved["passed"] is True
    assert saved["expected_provisions"] == 2
    assert saved["missing"] == []


def test_check_lists_every_missing_provision_without_stopping(index_reads, tmp_path):
    verified = {
        "direct_lookup": [
            entry("dl-01", const("Article III", "Section 12")),
            entry("dl-99", const("Article III", "Section 99")),
        ],
        "inference": [entry("inf-10", civil("Article 1318"), civil("Art. 1352"))],
        "negative": [entry("neg-01")],
    }
    assert ci.check_index(verified) is False
    saved = saved_check(tmp_path)
    assert saved["expected_provisions"] == 4
    assert saved["missing"] == [
        {"id": "dl-99", "expected": const("Article III", "Section 99")},
        {"id": "inf-10", "expected": civil("Art. 1352")},
    ]
    assert "entries" not in saved  # only what's missing is saved
    assert saved["index"] == INDEX_INFO


# --- skipping: check again only when the answer key or the index changed ---


def test_check_is_skipped_when_nothing_changed(index_reads):
    verified = {"direct_lookup": [entry("dl-99", const("Article III", "Section 99"))]}
    assert ci.check_index(verified) is False
    assert ci.check_index(verified) is False  # skipped, but still returns the saved result
    assert len(index_reads) == 1


def test_check_runs_again_when_the_answer_key_changes(index_reads):
    ci.check_index({"direct_lookup": [entry("dl-99", const("Article III", "Section 99"))]})
    assert ci.check_index({"direct_lookup": [entry("dl-99", const("Article III", "Section 12"))]}) is True
    assert len(index_reads) == 2


def test_check_runs_again_when_the_index_changes(index_reads, monkeypatch):
    verified = {"direct_lookup": [entry("dl-01", const("Article III", "Section 12"))]}
    ci.check_index(verified)
    monkeypatch.setattr(ci, "index_version", lambda: {"collection": "test_collection", "index_points": 999})
    ci.check_index(verified)
    assert len(index_reads) == 2


def test_force_checks_again(index_reads):
    verified = {"direct_lookup": [entry("dl-01", const("Article III", "Section 12"))]}
    ci.check_index(verified)
    ci.check_index(verified, force=True)
    assert len(index_reads) == 2
