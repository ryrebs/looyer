"""Item 4 (todo.md): check the answer key against the index.

Every provision a verified golden entry expects must exist on at least one indexed chunk.
If it doesn't, that question would score 0 because of the parser or the answer key, not retrieval.
The check goes through every entry and lists all missing provisions; it doesn't stop at the first.
It is skipped when the answer key and the index are the same as in a saved check.

Run on its own (e.g. after re-indexing), from the repo root:
    pipenv run python -m components.evaluator.check_index            # skipped if nothing changed
    pipenv run python -m components.evaluator.check_index --force    # check anyway
"""

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from components import build_durable_storage_context, build_qdrant_client, load_all_nodes
from components.evaluator.evaluate import (
    GOLDEN_DATASET_PATH,
    found,
    get_gds,
    hash_answer_key,
    sha256_of,
    source_name,
)

INDEX_CHECKS_DIR = Path("components/evaluator/index_checks")  # one JSON file per check


def load_verified() -> dict:
    """The golden set's verified entries per group (only those are scored)."""
    with open(GOLDEN_DATASET_PATH, encoding="utf-8") as f:
        return get_gds(json.load(f))


def get_index_version() -> dict:
    """Which index is checked: the collection and its number of chunks. Cheap: reads no chunks."""
    collection = os.getenv("COLLECTION_NAME")
    return {"collection": collection, "index_points": build_qdrant_client().get_collection(collection).points_count}


def create_fingerprint(answer_key: str, index_info: dict) -> str:
    """Same answer key + same index = same result, so no need to check again.

    A re-index that keeps the same number of chunks isn't detected: use --force after re-indexing.
    """
    return sha256_of({"answer_key_sha256": answer_key, "index": index_info})


def find_check(fingerprint: str) -> Path | None:
    """The newest saved check with this fingerprint, if any."""
    for path in sorted(INDEX_CHECKS_DIR.glob("*.json"), reverse=True):
        with open(path, encoding="utf-8") as f:
            if json.load(f).get("fingerprint") == fingerprint:
                return path
    return None


def index_metadata() -> list[dict]:
    """Metadata of every indexed chunk, same fields as get_metadata() in evaluate.py (source = file name)."""
    vector_store = build_durable_storage_context(
        os.getenv("COLLECTION_NAME"), client=build_qdrant_client()
    ).vector_store
    return [
        {
            "source": source_name(n.metadata.get("source")),
            "article": n.metadata.get("article"),
            "section": n.metadata.get("section"),
        }
        for n in load_all_nodes(vector_store)
    ]


def check_entries(verified: dict, index: list[dict]) -> list[dict]:
    """Per verified entry: is each expected provision on at least one indexed chunk?

    Same rule as the scoring (found()), so the check and the scoring can't disagree.
    Negatives expect nothing, so they have nothing to check and are left out.
    """
    return [
        {"id": d["id"], "expected_metadata": d["expected_metadata"], "in_index": found(d["expected_metadata"], index)}
        for entries in verified.values()
        for d in entries
        if d["expected_metadata"]
    ]


def missing_provisions(entries: list[dict]) -> list[dict]:
    """Expected provisions that no indexed chunk has."""
    return [
        {"id": e["id"], "expected": expected}
        for e in entries
        for expected, in_index in zip(e["expected_metadata"], e["in_index"])
        if not in_index
    ]


def write_check(check: dict, checked_at: datetime) -> Path:
    """Save the check as its own file, e.g. index_checks/2026-11-04T10-00-00Z.json."""
    INDEX_CHECKS_DIR.mkdir(parents=True, exist_ok=True)
    path = INDEX_CHECKS_DIR / f"{checked_at:%Y-%m-%dT%H-%M-%SZ}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(check, f, ensure_ascii=False, indent=2)
        f.write("\n")
    return path


def print_check(check: dict) -> None:
    """Print how many expected provisions are in the index, and each missing one."""
    total = check["expected_provisions"]
    missing = check["missing"]
    if missing:
        print(f"Index check FAILED: {len(missing)} of {total} expected provisions are not in the index:")
        for m in missing:
            print(f"  {m['id']}: {m['expected']}")
    else:
        print(f"Index check passed: {total}/{total} expected provisions are in the index.")


def check_index(verified: dict, force: bool = False) -> bool:
    """Check the verified entries against the index, save and print the result. True if nothing is missing.

    force=False: skip when a saved check has the same answer key and index, and return its result.
    """
    answer_key = hash_answer_key(verified)
    index_info = get_index_version()
    fingerprint = create_fingerprint(answer_key, index_info)

    previous = None if force else find_check(fingerprint)
    if previous:
        with open(previous, encoding="utf-8") as f:
            previous_check = json.load(f)
        print(f"Index check skipped: same answer key and index as {previous}.")
        print_check(previous_check)
        return previous_check["passed"]

    checked_at = datetime.now(timezone.utc)
    entries = check_entries(verified, index_metadata())
    missing = missing_provisions(entries)
    check = {
        "checked_at": checked_at.isoformat(timespec="seconds"),
        "fingerprint": fingerprint,
        "answer_key_sha256": answer_key,
        "index": index_info,
        "passed": not missing,
        "expected_provisions": sum(len(e["in_index"]) for e in entries),  # how many were checked
        "missing": missing,  # the ones not in the index
    }
    path = write_check(check, checked_at)
    print_check(check)
    print(f"Saved: {path}")
    return check["passed"]


if __name__ == "__main__":
    ## Running reindex and verifying
    ## pipenv run python -c "import os; from components import build_qdrant_client; print(build_qdrant_client().count(os.getenv('COLLECTION_NAME')).count)"
    ## pipenv run python main.py run-embed assets/1987_const.html
    ## pipenv run python -m components.evaluator.check_index

    passed = check_index(load_verified(), force="--force" in sys.argv)
    sys.exit(0 if passed else 1)  # set after the full report: tells scripts whether something is missing
