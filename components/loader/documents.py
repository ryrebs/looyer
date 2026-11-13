"""Metadata per indexed document, from documents.json (keyed by file name)."""

import json
from pathlib import Path

DOCUMENTS_PATH = Path(__file__).with_name("documents.json")


def load_document_meta(path: str) -> dict:
    """The document's metadata (law_name). Raises if the file isn't listed, so nothing is indexed unnamed."""
    name = Path(path).name
    documents = json.loads(DOCUMENTS_PATH.read_text(encoding="utf-8"))
    if name not in documents:
        raise ValueError(f"{name} is not in {DOCUMENTS_PATH}: add its law_name before indexing")
    return documents[name]
