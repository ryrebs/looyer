import os
from collections import Counter

import chromadb


def _get_collection():
    client = chromadb.PersistentClient(path=os.getenv("INDEX_DIR"))
    return client.get_collection(name=os.getenv("COLLECTION_NAME"))


# ─── GROUPING KEY ─────────────────────────────────────────────────────────────
# Uniqueness is defined by the full set of metadata fields on a node.
# The key is built dynamically from all present fields — no field names are
# hardcoded. Adding a new extractor/field to the pipeline automatically
# tightens uniqueness without touching this code.
#
# This works because SemanticSplitter propagates the parent node's metadata
# unchanged to all child chunks. Fragments of the same atom therefore share
# identical metadata and collapse to the same key.
# ──────────────────────────────────────────────────────────────────────────────


def _grouping_key(meta: dict) -> tuple:
    """
    Build the grouping key from all metadata fields present on the node.

    Keys are sorted so order is stable regardless of insertion order.
    Any new field added to the pipeline automatically contributes to uniqueness
    without changing this function.

    Two nodes sharing the same key are fragments of the same atom.
    """
    return tuple(sorted(meta.items()))


def audit_current_chunks() -> Counter:
    """
    Return a Counter mapping (source, atom) → node count.

    A count of 1 = the atom arrived as a single node (intact).
    A count > 1 = the atom was split by SemanticSplitter (fragmented).

    Usage:
        from components.audit.audit import audit_current_chunks, inspect_section
        counter = audit_current_chunks()

        # see fragmented atoms only
        fragmented = {k: v for k, v in counter.items() if v > 1}
    """
    collection = _get_collection()
    results = collection.get(include=["metadatas"])
    metas = results["metadatas"]

    if not metas:
        print("Collection is empty — run ingestion first.")
        return Counter()

    counter = Counter()
    for meta in metas:
        counter[_grouping_key(meta)] += 1

    return counter


def inspect_section(key: tuple, show_fragments: int = 10) -> None:
    """
    Print stored chunk text for a specific atom key.

    Get valid keys from audit_current_chunks():
        counter = audit_current_chunks()
        for key, count in counter.items():
            print(key, count)

    Then pass the key directly:
        inspect_section(('assets/civil_code.pdf', '/CHAPTER 1 .../', 'Article 5'))
    """
    collection = _get_collection()
    results = collection.get(include=["documents", "metadatas"])
    docs = results["documents"]
    metas = results["metadatas"]

    matches = []
    for doc, meta in zip(docs, metas):
        if _grouping_key(meta) == key:
            matches.append({"text": doc, "tokens": len(doc) // 4, "metadata": meta})

    if not matches:
        print(f"No nodes found for key: {key}")
        return

    print(f"\n=== Section Inspection ===")
    print(f"Key         : {key}")
    print(f"Fragments   : {len(matches)}")
    print(f"Total tokens: {sum(m['tokens'] for m in matches)} (approx)")
    print()

    for i, match in enumerate(matches, 1):
        print(f"─── Fragment {i} of {len(matches)} ({match['tokens']} tokens) ───")
        print(match["text"])
        if i >= show_fragments:
            break


if __name__ == "__main__":
    """What is the point of this audit?
    1. We need to check for fragmentation - meaning given metadata , are there nodes
    that have the same metadata? If yes, that node were fragmented, and possible lose it's context.

    2. Given fragmentation result we need to check the fragmentation count

    3. We also need to check token per node
    """
    counter = audit_current_chunks()
    fragmented = {k: v for k, v in counter.items() if v > 1}
    print(f"Total atoms  : {len(counter)}")
    print(f"Fragmented   : {len(fragmented)}")
    print(f"Intact       : {len(counter) - len(fragmented)}")
    print()
    print("Fragmented atoms (atom -> count):")
    for key, count in sorted(fragmented.items(), key=lambda x: -x[1])[:20]:
        print(f"  {count}x  {key}")

    # To inspect a specific unit, copy a key from the output above and pass it:
    inspect_section(
        (
            ("book", "Book I: Persons"),
            ("chapter", "Chapter 3"),
            ("header_path", "/"),
            ("is_amended", False),
            ("is_repealed", False),
            ("proviso_flag", False),
            ("source", "assets/civil_code.pdf"),
            ("title", "Title I: Civil Personality"),
        ),
        show_fragments=30,
    )
