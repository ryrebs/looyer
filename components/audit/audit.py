import os
from collections import Counter

import chromadb
from qdrant_client import QdrantClient
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.core.vector_stores.utils import metadata_dict_to_node



def _get_collection():
    client = chromadb.PersistentClient(path=os.getenv("INDEX_DIR"))
    return client.get_collection(name=os.getenv("COLLECTION_NAME"))

def _get_collection_from_qdrant():
    qclient = QdrantClient(
        url=os.getenv("QDRANT_URL", "http://127.0.0.1:6333"),
        
    )
    vector_store = QdrantVectorStore(
        client=qclient, collection_name=os.getenv("COLLECTION_NAME")
    )
    nodes = []
    offset = None
    while True:
        records, offset = vector_store.client.scroll(
            collection_name=vector_store.collection_name,
            limit=256,
            offset=offset,
            with_payload=True,
        )
        nodes.extend(metadata_dict_to_node(record.payload) for record in records)
        if offset is None:
            break
    return nodes



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
    nodes = _get_collection_from_qdrant()

    if not nodes:
        print("Collection is empty — run ingestion first.")
        return Counter()

    counter = Counter()
    for node in nodes:
        counter[_grouping_key(node.metadata)] += 1

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
    nodes = _get_collection_from_qdrant()

    matches = []
    for node in nodes:
        if _grouping_key(node.metadata) == key:
            doc = node.get_content()
            matches.append({"text": doc, "tokens": len(doc) // 4, "metadata": node.metadata})

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


    ## Sample output:
    # Total atoms  : 3766
    # Fragmented   : 335
    # Intact       : 3431

    # Fragmented atoms (atom -> count):
    # 20x  (('header_path', '/CHAPTER 1 General Provisions/'), ('source', './assets/civil_code.pdf'))
    # 20x  (('header_path', '/SECTION 2/'), ('is_amended', False), ('is_repealed', False), ('proviso_flag', False), ('source', './assets/civil_code.pdf'))
    # 16x  (('header_path', '/SECTION 2/'), ('source', './assets/civil_code.pdf'))
    # 15x  (('header_path', '/Limited Partnership (n)/'), ('is_amended', False), ('is_repealed', False), ('proviso_flag', False), ('source', './assets/civil_code.pdf'))
    # 12x  (('header_path', '/CONSTITUTION OF THE REPUBLIC OF THE PHILIPPINES/ARTICLE VII — EXECUTIVE DEPARTMENT/'), ('is_amended', False), ('is_repealed', False), ('proviso_flag', False), ('source', './assets/1987_const.html'))
    # 12x  (('header_path', '/Obligations of the Agent/'), ('is_amended', False), ('is_repealed', False), ('proviso_flag', False), ('source', './assets/civil_code.pdf'))
    # 12x  (('header_path', '/CHAPTER 4/'), ('source', './assets/civil_code.pdf'))
    # 10x  (('header_path', '/CHAPTER 3 Prescription of Actions/'), ('source', './assets/civil_code.pdf'))
    # 10x  (('chapter', 'Chapter 3'), ('header_path', '/'), ('is_amended', False), ('is_repealed', False), ('proviso_flag', False), ('source', './assets/civil_code.pdf'))
    # 10x  (('header_path', '/CONSTITUTION OF THE REPUBLIC OF THE PHILIPPINES/ARTICLE VI — THE LEGISLATIVE DEPARTMENT/'), ('is_amended', False), ('is_repealed', False), ('proviso_flag', False), ('source', './assets/1987_const.html'))
    # 10x  (('header_path', '/SECTION 1 Consent/'), ('source', './assets/civil_code.pdf'))
    # 10x  (('header_path', '/SECTION 1 Obligations of the Partners Among Themselves/'), ('source', './assets/civil_code.pdf'))
    # 10x  (('article_number', 'Section 3'), ('header_path', '/'), ('proviso_flag', False), ('section_header', 'SECTION 3'), ('source', './assets/civil_code.pdf'))
    # 10x  (('header_path', '/CONSTITUTION OF THE REPUBLIC OF THE PHILIPPINES/ARTICLE XIII — SOCIAL JUSTICE AND HUMAN RIGHTS/'), ('is_amended', False), ('is_repealed', False), ('proviso_flag', False), ('source', './assets/1987_const.html'))
    # 10x  (('article_number', 'Section 2'), ('header_path', '/'), ('proviso_flag', False), ('section_header', 'SECTION 2'), ('source', './assets/civil_code.pdf'))
    # 10x  (('header_path', '/Obligations of the Partners with Regard to Third Persons/'), ('is_amended', False), ('is_repealed', False), ('proviso_flag', False), ('source', './assets/civil_code.pdf'))
    # 10x  (('header_path', '/Article 1. This Act shall be known as the "Civil Code of the Philippines." (n)/'), ('is_amended', False), ('is_repealed', False), ('proviso_flag', False), ('source', './assets/civil_code.pdf'))
    # 9x  (('header_path', '/SECTION 1 Obligations of the Partners Among Themselves/'), ('is_amended', False), ('is_repealed', False), ('proviso_flag', False), ('source', './assets/civil_code.pdf'))
    # 9x  (('header_path', '/CHAPTER 2 Pledge/'), ('is_amended', False), ('is_repealed', False), ('proviso_flag', False), ('source', './assets/civil_code.pdf'))
    # 9x  (('header_path', '/CHAPTER 1 General Provisions/'), ('is_amended', False), ('is_repealed', False), ('proviso_flag', False), ('source', './assets/civil_code.pdf'))


    ## To inspect a specific unit, copy a key from the output above and pass it:
    inspect_section(
        (
            ('header_path', '/CHAPTER 1 General Provisions/'), ('is_amended', False), ('is_repealed', False), ('proviso_flag', False), ('source', './assets/civil_code.pdf')
        ),
        show_fragments=30,
    )
