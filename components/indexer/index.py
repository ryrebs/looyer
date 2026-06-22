"""Layer 3b — vector index seam.

Embeds chunked nodes and persists them in Chroma.
The index is the pivot: ingestion writes it, query reads it.
"""

import os

import chromadb
from llama_index.core.schema import BaseNode, MetadataMode


def build_index(nodes: list[BaseNode], embed_model=None) -> chromadb.Collection:
    """Embed nodes and persist them into a Chroma collection.

    Each node becomes one Chroma document:
      - id:        node's unique id
      - embedding: vector from bge-m3
      - document:  the text content (what gets returned at retrieval time)
      - metadata:  source, header_path, etc. carried from chunking
    """
    persist_dir = os.getenv("INDEX_DIR")
    collection_name = os.getenv("COLLECTION_NAME")

    client = chromadb.PersistentClient(path=persist_dir)
    collection = client.get_or_create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )

    texts = [node.get_content(metadata_mode=MetadataMode.EMBED) for node in nodes]
    ids = [node.node_id for node in nodes]
    metadatas = [node.metadata for node in nodes]

    print(f"Embedding {len(nodes)} chunks with {os.getenv('EMBED_MODEL')}...")
    embeddings = embed_model.get_text_embedding_batch(texts, show_progress=False)

    collection.upsert(
        ids=ids,
        embeddings=embeddings,
        documents=texts,
        metadatas=metadatas,
    )

    print(f"Index persisted → {persist_dir}  ({collection.count()} total docs)")
    return collection


def load_index() -> chromadb.Collection:
    """Load an existing persisted Chroma collection (query time)."""
    client = chromadb.PersistentClient(path=os.getenv("INDEX_DIR"))
    return client.get_collection(name=os.getenv("COLLECTION_NAME"))
