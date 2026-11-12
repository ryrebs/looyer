"""Layer 3b — vector index seam.

Embeds chunked nodes and persists them in Chroma.
The index is the pivot: ingestion writes it, query reads it.
"""

import os

## TO BE DEPRECATED (Chroma, replaced by Qdrant): chromadb is only used by build_index() and load_index()
import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.schema import BaseNode, MetadataMode
from llama_index.vector_stores.qdrant import QdrantVectorStore

from ..vector_store.config import build_qdrant_client


def build_durable_storage_context(collection_name: str, client=None):
    ## Qdrant scalable and production ready index storage; local Docker or Cloud from .env
    vector_store = QdrantVectorStore(client=client or build_qdrant_client(), collection_name=collection_name)

    return StorageContext.from_defaults(
        vector_store=vector_store,
    )


def build_index_from_durable_storage(
    nodes: list[BaseNode],
    embed_model=None,
    storage_context=None,
    excluded_embed_metadata_keys=None,
):
    ## We want to exclude unnecessary metadata keys for embedding such as file name, file type etc..
    if excluded_embed_metadata_keys:
        for node in nodes:
            node.excluded_embed_metadata_keys.extend(excluded_embed_metadata_keys)

    ## Automatically embeds the nodes and store it
    index = VectorStoreIndex(
        nodes,
        storage_context=storage_context,
        embed_model=embed_model,
        show_progress=True,
    )
    return index


## TO BE DEPRECATED (Chroma, replaced by Qdrant): unused, Qdrant indexing is build_index_from_durable_storage()
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


## TO BE DEPRECATED (Chroma, replaced by Qdrant): unused
def load_index() -> chromadb.Collection:
    """Load an existing persisted Chroma collection (query time)."""
    client = chromadb.PersistentClient(path=os.getenv("INDEX_DIR"))
    return client.get_collection(name=os.getenv("COLLECTION_NAME"))
