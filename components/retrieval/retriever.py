"""Layer 4 — retrieval seam.

Uses LlamaIndex's retriever to find the top-k most relevant chunks for a question.
These chunks are the evidence passed to Layer 5 (reasoning).

Retrieve and reason are kept deliberately separate — this layer only finds
evidence, it never interprets it.
"""

import os

import chromadb
from llama_index.core import VectorStoreIndex
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.schema import NodeWithScore, TextNode
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore

from ..reranker.rerank import rerank


_BM25_PERSIST_DIR = os.getenv("BM25_INDEX_DIR", ".cache/bm25")


def _nodes_from_chroma(collection) -> list[TextNode]:
    results = collection.get(include=["documents", "metadatas"])
    return [
        TextNode(text=doc, metadata=meta or {}, id_=id_)
        for doc, meta, id_ in zip(
            results["documents"], results["metadatas"], results["ids"]
        )
    ]


def _get_bm25_retriever(collection, k: int) -> BM25Retriever:
    """BM25Retriever needs to get the whole collection to process similarities,
    we cache the generated index.
    """
    if os.path.exists(_BM25_PERSIST_DIR):
        retriever = BM25Retriever.from_persist_dir(_BM25_PERSIST_DIR)
        retriever.similarity_top_k = k
        return retriever
    nodes = _nodes_from_chroma(collection)
    retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=k)
    retriever.persist(_BM25_PERSIST_DIR)
    return retriever


def _load_retriever(k: int, embed_model: HuggingFaceEmbedding):
    chroma_client = chromadb.PersistentClient(path=os.environ.get("INDEX_DIR"))
    collection = chroma_client.get_collection(name=os.getenv("COLLECTION_NAME"))
    vector_store = ChromaVectorStore(chroma_collection=collection)

    # Dense retriever — semantic similarity via bge-m3 vectors
    dense_retriever = VectorStoreIndex.from_vector_store(
        vector_store, embed_model=embed_model
    ).as_retriever(similarity_top_k=k)

    # Sparse retriever — BM25, built once and persisted
    bm25_retriever = _get_bm25_retriever(collection, k)

    # Merge with Reciprocal Rank Fusion
    return QueryFusionRetriever(
        retrievers=[dense_retriever, bm25_retriever],
        similarity_top_k=k,
        num_queries=1,  # no LLM query expansion, just fuse as-is
        mode="reciprocal_rerank",
        use_async=False,
        llm=None,
    )


def retrieve(
    question: str,
    k: int = 5,
    run_rerank: bool = False,
    embed_model: HuggingFaceEmbedding = None,
) -> list[NodeWithScore]:
    """Return the top-k most relevant chunks for the given question.

    Args:
        question:   the user's question in plain text.
        k:          number of chunks to return. Start with 5, tune after eval.
        run_rerank: whether to rerank results with cross-encoder before returning.

    Returns:
        List of NodeWithScore sorted by relevance (most relevant first).
    """
    retriever = _load_retriever(k, embed_model)
    nodes = retriever.retrieve(question)

    if run_rerank:
        nodes = rerank(nodes, question)

    return nodes
