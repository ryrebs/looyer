"""Layer 4 — retrieval seam.

Uses LlamaIndex's retriever to find the top-k most relevant chunks for a question.
These chunks are the evidence passed to Layer 5 (reasoning).

Retrieve and reason are kept deliberately separate — this layer only finds
evidence, it never interprets it.
"""

import glob
import hashlib
import os
import shutil

## TO BE DEPRECATED (Chroma, replaced by Qdrant): chromadb and ChromaVectorStore are only used by the section at the end
import chromadb
from llama_index.core import StorageContext, VectorStoreIndex
from llama_index.core.indices.query.query_transform.base import HyDEQueryTransform
from llama_index.core.llms import MockLLM
from llama_index.core.retrievers import QueryFusionRetriever, TransformRetriever
from llama_index.core.schema import NodeWithScore, QueryBundle, TextNode
from llama_index.core.vector_stores.utils import metadata_dict_to_node
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.vector_stores.qdrant import QdrantVectorStore

_BM25_PERSIST_DIR = os.getenv("BM25_INDEX_DIR", ".cache/bm25")


class _NoLLM(MockLLM):
    """Stand-in LLM for QueryFusionRetriever, which must never call one here (num_queries=1).

    QueryFusionRetriever treats llm=None as "use Settings.llm", whose default is OpenAI, so it
    needs a real object. A plain MockLLM would return made-up text if it were ever called (e.g.
    after setting num_queries > 1); this one fails loudly instead.
    """

    def complete(self, prompt, formatted=False, **kwargs):
        raise RuntimeError("QueryFusionRetriever tried to call an LLM; pass a real one if num_queries > 1.")

    async def acomplete(self, prompt, formatted=False, **kwargs):
        raise RuntimeError("QueryFusionRetriever tried to call an LLM; pass a real one if num_queries > 1.")


class _CheckedHyDE(HyDEQueryTransform):
    """HyDEQueryTransform that does not trust the LLM's passage.

    The stock transform embeds whatever the LLM returns. An empty passage (e.g. a reasoning
    model that spent max_tokens thinking) is averaged into the query vector without any error.
    Here an empty passage falls back to the question alone, and a cut-off one is reported.
    """

    def _run(self, query_bundle: QueryBundle, metadata: dict) -> QueryBundle:
        query_str = query_bundle.query_str
        response = self._llm.complete(self._hyde_prompt.format(context_str=query_str))
        passage = response.text.strip()
        choices = getattr(response.raw, "choices", None)
        finish_reason = choices[0].finish_reason if choices else None

        if not passage:
            print(f"HyDE: empty passage (finish_reason={finish_reason}); using the question only.")
            return query_bundle
        if finish_reason == "length":
            print("HyDE: passage was cut off at max_tokens; using it anyway.")

        embedding_strs = [passage]
        if self._include_original:
            embedding_strs.extend(query_bundle.embedding_strs)
        return QueryBundle(query_str=query_str, custom_embedding_strs=embedding_strs)


def load_all_nodes(vector_store: QdrantVectorStore) -> list[TextNode]:
    """Every chunk stored in the collection, read page by page (no vectors)."""
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


def _bm25_cache_prefix(vector_store: QdrantVectorStore) -> str:
    """The part of a BM25 cache folder name that identifies the Qdrant server and collection."""
    url = os.getenv("QDRANT_URL", "")
    collection = vector_store.collection_name
    # A short folder name, not a security use: usedforsecurity=False (same output as before)
    source_id = hashlib.sha1(f"{url}|{collection}".encode(), usedforsecurity=False).hexdigest()[:12]
    return os.path.join(_BM25_PERSIST_DIR, f"{collection}-{source_id}-")


def clear_bm25_cache(vector_store: QdrantVectorStore) -> None:
    """Delete this collection's BM25 copies (any point count). Call before changing the data.

    BM25 searches its own copy of the chunks, never Qdrant. A re-index that ends with the same
    point count would otherwise reuse the old copy; after this, the next search rebuilds it.
    """
    for cache_dir in glob.glob(_bm25_cache_prefix(vector_store) + "*"):
        shutil.rmtree(cache_dir)


def _get_bm25_retriever_from_qdrant(vector_store: QdrantVectorStore, k: int) -> BM25Retriever:
    """BM25 over the Qdrant collection's chunks, cached on disk.

    The cache holds its own copy of the chunks, so it is keyed by Qdrant URL, collection and
    point count: switching between Docker and Cloud, or re-indexing, builds a new one instead
    of silently returning chunks from the old data. A re-index with the same point count is
    covered by clear_bm25_cache(), which run-embed calls before writing.
    """
    points = vector_store.client.count(collection_name=vector_store.collection_name, exact=True).count
    cache_dir = _bm25_cache_prefix(vector_store) + str(points)

    if os.path.exists(cache_dir):
        retriever = BM25Retriever.from_persist_dir(cache_dir)
        retriever.similarity_top_k = k
        return retriever

    nodes = load_all_nodes(vector_store)
    retriever = BM25Retriever.from_defaults(nodes=nodes, similarity_top_k=k)
    retriever.persist(cache_dir)
    return retriever


def _get_dense_retriever(storage_context, embed_model, k: int, filters=None):
    dense_retriever = VectorStoreIndex.from_vector_store(
        storage_context.vector_store, embed_model=embed_model
    ).as_retriever(similarity_top_k=k, filters=filters)

    return dense_retriever


def retrieve(
    question: str,
    storage_context: StorageContext,
    k: int = 5,
    embed_model: HuggingFaceEmbedding = None,
    hyde_llm=None,
) -> list[NodeWithScore]:
    """Return the top-k most relevant chunks for the given question.

    Args:
        question:   the user's question in plain text.
        storage_context: the vector store to search, built by the caller
                    (main.py: build_durable_storage_context(COLLECTION_NAME, client=_qdrant)).
        k:         number of chunks to return. Start with 5, tune after eval.
        hyde_llm:   LLM that writes the HyDE hypothetical answer; None skips HyDE.

    Returns:
        List of NodeWithScore sorted by relevance (most relevant first).
    """

    final_retriever = None
    ## TODO: Optimization using metadata filters
    ## E.g. node.metadata["effective_date"] = "2024-01-15"
    ## node.metadata["version"] = "RA-11934-amended-2024"
    ## NOTE: This is just a sample metadata filter
    filters = None
    # MetadataFilters(filters=[
    #    MetadataFilter(key="effective_date", value="2024-01-15", operator="<=")
    # ])
    ## Dense retriever — semantic similarity via bge-m3 vectors, now backed by Qdrant
    dense_retriever = _get_dense_retriever(storage_context, embed_model, k, filters)

    ## Sparse retriever — BM25 still runs from local cache; builds from Qdrant on a cold cache
    sparse_retriever = _get_bm25_retriever_from_qdrant(storage_context.vector_store, k)

    ## Merge with Reciprocal Rank Fusion
    fusion_retriever = QueryFusionRetriever(
        retrievers=[dense_retriever, sparse_retriever],
        similarity_top_k=k,
        num_queries=1,  # no LLM query expansion, just fuse as-is
        mode="reciprocal_rerank",
        use_async=False,
        llm=_NoLLM(),  # not None: None would fall back to Settings.llm (OpenAI by default)
    )

    if hyde_llm is not None:
        ## Hypothetical Document Embeddings. HyDEQueryTransform or creating manually.
        ## Why?:
        ## "When can an accused be released without bail?" — that question phrasing may land far
        ## from the actual statute text in embedding space.
        ## A hypothetical answer like "The accused may be released without bail
        ## when the offense is not punishable by reclusion perpetua..."
        ## lands much closer to the real statute.
        hyde = _CheckedHyDE(llm=hyde_llm, include_original=True)
        final_retriever = TransformRetriever(fusion_retriever, hyde)

    else:
        final_retriever = fusion_retriever

    return final_retriever.retrieve(question)


## TO BE DEPRECATED (Chroma, replaced by Qdrant): everything below is unused. _load_retriever() also
## builds QueryFusionRetriever(llm=None), which needs Settings.llm = None (removed), so it no longer runs.
def _nodes_from_chroma(collection) -> list[TextNode]:
    results = collection.get(include=["documents", "metadatas"])
    return [
        TextNode(text=doc, metadata=meta or {}, id_=id_)
        for doc, meta, id_ in zip(results["documents"], results["metadatas"], results["ids"])
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
    dense_retriever = VectorStoreIndex.from_vector_store(vector_store, embed_model=embed_model).as_retriever(
        similarity_top_k=k
    )

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
