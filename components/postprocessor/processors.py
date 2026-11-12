import os
from functools import lru_cache

from llama_index.core.postprocessor import SimilarityPostprocessor
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker


def rerank_score_threshold() -> float:
    """RERANK_SCORE_THRESHOLD from the env (required): raw reranker score below which a chunk is dropped."""
    value = os.getenv("RERANK_SCORE_THRESHOLD", "").strip()
    if not value:
        raise RuntimeError("RERANK_SCORE_THRESHOLD is not set in .env (see env.sample)")
    try:
        return float(value)
    except ValueError as exc:
        raise RuntimeError(f"RERANK_SCORE_THRESHOLD must be a number, got {value!r}") from exc


@lru_cache(maxsize=None)
def _get_reranker(model_path: str, top_n: int) -> FlagEmbeddingReranker:
    """Load the reranker model once per (model, top_n); later calls get the same object back.

    Only the loaded model is cached (in memory, for this process), not any scores: every
    question's chunks are still scored fresh. Saves the ~1 s reload per question when one
    process handles many questions (the evaluation, an API); no change for the one-question CLI.
    """
    return FlagEmbeddingReranker(model=model_path, top_n=top_n)


def rerank(nodes, query: str, top_n=None):
    if top_n is None:
        top_n = 5

    reranker = _get_reranker(os.getenv("RERANKED_LLM_MODEL"), top_n)
    reranked_nodes = reranker.postprocess_nodes(nodes, query_str=query)
    return reranked_nodes


def simpostprocessor(threshold, nodes):
    smpr = SimilarityPostprocessor(similarity_cutoff=threshold)
    return smpr.postprocess_nodes(nodes)
