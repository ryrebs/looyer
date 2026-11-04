import os
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker


def rerank(nodes, query: str, top_n=None):
    if top_n is None:
        top_n = 5
        
    reranker = FlagEmbeddingReranker(model=os.getenv("RERANKED_LLM_MODEL"), top_n=top_n)
    reranked_nodes = reranker.postprocess_nodes(nodes, query_str=query)
    return reranked_nodes
