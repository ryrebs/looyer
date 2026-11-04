"""Embedding model: in-app, or on an Ollama server, chosen in .env.

The same model must embed the chunks (chunking's SemanticSplitter, indexing) and the
questions (retrieval): vectors from different models are not comparable, and Qdrant
only rejects them when the dimension differs.

From .env:
    EMBED_API_BASE    Ollama server URL, e.g. http://127.0.0.1:11434. Empty: bge-m3 runs
                      in-app with OpenVINO.
    EMBED_MODEL_NAME  model name on that server, e.g. bge-m3 (unused in-app)
    EMBED_API_KEY     sent as a Bearer token; leave empty for a local server
"""

import os

from llama_index.core.base.embeddings.base import BaseEmbedding

HF_CACHE_DIR = os.environ.get("HF_CACHE_DIR", "./.cache/hf")
OPENVINO_MODEL_DIR = os.path.join(HF_CACHE_DIR, "bge-m3-openvino")


def _load_openvino_bge_m3() -> BaseEmbedding:
    from llama_index.embeddings.huggingface import HuggingFaceEmbedding

    ## The HF repo has no OpenVINO files, so sentence-transformers re-exports bge-m3
    ## on every load. Export once, save it locally, and load the saved copy after that.
    if not os.path.isfile(os.path.join(OPENVINO_MODEL_DIR, "openvino", "openvino_model.xml")):
        from sentence_transformers import SentenceTransformer

        SentenceTransformer(
            "BAAI/bge-m3", backend="openvino", cache_folder=HF_CACHE_DIR
        ).save(OPENVINO_MODEL_DIR)
    return HuggingFaceEmbedding(model_name=OPENVINO_MODEL_DIR, backend="openvino")


def build_embed_model() -> BaseEmbedding:
    """Build the embedding model from .env: Ollama if EMBED_API_BASE is set, else in-app."""
    api_base = os.getenv("EMBED_API_BASE")
    if not api_base:
        return _load_openvino_bge_m3()

    from llama_index.embeddings.ollama import OllamaEmbedding

    model_name = os.getenv("EMBED_MODEL_NAME")
    if not model_name:
        raise RuntimeError("EMBED_MODEL_NAME is not set. See env.sample.")
    api_key = os.getenv("EMBED_API_KEY")
    return OllamaEmbedding(
        model_name=model_name,
        base_url=api_base,
        client_kwargs={"headers": {"Authorization": f"Bearer {api_key}"}} if api_key else None,
    )


def embedding_dimension(embed_model: BaseEmbedding) -> int:
    """Embed one word and return the vector size; fails early if the server is unreachable."""
    try:
        return len(embed_model.get_text_embedding("ping"))
    except Exception as exc:
        raise RuntimeError(
            f"Embedding model is not usable (EMBED_API_BASE={os.getenv('EMBED_API_BASE') or 'in-app'}, "
            f"EMBED_MODEL_NAME={os.getenv('EMBED_MODEL_NAME')}). Is the server running and the model pulled?"
        ) from exc
