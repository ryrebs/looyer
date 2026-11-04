import sys
import os

from llama_index.core import Settings

Settings.llm = None

from components import (
    load_document,
    markdown_chunk_text,
    transform_mk_cv_code,
    transform_html_const,
    _clean_heading_markup,
    _promote_articles,
    build_index,
    build_durable_storage_context,
    build_index_from_durable_storage,
    retrieve,
    rerank,
    rag_response_syn,
    irac_qa_template,
    irac_refine_template,
    build_llm,
    check_llm_server,
    build_embed_model,
    embedding_dimension,
    build_qdrant_client,
    check_qdrant,
    evaluate
)


import chromadb
from pydantic import BaseModel, Field
from llama_index.core.schema import NodeWithScore, TextNode


## Embedding model: in-app or Ollama, from .env (EMBED_API_BASE). See components/embedding/config.py.
_embed_model = build_embed_model()


## LLMs run on OpenAI-compatible servers; build_llm(role) only creates an HTTP client.
## Request settings: components/llm/profiles.json. Servers: .env and README "LLM servers".
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
INDEX_DIR = os.getenv("INDEX_DIR")
## Qdrant: local Docker or Cloud, from .env (QDRANT_URL, QDRANT_API_KEY)
_qdrant = build_qdrant_client()


class ParseData(BaseModel):
    content: str = Field(default="", description="Parsed data")
    path: str = Field(default="", description="File path")


def load_data(path: str) -> ParseData:
    use_docling = ".html" in path
    return ParseData(
        content=load_document(path, use_docling=use_docling),
        path=path,
    )


def load_chunked_embed(path: str):
    mk = load_data(path)
    if ".pdf" in mk.path:
        texts = transform_mk_cv_code(
            mk.content, [_clean_heading_markup, _promote_articles]
        )
    else:
        texts = transform_html_const(mk.content)

    nodes = markdown_chunk_text(
        texts,
        mk.path,
        embed_model=_embed_model,
    )

    ## ChromaDB
    # build_index(nodes, embed_model=_embed_model)

    ## Qdrant
    storage_context = build_durable_storage_context(COLLECTION_NAME, client=_qdrant)
    ## The file path is the same on every chunk of a document: noise for search, so it is
    ## kept out of the embedded text (the answer LLM still sees it for citing)
    build_index_from_durable_storage(
        nodes,
        embed_model=_embed_model,
        storage_context=storage_context,
        excluded_embed_metadata_keys=["source"],
    )


def _collection_exists() -> bool:
    try:
        client = chromadb.PersistentClient(path=INDEX_DIR)
        client.get_collection(name=COLLECTION_NAME)
        return True
    except Exception:
        return False


def _qdrant_collection_exists() -> bool:
    return _qdrant.collection_exists(COLLECTION_NAME)


def _check_embed_model() -> None:
    ## Qdrant only rejects vectors of the wrong size, so check the model against the index first
    dim = embedding_dimension(_embed_model)
    if _qdrant.collection_exists(COLLECTION_NAME):
        index_dim = _qdrant.get_collection(COLLECTION_NAME).config.params.vectors.size
        if dim != index_dim:
            raise RuntimeError(
                f"Embedding model returns {dim}-dim vectors but collection '{COLLECTION_NAME}' "
                f"holds {index_dim}-dim vectors. Use the model the index was built with, or re-index."
            )


def run_rag_pipeline(query: str) -> str:
    ## HyDE only needs a short hypothetical answer, so it has its own role (a small
    ## instruct model) instead of the answer model.
    nodes = retrieve(
        query, k=20, run_rerank=True, embed_model=_embed_model, hyde_llm=build_llm("hyde")
    )
    if not nodes:
        print("No relevant documents found for your query.")
        sys.exit(1)

    return rag_response_syn(
        nodes,
        query,
        llm=build_llm("answer"),
        text_qa_template=irac_qa_template,
        refine_template=irac_refine_template,
    )


if __name__ == "__main__":
    args = sys.argv[1:]

    if not args:
        print("Usage:")
        print("  python main.py run-embed <path>   # index a document")
        print("  python main.py <question>         # query the index")
        print("  python main.py evaluate           # evaluate the goldent texts")
        sys.exit(0)
    if args[0] == "run-embed":
        if len(args) < 2:
            print("Usage: python main.py run-embed <path>")
            sys.exit(1)
        check_qdrant(_qdrant)
        _check_embed_model()
        print(f"Building index for {args[1]}...")
        load_chunked_embed(args[1])
        print("Done.")
    if args[0] == "evaluate":
        evaluate()
        print("Done.")
    else:
        check_qdrant(_qdrant)
        if not _qdrant_collection_exists():
            print("No index found. Run with 'run-embed' first to build the index.")
            sys.exit(1)
        _check_embed_model()
        check_llm_server("hyde")
        check_llm_server("answer")
        question = " ".join(args)
        answer = run_rag_pipeline(question)
        print(f"\n{answer}")
