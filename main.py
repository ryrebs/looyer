import logging
import os
import sys

## TO BE DEPRECATED (Chroma, replaced by Qdrant): only used by _collection_exists() below
import chromadb
from pydantic import BaseModel, Field

from components import (
    _clean_heading_markup,
    _promote_articles,
    build_durable_storage_context,
    build_embed_model,
    build_index_from_durable_storage,
    build_llm,
    build_qdrant_client,
    check_llm_server,
    check_qdrant,
    clear_bm25_cache,
    delete_document_chunks,
    embedding_dimension,
    irac_qa_template,
    irac_refine_template,
    load_document,
    load_document_meta,
    markdown_chunk_text,
    rag_response_syn,
    rerank,
    rerank_score_threshold,
    retrieve,
    simpostprocessor,
    transform_html_const,
    transform_mk_cv_code,
)
from components.evaluator.evaluate import evaluate
from components.synthesizer.templates import INSUFFICIENT_CONTEXT, REFUSAL_MESSAGE
from components.tools.utils import traced
from components.tracing.config import EXPERIMENT_MAIN, setup_tracing

_logger = logging.getLogger(__name__)

## Embedding model: in-app or Ollama, from .env (EMBED_API_BASE). See components/embedding/config.py.
_embed_model = build_embed_model()


## LLMs run on OpenAI-compatible servers; build_llm(role) only creates an HTTP client.
## Request settings: components/llm/profiles.json. Servers: .env and README "LLM servers".
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
## TO BE DEPRECATED (Chroma, replaced by Qdrant): Chroma's folder; Qdrant doesn't use it
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
    doc_meta = load_document_meta(path)  # first, so an unlisted file fails before parsing
    mk = load_data(path)
    if ".pdf" in mk.path:
        texts = transform_mk_cv_code(mk.content, [_clean_heading_markup, _promote_articles])
    else:
        texts = transform_html_const(mk.content)

    nodes = markdown_chunk_text(
        texts,
        mk.path,
        embed_model=_embed_model,
        doc_meta=doc_meta,
    )

    ## TO BE DEPRECATED (Chroma, replaced by Qdrant)
    ## ChromaDB
    # build_index(nodes, embed_model=_embed_model)

    ## Qdrant
    storage_context = build_durable_storage_context(COLLECTION_NAME, client=_qdrant)
    ## The data is about to change: delete BM25's local copy of the chunks, so the next search
    ## rebuilds it from Qdrant. Before writing, so a crash halfway also leaves no stale copy.
    clear_bm25_cache(storage_context.vector_store)
    ## Re-indexing replaces the document: delete its old chunks (matched on `source`)
    delete_document_chunks(storage_context, mk.path)
    ## The file path and law name are the same on every chunk of a document: noise for search, so
    ## they are kept out of the embedded text (the answer LLM still sees them for citing)
    build_index_from_durable_storage(
        nodes,
        embed_model=_embed_model,
        storage_context=storage_context,
        excluded_embed_metadata_keys=["source", "law_name"],
    )


## TO BE DEPRECATED (Chroma, replaced by Qdrant): unused, see _qdrant_collection_exists()
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


@traced(
    raise_if_not_setup=True
)  # one trace per question; LlamaIndex steps nest under it. Warns if tracing isn't set up
def run_rag_pipeline(query: str) -> str:
    ## Built here, not at import: creating it already contacts Qdrant (checks the collection),
    ## so it must come after check_qdrant(). Reuses this module's Qdrant client.
    storage_context = build_durable_storage_context(COLLECTION_NAME, client=_qdrant)

    ## HyDE only needs a short hypothetical answer, so it has its own role (a small
    ## instruct model) instead of the answer model.
    nodes = retrieve(
        query,
        storage_context,
        k=20,
        embed_model=_embed_model,
        hyde_llm=build_llm("hyde"),
    )
    ## Of the 20 fused chunks, the reranker keeps the 5 the answer LLM sees
    nodes = rerank(nodes, query)
    ## Drop chunks the reranker scored below RERANK_SCORE_THRESHOLD
    nodes = simpostprocessor(rerank_score_threshold(), nodes)
    ## Gate refusal: no chunk passed the threshold, so the answer model isn't called
    if not nodes:
        return REFUSAL_MESSAGE

    answer = rag_response_syn(
        nodes,
        query,
        llm=build_llm("answer"),
        text_qa_template=irac_qa_template,
        refine_template=irac_refine_template,
    )
    ## Model refusal: the reply is only the refusal token
    return REFUSAL_MESSAGE if answer.strip() == INSUFFICIENT_CONTEXT else answer


if __name__ == "__main__":
    ## Status and errors go to logging (stderr); usage text and the answer are the CLI's output (stdout).
    ## Libraries stay at WARNING so per-request INFO lines (e.g. httpx) don't flood the output.
    logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    _logger.setLevel(logging.INFO)
    args = sys.argv[1:]

    if not args:
        print("Usage:")
        print("  python main.py run-embed <path>   # index a document")
        print("  python main.py <question>         # query the index")
        print("  python main.py evaluate           # evaluate against the golden dataset")
        sys.exit(0)
    if args[0] == "run-embed":
        if len(args) < 2:
            print("Usage: python main.py run-embed <path>")
            sys.exit(1)
        check_qdrant(_qdrant)
        _check_embed_model()
        _logger.info("Indexing %s", args[1])
        load_chunked_embed(args[1])
        _logger.info("Indexed %s", args[1])
    elif args[0] == "evaluate":
        evaluate()
        _logger.info("Evaluation done")
    else:
        check_qdrant(_qdrant)
        if not _qdrant_collection_exists():
            _logger.error("No index found. Run 'run-embed' first to build the index.")
            sys.exit(1)
        _check_embed_model()
        check_llm_server("hyde")
        check_llm_server("answer")
        setup_tracing(EXPERIMENT_MAIN)
        question = " ".join(args)
        answer = run_rag_pipeline(question)
        print(f"\n{answer}")
