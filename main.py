import sys
import os
import gc

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
    retrieve,
    rerank,
    rag_response_syn,
    irac_qa_template,
    irac_refine_template,
)


import chromadb
from pydantic import BaseModel, Field
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.core.schema import NodeWithScore, TextNode


## Load models
_embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-m3",
    backend="openvino",
    model_kwargs={
        "cache_folder": os.environ.get("HF_CACHE_DIR"),
    },
)

_llm = LlamaCPP(
    verbose=False,
    model_path=os.getenv("LLM_MODEL"),
    temperature=0.1,
    max_new_tokens=1024,
    context_window=8192,
    model_kwargs={"n_threads": os.cpu_count() - 2},
)


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
    nodes = markdown_chunk_text(texts, mk.path, embed_model=_embed_model, llm=_llm)
    build_index(nodes, embed_model=_embed_model)


def _collection_exists() -> bool:
    try:
        client = chromadb.PersistentClient(path=os.getenv("INDEX_DIR"))
        client.get_collection(name=os.getenv("COLLECTION_NAME"))
        return True
    except Exception:
        return False


def run_rag_pipeline(query: str) -> str:
    nodes = retrieve(query, k=20, run_rerank=True, embed_model=_embed_model)
    if not nodes:
        print("No relevant documents found for your query.")
        sys.exit(1)

    return rag_response_syn(
        nodes,
        query,
        llm=_llm,
        text_qa_template=irac_qa_template,
        refine_template=irac_refine_template,
    )


def cleanup():
    global _llm
    del _llm
    gc.collect()


if __name__ == "__main__":
    args = sys.argv[1:]

    if not args:
        print("Usage:")
        print("  python main.py run-embed <path>   # index a document")
        print("  python main.py <question>         # query the index")
        cleanup()
        sys.exit(0)
    if args[0] == "run-embed":
        if len(args) < 2:
            print("Usage: python main.py run-embed <path>")
            cleanup()
            sys.exit(1)
        print(f"Building index for {args[1]}...")
        load_chunked_embed(args[1])
        cleanup()
        print("Done.")
    else:
        if not _collection_exists():
            print("No index found. Run with 'run-embed' first to build the index.")
            sys.exit(1)
        question = " ".join(args)
        answer = run_rag_pipeline(question)
        cleanup()
        print(f"\n{answer}")
