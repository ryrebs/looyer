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
    retrieve,
    reason,
    rerank,
    rag_response_syn,
    irac_qa_template,
    irac_refine_template,
)


from pydantic import BaseModel, Field
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.llama_cpp import LlamaCPP
from llama_index.core.schema import NodeWithScore, TextNode


## We load and parse the sources or data
# data = ["assets/civil_code.pdf", "assets/1987_const.html"]
data = ["assets/1987_const.html"]

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


def load_data():
    all_data = []
    for data_path in data:
        if ".html" in data_path:
            use_docling = True
        else:
            use_docling = False
        all_data.append(
            ParseData(
                content=load_document(data_path, use_docling=use_docling),
                path=data_path,
            )
        )
    return all_data


def load_chunked_embed():
    for mk in load_data():
        if ".pdf" in mk.path:
            texts = transform_mk_cv_code(
                mk.content, [_clean_heading_markup, _promote_articles]
            )
        else:
            texts = transform_html_const(mk.content)
        nodes = markdown_chunk_text(texts, mk.path, embed_model=_embed_model, llm=_llm)
        build_index(nodes, embed_model=_embed_model)


def run_rag_pipeline(query: str, index: bool = False) -> str:
    if index:
        load_chunked_embed()

    nodes = retrieve(query, k=20, run_rerank=True, embed_model=_embed_model)
    return rag_response_syn(
        nodes,
        query,
        llm=_llm,
        text_qa_template=irac_qa_template,
        refine_template=irac_refine_template,
    )


if __name__ == "__main__":

    def start():
        question = " ".join(sys.argv[1:]) or "What are the rights of an accused person?"
        answer = run_rag_pipeline(question, index=True)
        print(f"\n{answer}")

    start()

    ## Manual cleanup
    del _llm
    import gc

    gc.collect()
