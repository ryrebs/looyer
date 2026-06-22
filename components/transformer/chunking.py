"""Processing & chunking pipeline."""

import os

from llama_index.core import Document
from llama_index.core.node_parser import MarkdownNodeParser, SemanticSplitterNodeParser
from llama_index.core.schema import BaseNode
from llama_index.core.ingestion import IngestionPipeline, IngestionCache

from .extractors import LegalMetadataExtractor


def _to_document(text: str, source: str, doc_meta: dict = None) -> Document:
    meta = {"source": source}
    if doc_meta:
        meta.update(doc_meta)
    return Document(text=text, metadata=meta)


def get_pipeline_cache(file_name: str):
    file_name_new = file_name.replace(".", "_")
    cache_dir_name = os.path.join(os.environ.get("PIPELINE_CACHE_DIR"), file_name_new)

    if os.path.exists(cache_dir_name):
        return (
            IngestionCache.from_persist_path(cache_dir_name),
            None,
        )

    return IngestionCache(), cache_dir_name


def markdown_chunk_text(
    text: str, source: str, embed_model=None, llm=None, doc_meta: dict = None
) -> list[BaseNode]:
    """Chunk document using markdown structure and semantic splitting."""

    # Built-in LlamaIndex transformations
    _markdown_parser = MarkdownNodeParser()
    _semantic_splitter = SemanticSplitterNodeParser(
        buffer_size=1, breakpoint_percentile_threshold=95, embed_model=embed_model
    )

    # Custom extractor
    _custom_extractors = [LegalMetadataExtractor(llm=llm)]

    document = _to_document(text, source, doc_meta=doc_meta)
    transformations = [_markdown_parser, _semantic_splitter] + _custom_extractors

    file_name = source.split("/")[-1]
    cache, cache_dir = get_pipeline_cache(file_name)

    pipeline = IngestionPipeline(transformations=transformations, cache=cache)
    nodes = pipeline.run(documents=[document], show_progress=True)

    if cache_dir:
        cache.persist(cache_dir)

    return nodes
