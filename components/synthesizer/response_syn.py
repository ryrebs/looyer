"""Layer 5 — response synthesis seam."""

from llama_index.core import get_response_synthesizer
from llama_index.core.response_synthesizers import ResponseMode
from llama_index.core.schema import NodeWithScore


def rag_response_syn(
    nodes: list[NodeWithScore],
    query: str,
    llm=None,
    text_qa_template=None,
    refine_template=None,
) -> str:
    _synthesizer = get_response_synthesizer(
        response_mode=ResponseMode.COMPACT,
        llm=llm,
        text_qa_template=text_qa_template,
        refine_template=refine_template,
    )
    response = _synthesizer.synthesize(query=query, nodes=nodes)
    return response.response
