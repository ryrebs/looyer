"""Smoke test for the reasoning layer (Layer 5).

Uses dummy NodeWithScore objects — no index, no retrieval, no embeddings needed.
Verifies that reason() accepts nodes, converts to EvidenceChunk, and returns
a valid ReasoningResult.
"""

from llama_index.core.schema import NodeWithScore, TextNode

from components.synthesizer.test_completion import reason, ReasoningResult


def _make_node(text: str, source: str, score: float = 0.9) -> NodeWithScore:
    node = TextNode(text=text, metadata={"source": source})
    return NodeWithScore(node=node, score=score)


DUMMY_NODES = [
    _make_node(
        text=(
            "Section 14. No person shall be held to answer for a criminal offense "
            "without due process of law."
        ),
        source="1987_const.html",
        score=0.95,
    ),
    _make_node(
        text=(
            "Section 17. No person shall be compelled to be a witness against himself."
        ),
        source="1987_const.html",
        score=0.88,
    ),
    _make_node(
        text=(
            "Article 1. This Act shall be known as the Civil Code of the Philippines."
        ),
        source="civil_code.pdf",
        score=0.42,
    ),
]


def test_reason_returns_reasoning_result():
    result = reason("What are the rights of an accused person?", DUMMY_NODES)
    assert isinstance(result, ReasoningResult)


def test_reason_has_answer():
    result = reason("What are the rights of an accused person?", DUMMY_NODES)
    assert isinstance(result.answer, str)
    assert len(result.answer) > 0


def test_reason_low_relevance_nodes():
    """Nodes with low relevance — model should indicate evidence is insufficient."""
    nodes = [
        _make_node(
            text="Article 1. This Act shall be known as the Civil Code of the Philippines.",
            source="civil_code.pdf",
            score=0.1,
        )
    ]
    result = reason("What is the penalty for murder?", nodes)
    assert isinstance(result, ReasoningResult)
