"""LLM judge (todo items 19-22): two YES/NO checks on a generated answer.

- faithfulness: is the answer supported by the chunks the answer model was given?
- correctness:  does the answer reach the same legal result as reference_answer (the answer key)?

The judge must reply exactly YES or NO. Anything else is a judge error, never a pass
(LlamaIndex's evaluators pass any reply that merely contains "yes"). Prompts are plain chat
messages with no model-specific wording. Use a different model from the answer role.
"""

from llama_index.core.llms import ChatMessage
from llama_index.core.schema import MetadataMode

JUDGE_SYSTEM = """\
You grade answers from a Philippine legal research assistant. Reply with exactly one word: \
YES or NO. No explanation, no punctuation."""

FAITHFULNESS_PROMPT = """\
PROVISIONS (the only context the assistant was given):
{context}

ANSWER:
{answer}

Is every claim, citation and quotation in the ANSWER supported by the PROVISIONS, with \
nothing added from outside them? Reasoning that follows from a quoted provision counts as \
supported. Reply YES or NO."""

CORRECTNESS_PROMPT = """\
QUESTION:
{question}

REFERENCE ANSWER (correct):
{reference}

ANSWER TO GRADE:
{answer}

Does the ANSWER TO GRADE reach the same legal result as the REFERENCE ANSWER? Ignore \
wording, length and extra detail; judge only the legal result. Reply YES or NO."""


def parse_verdict(reply: str | None) -> str | None:
    """'YES' or 'NO' when the reply is exactly that after trimming; otherwise None."""
    text = (reply or "").strip()
    return text if text in ("YES", "NO") else None


def ask(llm, prompt: str) -> tuple[float | None, str | None]:
    """One judge call. Returns (1.0 for YES or 0.0 for NO, None), or (None, the error)."""
    messages = [ChatMessage(role="system", content=JUDGE_SYSTEM), ChatMessage(role="user", content=prompt)]
    try:
        reply = llm.chat(messages).message.content
    except Exception as exc:  # pylint: disable=broad-exception-caught
        return None, f"{type(exc).__name__}: {exc}"
    verdict = parse_verdict(reply)
    if verdict is None:
        return None, f"not YES/NO: {reply!r:.200}"
    return (1.0 if verdict == "YES" else 0.0), None


def context_text(nodes) -> str:
    """The chunks as the answer model saw them: text plus metadata (source, article, section)."""
    return "\n\n---\n\n".join(n.node.get_content(metadata_mode=MetadataMode.LLM) for n in nodes)


def judge_faithfulness(llm, nodes, answer: str) -> tuple[float | None, str | None]:
    return ask(llm, FAITHFULNESS_PROMPT.format(context=context_text(nodes), answer=answer))


def judge_correctness(llm, question: str, reference: str, answer: str) -> tuple[float | None, str | None]:
    return ask(llm, CORRECTNESS_PROMPT.format(question=question, reference=reference, answer=answer))
