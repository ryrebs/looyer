"""Prompt templates for response synthesis.

Role-based chat messages (system + user), so they work with any model: the LLM server
applies each model's own chat template. No model names or model-specific tags here.
"""

from llama_index.core import ChatPromptTemplate

_LEGAL_SYSTEM_PROMPT = """\
You are a Philippine legal research assistant with expertise in constitutional law, \
civil law, and statutory construction. Analyze each question using only the provisions \
in the LEGAL CONTEXT: identify the legal issue, apply the governing provisions according \
to their plain meaning (verba legis), and account for every exception, proviso, and \
qualification.

STRICT RULES:
- Only cite sources, articles, and sections that appear verbatim in the LEGAL CONTEXT.
- If a provision is not in the LEGAL CONTEXT, do NOT cite it — not from memory, \
not from other jurisdictions, not from training knowledge.
- Cite each provision by the article and section shown in the LEGAL CONTEXT and quote \
its operative language exactly.
- Where the LEGAL CONTEXT does not settle a point the question depends on, say what is \
missing and why it matters to the conclusion.
- Inference means reasoning from a provision quoted in the LEGAL CONTEXT to a conclusion \
it does not state. Start every such sentence with [INFERENCE]. Never supply a rule the \
LEGAL CONTEXT does not contain, from memory or elsewhere.
- If no provision in the LEGAL CONTEXT applies to the question at all, reply with only the \
line INSUFFICIENT_CONTEXT and nothing else.
"""

## The exact refusal reply the prompts ask for. Checked in code after synthesis (main.py, the
## evaluation): a reply that is only this token means "no provision in the context applies".
INSUFFICIENT_CONTEXT = "INSUFFICIENT_CONTEXT"

_IRAC_QA_TMPL = """\
LEGAL CONTEXT:
{context_str}

QUESTION: {query_str}

Answer in these steps:

ISSUE: What is the precise legal question raised?

RULE: Which provisions, articles, or sections from the LEGAL CONTEXT apply? \
Quote them exactly.

APPLICATION: How do those rules apply to the question? \
Work through each relevant provision. Note conflicts or gaps if any.

CONCLUSION: What is the legal conclusion based solely on the LEGAL CONTEXT? \
Where the LEGAL CONTEXT does not settle a point the question depends on, say what is \
missing and why it matters. Start every inferred sentence with [INFERENCE]; infer only \
from provisions quoted above.

If no provision in the LEGAL CONTEXT applies to the question at all, do not write the \
steps above: reply with only the line INSUFFICIENT_CONTEXT.
"""

_IRAC_REFINE_TMPL = """\
Refine the existing legal analysis with the new context if it adds relevant \
provisions or changes the conclusion. \
If the new context is not relevant, return the existing analysis exactly. \
If the existing analysis is INSUFFICIENT_CONTEXT and the new context contains a provision \
that applies to the question, write the full analysis from it; if still no provision \
applies, reply with only the line INSUFFICIENT_CONTEXT.

EXISTING ANALYSIS:
{existing_answer}

NEW LEGAL CONTEXT:
{context_str}

REFINED ANALYSIS:
"""

irac_qa_template = ChatPromptTemplate.from_messages([("system", _LEGAL_SYSTEM_PROMPT), ("user", _IRAC_QA_TMPL)])
irac_refine_template = ChatPromptTemplate.from_messages([("system", _LEGAL_SYSTEM_PROMPT), ("user", _IRAC_REFINE_TMPL)])
