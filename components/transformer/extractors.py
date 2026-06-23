"""Custom extractors and text transformation helpers."""

import json
import re
from typing import Any, Sequence

from llama_index.core.extractors.interface import BaseExtractor
from llama_index.core.schema import BaseNode


# --- Regexes for text transformations ---

_HEADING_LINE = re.compile(r"^(#{1,6})\s+(.*)$")
_EMPHASIS = re.compile(r"\*{1,2}|_{1,2}")
_ARTICLE_LINE = re.compile(r"^\s*-?\s*\*\*\s*(Article\s+\d+\.?)\s*\*\*\s*(.*)")

_CONST_ARTICLE_RE = re.compile(r"^(ARTICLE\s+[IVXLCDM]+)\s+(.*)", re.IGNORECASE)
_CONST_SECTION_RE = re.compile(r"^(Section\s+\d+\.?)\s*$", re.IGNORECASE)
_CONST_ALLCAPS_RE = re.compile(r"^[A-Z][A-Z\s,\-]+$")

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)

# --- LLM extraction prompt ---

_LEGAL_META_PROMPT = """\
Extract legal document metadata from the text below.
Return ONLY a valid JSON object, no other text.

Fields (use null for missing fields, true/false for boolean fields):
- book: book or part name (e.g. "Book I: Persons")
- title: title within book (e.g. "Title I: Civil Personality")
- chapter: chapter name (e.g. "Chapter 1 Effect and Application of Laws")
- article_number: article number (e.g. "Article 1", "Art. 40")
- section_header: section heading if present
- proviso_flag: true if text contains a proviso ("provided that", "provided however")
- is_amended: true if text indicates it has been amended
- is_repealed: true if text indicates it has been repealed

Text:
{context}

JSON:"""


# --- Custom extractor ---

class LegalMetadataExtractor(BaseExtractor):
    llm: Any = None
    max_context_length: int = 600

    async def aextract(self, nodes: Sequence[BaseNode]) -> list[dict[str, Any]]:
        results = []
        for node in nodes:
            if self.llm is None:
                results.append({})
                continue
            headers = " > ".join(
                v for k, v in node.metadata.items() if k.startswith("Header")
            )
            context = node.get_content()[:self.max_context_length]
            if headers:
                context = f"Headers: {headers}\n\n{context}"
            prompt = _LEGAL_META_PROMPT.format(context=context)
            try:
                raw = self.llm.complete(
                    prompt, stop=["\n\n\n", "\nText:", "\nFields:"]
                ).text.strip()
                m = _JSON_RE.search(raw)
                meta = json.loads(m.group(0)) if m else {}
                meta = {k: v for k, v in meta.items() if v is not None}
            except Exception:
                meta = {}
            results.append(meta)
        return results


# --- Text transformation helpers ---

def _clean_heading_markup(markdown: str) -> str:
    """Strip inline bold/italic markers from heading lines only.

    pymupdf4llm emits headings like '## **Title**' (heading + bold). The bold
    markers break MarkdownNodeParser's heading detection (empty header_path) and
    leak '**' into section titles. We remove emphasis from HEADING lines only,
    leaving body text (where bold may carry meaning) untouched.
    """
    out = []
    for line in markdown.splitlines():
        m = _HEADING_LINE.match(line)
        if m:
            hashes, text = m.group(1), m.group(2)
            clean_text = _EMPHASIS.sub("", text).strip()
            out.append(f"{hashes} {clean_text}")
        else:
            out.append(line)
    return "\n".join(out)


def _promote_articles(markdown: str) -> str:
    out = []
    for line in markdown.splitlines():
        m = _ARTICLE_LINE.match(line)
        if m:
            out.append(f"### {m.group(1)}")
            rest = m.group(2).strip()
            if rest:
                out.append(rest)
        else:
            out.append(line)
    return "\n".join(out)


def transform_mk_cv_code(mk, trfs):
    if trfs is None:
        return mk
    trsf = mk
    for trf in trfs:
        trsf = trf(trsf)
    return trsf


def transform_html_const(markdown: str) -> str:
    """Promote flat Docling HTML output for the 1987 Constitution to hierarchical Markdown."""
    out = []
    is_first = True

    for tok in re.split(r"(\*\*.*?\*\*)", markdown, flags=re.DOTALL):
        bold_m = re.match(r"^\*\*(.*?)\*\*$", tok, re.DOTALL)
        if bold_m:
            inner = bold_m.group(1).strip()
            if is_first:
                out.append(f"\n# {inner}\n")
                is_first = False
            elif inner.upper() == "PREAMBLE":
                out.append(f"\n## {inner}\n")
            elif m := _CONST_ARTICLE_RE.match(inner):
                num, title = m.group(1), m.group(2).strip()
                out.append(f"\n## {num} — {title}\n")
            elif _CONST_SECTION_RE.match(inner):
                out.append(f"\n### {inner}\n")
            elif _CONST_ALLCAPS_RE.match(inner):
                out.append(f"\n### {inner}\n")
            else:
                out.append(f"**{inner}**")
        else:
            text = re.sub(r" {2,}", "\n\n", tok).strip()
            if text:
                out.append(f"\n\n{text}")

    return re.sub(r"\n{3,}", "\n\n", "".join(out)).strip()
