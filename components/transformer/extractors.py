"""Custom extractors and text transformation helpers."""

import re
from typing import Any, Sequence

from llama_index.core.extractors.interface import BaseExtractor
from llama_index.core.schema import BaseNode


# --- Regexes for text transformations ---

_HEADING_LINE = re.compile(r"^(#{1,6})\s+(.*)$")
_EMPHASIS = re.compile(r"\*{1,2}|_{1,2}")
_ARTICLE_LINE = re.compile(r"^\s*-?\s*\*\*\s*(Article\s+\d+\.?)\s*\*\*\s*(.*)")
# A bold article marker after other text on the same line (not a "- **Article" list item)
_MIDLINE_ARTICLE = re.compile(r"(?<=[^\s-])[ \t]+(\*\*\s*Article\s+\d+\.?\s*\*\*)")
# "Article" and its number split across lines: "**Article**\n\n**368.**" or "... Belongs Article\n\n**419.**"
_SPLIT_ARTICLE = re.compile(r"[ \t]*(?:\*\*)?Article(?:\*\*)?[ \t]*\n+[ \t]*\*\*(\d+)\.\*\*")
# Article inside another bold heading: "**CHAPTER 2 Quasi-delicts Article 2176.**"
_HEADING_WITH_ARTICLE = re.compile(r"\*\*([^*]+?)\s+Article\s+(\d+)\.\*\*")
# Plain uppercase marker: "ARTICLE 1642." (prose cross-references are lowercase "article")
_PLAIN_ARTICLE = re.compile(r"(?<![\w*])ARTICLE\s+(\d+)\.(?!\*)")
# Bare bold number at line start, where OCR lost the word "Article": "**1156.** An obligation ..."
_BARE_ARTICLE_NUMBER = re.compile(r"(?m)^[ \t]*\*\*(\d+)\.\*\*")

_CONST_ARTICLE_RE = re.compile(r"^(ARTICLE\s+[IVXLCDM]+)\s+(.*)", re.IGNORECASE)
_CONST_SECTION_RE = re.compile(r"^(Section\s+\d+\.?)\s*$", re.IGNORECASE)
_CONST_ALLCAPS_RE = re.compile(r"^[A-Z][A-Z\s,\-]+$")

_HEADING_ARTICLE_RE = re.compile(r"^ARTICLE\s+([IVXLCDM]+|\d+)\b", re.IGNORECASE)
_HEADING_SECTION_RE = re.compile(r"^SECTION\s+(\d+)\b", re.IGNORECASE)



# --- Custom extractors ---

class LegalHeadingExtractor(BaseExtractor):
    """Set `article` and `section` from the Markdown headings, without an LLM.

    Must run after MarkdownNodeParser and before any splitter: header_path only
    holds the parent headings, so the node's own heading ("### Section 14.") is
    read from its first line, which a splitter would cut off. Splitters copy
    metadata to every chunk, so each chunk keeps its citation.
    """

    async def aextract(self, nodes: Sequence[BaseNode]) -> list[dict[str, Any]]:
        results = []
        for node in nodes:
            headings = [h for h in node.metadata.get("header_path", "").split("/") if h]
            first_line = node.get_content().split("\n", 1)[0]
            if m := _HEADING_LINE.match(first_line):
                headings.append(m.group(2).strip())

            meta = {}
            for heading in headings:
                if m := _HEADING_ARTICLE_RE.match(heading):
                    meta["article"] = f"Article {m.group(1).upper()}"
                elif m := _HEADING_SECTION_RE.match(heading):
                    meta["section"] = f"Section {m.group(1)}"
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
    # The PDF writes article markers in several layouts. Normalize each to
    # "**Article N.**" at the start of a line, or the article merges into the previous one.
    markdown = _SPLIT_ARTICLE.sub(r"\n**Article \1.**", markdown)
    markdown = _HEADING_WITH_ARTICLE.sub(r"**\1**\n**Article \2.**", markdown)
    markdown = _PLAIN_ARTICLE.sub(r"\n**Article \1.**", markdown)
    markdown = _BARE_ARTICLE_NUMBER.sub(r"**Article \1.**", markdown)
    # Some lines hold several articles ("... (n) **Article 85.** A marriage ...")
    markdown = _MIDLINE_ARTICLE.sub(r"\n\1", markdown)
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
