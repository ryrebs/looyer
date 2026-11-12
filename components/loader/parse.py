"""Docling  and pymupdf4llm parsing. Any supported file → structured Markdown."""

from pathlib import Path

import pymupdf4llm


def _parse_with_pymupdf(path: Path) -> str:
    """Parse via pymupdf4llm (fast, font-based, local, no model)."""
    return pymupdf4llm.to_markdown(str(path))


def _parse_with_docling(path: Path) -> str:
    """Parse via Docling (handles formats pymupdf4llm doesn't, e.g. HTML).

    Imported lazily so the Docling dependency is only loaded when actually used.
    """
    from docling.document_converter import DocumentConverter  # pylint: disable=import-outside-toplevel

    converter = DocumentConverter()
    result = converter.convert(str(path))
    return result.document.export_to_markdown()


# Already text: read as-is. pymupdf4llm would render them as a page and guess the Markdown back from
# font sizes, which changes the text (a sentence became a "##" heading and lost its period).
TEXT_EXTENSIONS = {".md", ".txt"}


def parse_to_markdown(path, use_docling: bool = False) -> str:
    """Parse a document to Markdown.

    One unified entry point:
      - .md / .txt                  -> read as-is (use_docling is ignored)
      - use_docling=False (default) -> pymupdf4llm
      - use_docling=True            -> Docling

    If the default backend doesn't handle a given file well, switch manually
    by passing use_docling=True.
    """
    p = Path(path)
    if p.suffix.lower() in TEXT_EXTENSIONS:
        return p.read_text(encoding="utf-8")
    if use_docling:
        return _parse_with_docling(p)
    return _parse_with_pymupdf(p)
