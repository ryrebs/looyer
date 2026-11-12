"""
Load and parse the necessary document.
"""

from pathlib import Path

from .cache import read_cache, write_cache
from .parse import parse_to_markdown

# File types the loader accepts. Anything else is rejected here, before a parser tries to guess.
SUPPORTED_EXTENSIONS = {".pdf", ".html", ".md", ".txt"}


def load_document(path: str, use_docling: str = False) -> str:
    """Create the markdown file"""

    p = Path(path)
    if not p.is_file():
        raise FileNotFoundError(f"No such file: {path}")
    if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported file type {p.suffix!r}: {path}. Supported: {sorted(SUPPORTED_EXTENSIONS)}")

    # 1. Already parsed this exact content? Return the saved Markdown instantly.
    cached = read_cache(p)
    if cached is not None:
        return cached

    # 2. Miss — run the expensive parse once, save it, return it.
    markdown = parse_to_markdown(p, use_docling=use_docling)
    markdown = markdown.strip()
    if not markdown:
        raise ValueError(f"No extractable text found in {path}")

    write_cache(p, markdown)
    return markdown
