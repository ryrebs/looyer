"""Parsed-document cache.

Docling parsing is expensive, so we run it once per unique file *content* and
save the resulting Markdown to disk. The cache key is a hash of the file's
bytes: identical content -> same key (cache hit); changed content -> new key
(automatic re-parse). No timestamps, no manual invalidation.
"""

import hashlib
from pathlib import Path

# Default cache location. Configurable later — you said you'd specify a path.
CACHE_DIR = Path(".cache/parsed")


def content_hash(path: Path) -> str:
    """Return a stable hex hash of the file's raw bytes.

    Reads in chunks so a large PDF doesn't load entirely into memory just to
    be hashed. The hash identifies the *content* — rename the file, same hash;
    edit one byte, different hash.
    """
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(65536), b""):
            h.update(block)
    return h.hexdigest()


def get_cache_path(path: Path, cache_dir: Path = CACHE_DIR) -> Path:
    """Where the parsed Markdown for this source file would live.

    The filename is the content hash + .md, so the same content always maps to
    the same cache file, and changed content maps to a new one.
    """
    return cache_dir / f"{content_hash(path)}.md"


def read_cache(path: Path, cache_dir: Path = CACHE_DIR) -> str | None:
    """Return cached Markdown if we've already parsed this exact content, else None.

    None is the signal to the caller: 'not cached — you need to parse it.'
    """
    cached = get_cache_path(path, cache_dir)
    if cached.exists():
        return cached.read_text(encoding="utf-8")
    return None


def write_cache(path: Path, markdown: str, cache_dir: Path = CACHE_DIR) -> None:
    """Save freshly-parsed Markdown to the cache for next time."""
    cache_dir.mkdir(parents=True, exist_ok=True)  # create .cache/parsed/ if needed
    get_cache_path(path, cache_dir).write_text(markdown, encoding="utf-8")
