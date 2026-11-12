"""Lock-in tests for Layer 1"""

from pathlib import Path

import pytest

from components.loader import cache, loaders
from components.loader.loaders import load_document


@pytest.fixture(autouse=True)
def temporary_cache(monkeypatch, tmp_path: Path):
    """Every test gets an empty cache in a temporary folder, never the real .cache/parsed/.

    Otherwise a test reads a result saved by an earlier run (even from older code) instead of parsing.
    """
    cache_dir = tmp_path / "cache"
    monkeypatch.setattr(loaders, "read_cache", lambda p: cache.read_cache(p, cache_dir))
    monkeypatch.setattr(loaders, "write_cache", lambda p, markdown: cache.write_cache(p, markdown, cache_dir))


# --- happy paths: each supported format returns usable text ---


def test_loads_plain_text(tmp_path: Path):
    f = tmp_path / "note.txt"
    f.write_text("Hotel guests are liable for deposited effects.", encoding="utf-8")
    assert "Hotel guests" in load_document(str(f))


def test_loads_markdown_keeps_structure(tmp_path: Path):
    f = tmp_path / "doc.md"
    f.write_text("# Title\n\nSome rule about deposits.", encoding="utf-8")
    out = load_document(str(f))
    assert "# Title" in out  # we deliberately KEEP markdown structure
    assert "deposits" in out


def test_loads_html_strips_boilerplate(tmp_path: Path):
    f = tmp_path / "page.html"
    f.write_text(
        "<html><body><nav>menu</nav><article><p>The deposit of effects is necessary.</p></article></body></html>",
        encoding="utf-8",
    )
    out = load_document(str(f))
    assert "deposit of effects" in out


def test_html_handles_non_utf8_encoding(tmp_path: Path):
    # The bug you actually hit: a Windows-1252 byte (0xd7 = '×') must not crash.
    f = tmp_path / "legacy.html"
    f.write_bytes(
        "<html><body><article><p>Price 5 \xd7 2 dollars here.</p></article></body></html>".encode("windows-1252")
    )
    out = load_document(str(f))  # should NOT raise UnicodeDecodeError
    assert "Price" in out


# --- boundary failures: bad input must fail loudly, not silently ---


def test_missing_file_raises(tmp_path: Path):
    with pytest.raises(FileNotFoundError):
        load_document(str(tmp_path / "nope.txt"))


def test_unsupported_extension_raises(tmp_path: Path):
    f = tmp_path / "data.xyz"
    f.write_text("whatever", encoding="utf-8")
    with pytest.raises(ValueError):
        load_document(str(f))


def test_empty_file_raises(tmp_path: Path):
    f = tmp_path / "empty.txt"
    f.write_text("   \n  ", encoding="utf-8")  # whitespace only
    with pytest.raises(ValueError):
        load_document(str(f))
