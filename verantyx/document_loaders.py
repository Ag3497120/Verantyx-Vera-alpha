"""Files in, `Document`s out — with a typed refusal when a format is beyond us.

`ingest_documents` takes text. Getting text out of the things people actually
have — a PDF from a city office, a .docx circular, an exported CSV — is a
separate job, and one where guessing is expensive: a loader that silently
returns an empty string produces a knowledge base that looks ingested and
holds nothing.

So every loader either returns text or says why it could not, and
`load_path` reports `UNKNOWN_NO_PARSER` / `UNKNOWN_UNREADABLE` rather than an
empty `Document`. An empty document is indistinguishable from a document that
genuinely said nothing, and those need opposite responses.

Formats, and what each costs:

    .txt .md .log     stdlib
    .html .htm        stdlib (html.parser)
    .csv .tsv         stdlib — each row becomes a sentence
    .json             stdlib — leaf strings, path-prefixed
    .docx             stdlib — a .docx is a zip of XML, so no dependency
    .pdf              needs the `docs` extra (pypdf); refuses by name without it

The .docx path is worth stating plainly because it looks like it should need
a library: Word documents are zip archives whose word/document.xml holds the
text, so zipfile and ElementTree are enough. PDF has no such shortcut — the
text is inside a compressed content stream with its own encoding — which is
why it is the one format that costs a dependency.
"""
from __future__ import annotations

import csv
import importlib.util
import io
import json
import re
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional

from .document_ingest import Document

TEXT_SUFFIXES = {".txt", ".md", ".markdown", ".log", ".rst"}
SUPPORTED = TEXT_SUFFIXES | {".html", ".htm", ".csv", ".tsv", ".json",
                             ".docx", ".pdf"}


class _TextExtractor(HTMLParser):
    """Visible text only. script/style contents are markup that happens to be
    made of words, and ingesting them fills the store with variable names."""

    _SKIP = {"script", "style", "noscript", "head"}

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []
        self._depth = 0

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in self._SKIP:
            self._depth += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._depth:
            self._depth -= 1

    def handle_data(self, data: str) -> None:
        if self._depth == 0 and data.strip():
            self.parts.append(data.strip())


def _from_html(raw: str) -> str:
    p = _TextExtractor()
    p.feed(raw)
    return "\n".join(p.parts)


def _from_csv(raw: str, delimiter: str = ",") -> str:
    """One row per line, `column: value` joined.

    Naming the column matters: a bare row of values loses which field each
    one was, and a store full of unlabelled values cannot be questioned.
    """
    out: List[str] = []
    reader = csv.reader(io.StringIO(raw), delimiter=delimiter)
    rows = list(reader)
    if not rows:
        return ""
    header = rows[0]
    for row in rows[1:]:
        pairs = [f"{h} {v}" for h, v in zip(header, row) if v.strip()]
        if pairs:
            out.append(", ".join(pairs) + ".")
    return "\n".join(out)


def _from_json(raw: str) -> str:
    """Leaf strings, prefixed with their path, so nesting survives as words."""
    try:
        obj = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON: {exc}") from exc

    out: List[str] = []

    def walk(node: Any, path: str) -> None:
        if isinstance(node, dict):
            for k, v in node.items():
                walk(v, f"{path} {k}".strip())
        elif isinstance(node, list):
            for v in node:
                walk(v, path)
        elif isinstance(node, str) and node.strip():
            out.append(f"{path} {node}".strip() + ".")
        elif isinstance(node, (int, float, bool)):
            out.append(f"{path} {node}.")

    walk(obj, "")
    return "\n".join(out)


_W_NS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"


def _from_docx(path: Path) -> str:
    """Paragraph text out of word/document.xml.

    A .docx is a zip; no third-party reader is needed. Runs inside a
    paragraph are joined without separators because Word splits a single
    sentence across runs whenever formatting changes mid-line — joining with
    spaces would insert them inside words.
    """
    import xml.etree.ElementTree as ET

    with zipfile.ZipFile(path) as zf:
        try:
            xml = zf.read("word/document.xml")
        except KeyError as exc:
            raise ValueError("not a Word document (no word/document.xml)") from exc
    root = ET.fromstring(xml)
    paragraphs: List[str] = []
    for para in root.iter(f"{_W_NS}p"):
        text = "".join(t.text or "" for t in para.iter(f"{_W_NS}t"))
        if text.strip():
            paragraphs.append(text.strip())
    return "\n".join(paragraphs)


def _from_pdf(path: Path) -> str:
    if importlib.util.find_spec("pypdf") is None:
        # `name=` matters: the caller distinguishes this from any other
        # missing import by `exc.name`, and the positional message alone
        # leaves it None, so the guard let it through and the whole batch
        # died on one PDF.
        raise ModuleNotFoundError("pypdf is required for PDF", name="pypdf")
    from pypdf import PdfReader  # type: ignore

    reader = PdfReader(str(path))
    pages = [(page.extract_text() or "").strip() for page in reader.pages]
    return "\n".join(p for p in pages if p)


def load_path(path: str, source: Optional[str] = None) -> Dict[str, Any]:
    """One file → a Document, or a typed reason it is not one.

    `source` defaults to the filename, which is what a reader needs to see
    beside a disputed claim. Callers with a real citation (a URL, an agency
    name) should pass it.
    """
    p = Path(path)
    label = source or p.name
    if not p.is_file():
        return {"verdict": "UNKNOWN_UNREADABLE", "path": str(p),
                "reason": "no such file"}

    suffix = p.suffix.lower()
    if suffix not in SUPPORTED:
        return {"verdict": "UNKNOWN_NO_PARSER", "path": str(p),
                "suffix": suffix or "(none)",
                "reason": f"no loader for '{suffix}'",
                "supported": sorted(SUPPORTED)}

    try:
        if suffix == ".pdf":
            text = _from_pdf(p)
        elif suffix == ".docx":
            text = _from_docx(p)
        else:
            raw = p.read_text(encoding="utf-8", errors="replace")
            if suffix in {".html", ".htm"}:
                text = _from_html(raw)
            elif suffix == ".csv":
                text = _from_csv(raw)
            elif suffix == ".tsv":
                text = _from_csv(raw, delimiter="\t")
            elif suffix == ".json":
                text = _from_json(raw)
            else:
                text = raw
    except ModuleNotFoundError as exc:
        if exc.name != "pypdf":
            raise
        return {"verdict": "UNKNOWN_NO_PARSER", "path": str(p), "suffix": suffix,
                "reason": "PDF support needs the `docs` extra",
                "install": "pip install verantyx-vera[docs]"}
    except (OSError, ValueError, zipfile.BadZipFile) as exc:
        return {"verdict": "UNKNOWN_UNREADABLE", "path": str(p),
                "reason": f"{type(exc).__name__}: {exc}"}

    if not text.strip():
        # Distinct from a parse failure, and it matters: a scanned PDF is an
        # image of text, and no amount of retrying the loader will help. The
        # answer is OCR, which this does not do and should not pretend to.
        return {"verdict": "UNKNOWN_EMPTY_DOCUMENT", "path": str(p),
                "reason": "parsed, but no extractable text (a scanned image?)"}

    return {"verdict": "ANSWER", "path": str(p), "source": label,
            "chars": len(text), "document": Document(source=label, text=text)}


def load_paths(paths: List[str]) -> Dict[str, Any]:
    """Load many, keeping the failures. A batch that drops what it could not
    read reports a smaller corpus as if it were the whole one."""
    loaded: List[Document] = []
    skipped: List[Dict[str, Any]] = []
    for path in paths:
        res = load_path(path)
        if res["verdict"] == "ANSWER":
            loaded.append(res["document"])
        else:
            skipped.append({k: v for k, v in res.items() if k != "document"})
    return {"documents": loaded, "loaded": len(loaded), "skipped": skipped}


def load_directory(root: str, recursive: bool = True) -> Dict[str, Any]:
    """Every supported file under a folder — the "here is the whole folder of
    circulars" path."""
    base = Path(root)
    if not base.is_dir():
        return {"documents": [], "loaded": 0,
                "skipped": [{"verdict": "UNKNOWN_UNREADABLE", "path": str(base),
                             "reason": "not a directory"}]}
    it = base.rglob("*") if recursive else base.glob("*")
    files = sorted(str(p) for p in it
                   if p.is_file() and p.suffix.lower() in SUPPORTED)
    out = load_paths(files)
    out["scanned"] = len(files)
    return out
