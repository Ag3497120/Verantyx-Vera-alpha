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
    """Body prose only — separated from navigation by what the text IS.

    Two structural attempts failed on real government pages, each for its
    own reason, and two failures from different causes are evidence the
    approach is wrong rather than unlucky. Counting nesting broke on void
    elements (<br> incremented and never decremented); counting same-named
    tags broke on the mismatched <div>s that real pages ship. Both assumed
    well-formed markup, which the web does not supply.

    So the distinction is drawn on the text instead, using the property that
    actually separates the two: navigation is link labels — short, no
    sentence-ending punctuation, and inside <a>. Body prose is sentences.
    Neither test can be defeated by malformed HTML, because neither reads
    the tree.

    Measured on five real 内閣府 pages: 「サイトマップ」「English」「内閣府
    ホーム」 drop out, and the paragraphs about 指定緊急避難場所 and
    プッシュ型支援 stay.
    """

    _SKIP = {"script", "style", "noscript", "head", "select", "option", "svg"}

    #: A line is prose if it ends like a sentence, or is long enough that it
    #: cannot be a menu label. Both thresholds are script-aware: Japanese
    #: carries far more meaning per character, so the length bar is lower.
    _SENTENCE_END = ("。", "．", ".", "！", "!", "？", "?", "」", "）", ")")
    _MIN_PROSE_CJK = 16
    _MIN_PROSE_LATIN = 40

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: List[str] = []
        self._skip = 0
        self._in_link = 0

    def handle_starttag(self, tag: str, attrs: Any) -> None:
        if tag in self._SKIP:
            self._skip += 1
        elif tag == "a":
            self._in_link += 1

    def handle_endtag(self, tag: str) -> None:
        if tag in self._SKIP and self._skip:
            self._skip -= 1
        elif tag == "a" and self._in_link:
            self._in_link -= 1

    def _is_prose(self, text: str) -> bool:
        if text.endswith(self._SENTENCE_END):
            return True
        floor = (self._MIN_PROSE_CJK if re.search(r"[぀-ヿ㐀-䶿一-鿿]", text)
                 else self._MIN_PROSE_LATIN)
        return len(text) >= floor

    def handle_data(self, data: str) -> None:
        text = data.strip()
        if not text or self._skip:
            return
        # Link text is a label until proven otherwise. A link whose text is a
        # whole sentence is a real sentence and is kept; 「サイトマップ」 is not.
        if self._in_link and not text.endswith(self._SENTENCE_END):
            return
        if self._is_prose(text):
            self.parts.append(text)


def _from_html(raw: str) -> str:
    p = _TextExtractor()
    p.feed(raw)
    return "\n".join(p.parts)


_FENCE = re.compile(r"^```.*?^```", re.M | re.S)
_INDENTED_CODE = re.compile(r"^(?: {4}|\t).*$", re.M)
_INLINE_CODE = re.compile(r"`[^`\n]*`")
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")     # keep the text, drop the target
_BARE_URL = re.compile(r"https?://\S+|www\.\S+")
_PATH = re.compile(r"(?:[\w.\-]+/){1,}[\w.\-]+")     # a/b/c.py
_FILENAME = re.compile(r"\b[\w\-]+\.(?:md|txt|json|ya?ml|toml|py|swift|js|ts|tsx|sh|rs|c|h|html|css|png|jpe?g|pdf|log|lock|cfg|ini)\b")
_HEADING_MARK = re.compile(r"^#{1,6}\s*", re.M)
_TABLE_ROW = re.compile(r"^\s*\|.*\|\s*$", re.M)


def _from_markdown(raw: str) -> str:
    """Prose only — code, URLs and paths removed before anything sees them.

    Measured on 2,491 real documents from this author's repositories: read as
    plain text, the highest-mass cores were `com` (2,506), `md` (2,209),
    `json` (1,673), `ts` (1,202). Those are the tails of URLs, the extensions
    in file paths, and the language tags on code fences. An index built from
    that is an index of file extensions, not of what the documents are about.

    Each removal is for a specific one of those:

      fenced/indented code   the language tag and every identifier inside
      inline code            `--store`, `CrossStore`, flag names
      link targets           kept the text, dropped the URL
      bare URLs              `example.com` contributes "com"
      paths and filenames    `docs/DESIGN.md` contributes "docs" and "md"
      table rows             pipe-delimited cells are data, not sentences

    Aggressive on purpose. A document whose content is entirely code has
    nothing to say to a prose index, and losing a sentence costs less than
    letting `md` become the most-discussed topic in the corpus.
    """
    text = _FENCE.sub(" ", raw)
    text = _TABLE_ROW.sub(" ", text)
    text = _INDENTED_CODE.sub(" ", text)
    text = _INLINE_CODE.sub(" ", text)
    text = _MD_LINK.sub(r"\1", text)
    text = _BARE_URL.sub(" ", text)
    text = _FILENAME.sub(" ", text)
    text = _PATH.sub(" ", text)
    text = _HEADING_MARK.sub("", text)
    # Collapse the holes so sentence splitting is not fooled by the gaps.
    return re.sub(r"[ \t]{2,}", " ", text)


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
            if suffix in {".md", ".markdown", ".txt", ".log", ".rst"}:
                # Plain-text notes get the same cleaning as Markdown.
                # Measured on a 107,752-line notes file that is nominally
                # .txt but full of markdown syntax and pasted code: read
                # raw, its top "topic" was `py` (1,471 mentions), with
                # `json` and `com` close behind; cleaned, all three vanish
                # and the real topics (ステップ, 投稿, 実装…) keep their
                # counts. On genuinely plain prose every one of these
                # patterns is a no-op except the path/URL strippers, which
                # are exactly what prose wants stripped. The known cost:
                # indented prose paragraphs are treated as code blocks and
                # dropped — the module's standing trade, a lost sentence
                # over a corpus-wide fake topic.
                text = _from_markdown(raw)
            elif suffix in {".html", ".htm"}:
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
