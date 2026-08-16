"""A document has sections and labels. The answer usually lives in one.

Why
---
A contest PDF was loaded and the store held it — 「必須要件」 sixteen times
over — and the engine still answered 「本課題 必須要件」, an index rather
than an answer. The document's actual answer was two lines the ingest
never kept:

    2. 必須要件
        データベースへのデータ登録（INSERT 処理）
        データベースからのデータ参照（SELECT 処理）

Sentence-level ingest read 51 of 68 lines, and the 17 it dropped were the
answer. A bullet is not a sentence, so it was never placed.

What this does, and what it refuses to do
-----------------------------------------
It **quotes**. A heading's section is returned verbatim, in document
order; a label's value is returned as written. Nothing here paraphrases,
summarises, or infers, and every emitted string is a substring of the
source or it does not appear at all — the pass line the pre-registration
calls the one that decides the mechanism.

That makes this a retrieval claim, not a comprehension one. The engine
does not understand the requirements; it can quote the right two lines
when asked about the heading they sit under. Every verdict here is worded
to keep saying so.

A sidecar, never a vote
-----------------------
This index is written beside the store and read by its own stage under
its own door name. It never enters the census, the same rule
`jawiki_shallow` has lived under since 2026-08-14: material a reader can
be handed is not material that gets to vote.

The PDF's indentation is gone
-----------------------------
Measured on the contest PDF: every extracted line reports indent 0, so
the nesting a reader sees on the page is not in the text. Structure is
recovered from what survived — numbered headings (`1. …`) and labelled
values (`提出期限: …`) — and from line width, because a line that wrapped
is a line that ran out of room. That last one is a measured property of
the document, not a constant: the wrap width is taken from the document
itself.
"""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

#: `1. 必須要件` — a numbered heading. The number must be followed by a
#: space, or 「1.5倍」 becomes a section.
_HEADING = re.compile(r"^(\d+)[.．]\s+(\S.*)$")

#: `提出期限: 2026 年 9 月 11 日（金） 23:59 まで` — a labelled value. The
#: name is short by construction; a long left side is a sentence that
#: happens to contain a colon.
_LABEL = re.compile(r"^(.{1,14}?)\s*[：:]\s*(\S.*)$")

#: A line ending in one of these has finished; anything else at full
#: width is a wrap.
_CLOSED = ("。", "）", ")", "」", "』", "：", ":", "；", ";", "！", "？")

#: A bullet marker at the head of a line. Measured: `load_paths` keeps
#: these (「• 提出期限: …」, 「o 入力フォームから…」) where the domain reader
#: strips them, and they are the strongest structural signal the PDF still
#: carries now that its indentation is gone. A line that starts with one
#: begins an item, so it can never be the continuation of the line above —
#: which is also what stops two adjacent items being welded into
#: 「…機能の実装o 画像ファイルの…」.
_BULLET = re.compile(r"^\s*(?:[•・○●◦▪‧·\u2022\u25e6]|o|[-*])\s+")


@dataclass
class Section:
    """One heading and the lines beneath it, verbatim and in order."""

    ordinal: int
    heading: str
    lines: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return {"ordinal": self.ordinal, "heading": self.heading,
                "lines": list(self.lines)}


def _display_width(s: str) -> int:
    """Rendered width, counting a CJK character as two columns.

    Measured, and the reason this function exists: in characters, a wrapped
    line came out at 36 and an unwrapped item at 35, so no ratio separated
    them. The document mixes ASCII (Python, JavaScript, UPDATE) with
    Japanese, and ASCII is half as wide — counting characters was measuring
    the wrong quantity. In display columns the same lines are 67 and 57,
    and the ordinary 0.8 floor separates them with room to spare. The
    threshold was not moved; the measurement was corrected.
    """
    return sum(2 if unicodedata.east_asian_width(c) in "WFA" else 1
               for c in s)


def _wrap_width(lines: List[str]) -> int:
    """The width this document wraps at, from the document."""
    widths = sorted(_display_width(x) for x in lines if x.strip())
    if not widths:
        return 0
    # The long tail is the wrapped body text; the max is the wrap width.
    return widths[-1]


def rejoin(text: str) -> List[str]:
    """Physical lines back into logical ones.

    A PDF line break is not a sentence break. 「以下」/「の要件および…」 is
    one clause split by the page, and reading it as two lines is why the
    requirement lines looked like fragments.

    A line is treated as wrapped when it is near the document's own wrap
    width AND does not end on a closing mark. The width test is what keeps
    「テーマ設定の例」 — seven characters — from swallowing the example
    beneath it.
    """
    raw = [x.rstrip() for x in (text or "").splitlines()]
    width = _wrap_width(raw)
    floor = int(width * 0.8) if width else 0
    out: List[str] = []
    buf = ""
    for line in raw:
        s = line.strip()
        if not s:
            if buf:
                out.append(buf)
                buf = ""
            continue
        if buf and (_BULLET.match(line) or _HEADING.match(s)):
            # A new item cannot be the tail of the previous one.
            out.append(buf)
            buf = ""
        buf = (buf + s) if buf else s
        wrapped = (_display_width(line.rstrip()) >= floor
                   and not s.endswith(_CLOSED))
        if not wrapped:
            out.append(buf)
            buf = ""
    if buf:
        out.append(buf)
    return out


def sections(text: str) -> Tuple[List[Section], Dict[str, str]]:
    """(sections, labels) — the document's own structure, verbatim.

    Text before the first heading belongs to no section; it is the
    document's preamble and is returned as section 0 so a question about
    the document itself has somewhere to land.
    """
    lines = rejoin(text)
    secs: List[Section] = [Section(ordinal=0, heading="", lines=[])]
    labels: Dict[str, str] = {}
    top = 0
    for line in lines:
        head = _BULLET.sub("", line).strip()
        m = _HEADING.match(head)
        if m:
            n = int(m.group(1))
            # A numbered line whose number does not continue the document's
            # own sequence is a nested list, not a new section. 「6. 提出物
            # および提出期限」 is followed by 「1. プロジェクトフォルダ一式」,
            # and reading that 1 as a section split the submission list into
            # four empty headings — so the section that should answer
            # 「提出物は」 held two lines and none of its four items.
            if n > top:
                top = n
                secs.append(Section(ordinal=n, heading=m.group(2).strip()))
                continue
        secs[-1].lines.append(line)
        lm = _LABEL.match(head)
        if lm:
            name, value = lm.group(1).strip(), lm.group(2).strip()
            # First writing wins: a document that says 提出期限 twice is
            # stating it once and referring back to it, and overwriting
            # would silently prefer the reference.
            if name and value and name not in labels:
                labels[name] = value
    if not secs[0].lines:
        secs.pop(0)
    return secs, labels


def index(text: str, source: str) -> Dict[str, Any]:
    """The whole document, indexed by what a person would ask about."""
    secs, labels = sections(text)
    return {"source": source,
            "sections": [s.as_dict() for s in secs],
            "labels": labels,
            "lines": len(rejoin(text))}


def sidecar_path(store_path: Path) -> Path:
    return Path(store_path).with_suffix(".documents.json")


def load(store_path: Path) -> Dict[str, Any]:
    p = sidecar_path(store_path)
    if not p.is_file():
        return {"documents": []}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (ValueError, OSError):
        return {"documents": []}


def save(store_path: Path, doc: Dict[str, Any]) -> Dict[str, Any]:
    """Add one document to the sidecar, replacing any earlier read of it."""
    book = load(store_path)
    docs = [d for d in book.get("documents", [])
            if d.get("source") != doc.get("source")]
    docs.append(doc)
    book["documents"] = docs
    sidecar_path(store_path).write_text(
        json.dumps(book, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"verdict": "WROTE", "source": doc.get("source"),
            "sections": len(doc.get("sections", [])),
            "labels": len(doc.get("labels", {})),
            "documents": len(docs),
            "note": "見出しと項目をそのまま保存した。引用のみで、票は持たない"}


def _norm(s: str) -> str:
    """Compare on content, not on the spaces a PDF sprinkled in."""
    return re.sub(r"[\s　]+", "", s or "")


def lookup(subject: str, book: Dict[str, Any]) -> Dict[str, Any]:
    """What the loaded documents say about this subject, verbatim.

    Labels are consulted before headings: 「提出期限」 is a labelled value
    in one document and would otherwise be answered by whatever section
    happens to contain the word.
    """
    q = _norm(subject)
    if not q:
        return {"verdict": "UNKNOWN_NO_SUBJECT"}

    for doc in book.get("documents", []):
        for name, value in (doc.get("labels") or {}).items():
            if _norm(name) == q:
                return {"verdict": "DOCUMENT_LABEL", "subject": name,
                        "text": "%s: %s" % (name, value),
                        "value": value, "source": doc.get("source"),
                        "quoted": True,
                        "note": "文書の記載をそのまま引用。要約も推論もしていない"}

    # Exact heading, then a heading that contains the subject. Containment
    # is second so 「必須要件」 cannot be taken by 「必須要件を満たす場合」
    # while an exact heading for it exists.
    for exact in (True, False):
        for doc in book.get("documents", []):
            for sec in doc.get("sections", []):
                h = _norm(sec.get("heading", ""))
                if not h:
                    continue
                hit = (h == q) if exact else (q in h)
                if not hit:
                    continue
                lines = list(sec.get("lines") or [])
                if not lines:
                    # A heading with nothing under it. Assembling an
                    # answer from neighbouring text would be inventing a
                    # section the document does not have.
                    return {"verdict": "UNKNOWN_NO_ITEMS",
                            "subject": sec.get("heading"),
                            "source": doc.get("source"),
                            "note": "見出しはあるが、その下に記載が無い"}
                return {"verdict": "DOCUMENT_SECTION",
                        "subject": sec.get("heading"),
                        "text": "\n".join(lines),
                        "lines": lines,
                        "ordinal": sec.get("ordinal"),
                        "source": doc.get("source"),
                        "quoted": True,
                        "note": "文書の該当節をそのまま引用。並び順は原文のまま"}
    return {"verdict": "UNKNOWN_NOT_IN_DOCUMENTS", "subject": subject}


#: WITHDRAWN 2026-08-17 — kept only so the failure is legible. Do not
#: wire this in. See docs/PREREGISTERED_2026-08-17_lexicon_heading_alias.md:
#: the floor was derived from twenty single-WORD negative controls and
#: measured against them, and it holds for words. People type SENTENCES,
#: and a sentence's token average drifts toward a document's general
#: vector: 29 of 50 commonsense questions crossed the floor and were
#: answered with a section of a contest PDF. 「氷は冷たいですか」 returned
#: 「原則として、授業で利用した以下の環境を用いて…Python, Flask」.
#:
#: The stop condition named exactly this and forbids a second threshold,
#: a margin term, or a bigger lexicon. It is honoured.
#:
#: The floor a proposed heading must clear. Pre-registered
#: 2026-08-17 as the smallest value that refuses all twenty declared
#: negative controls on the fit document, and confirmed on a held-out
#: document where the highest nonsense score was 0.348. It is a property
#: of THIS lexicon: swap the table and it must be re-derived, because a
#: floor carried over from another model is a number with no measurement
#: behind it.
ALIAS_FLOOR = 0.42


def propose_heading(subject: str, book: Dict[str, Any],
                    lex: Any, floor: float = ALIAS_FLOOR) -> Optional[str]:
    """A heading the loaded documents hold that this subject may mean.

    The dictionary PROPOSES; `lookup` still decides. It may only name a
    heading the index already contains, so the worst it can do is send the
    reader to the wrong section of a real document — never to a section
    that does not exist. Measured on a held-out document: 4 of 8
    paraphrases reached their heading, 4 were refused, 0 went wrong.

    A ranking read of a static embed table. No generation, no sampling —
    0-4 ms and byte-identical across runs, so determinism survives. The
    same table's POLARITY is forbidden (54.8%, a coin flip, measured
    2026-08-08) and is not touched here.
    """
    if lex is None or not subject.strip():
        return None
    keys: List[str] = []
    for doc in book.get("documents", []):
        keys.extend(str(sec.get("heading")) for sec in doc.get("sections", [])
                    if sec.get("heading"))
        keys.extend(str(k) for k in (doc.get("labels") or {}))
    if not keys:
        return None
    try:
        hits = lex.nearest(subject, keys, k=1)
    except Exception:
        return None
    if hits and hits[0][1] >= floor:
        return hits[0][0]
    return None


def verify_quoted(result: Dict[str, Any], text: str) -> bool:
    """Every emitted line is a substring of the source, or this fails.

    The mechanical check the pre-registration puts the whole mechanism on:
    a sentence naming a requirement the document does not contain is worse
    than the index it replaces.
    """
    flat = _norm(text)
    for line in (result.get("lines") or []):
        if _norm(line) not in flat:
            return False
    v = result.get("value")
    return not (v and _norm(v) not in flat)
