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

#: 「第2条 出張旅費」「第三章 総則」。日本語の規程はこの形で区切られるので、
#: アラビア数字とピリオドしか見ていない読み手には、条文書は一枚の節として
#: 届く。実測: 4条の社内規程が節1つに畳まれ、「第2条は」に見出し行だけが
#: 返り、その条の中身は一行も出なかった。
_JP_HEADING = re.compile(
    r"^第\s*([0-9０-９一二三四五六七八九十百]+)\s*([条章節項編款])\s*(.*)$")

_KANJI_DIGIT = {"〇": 0, "一": 1, "二": 2, "三": 3, "四": 4,
                "五": 5, "六": 6, "七": 7, "八": 8, "九": 9}


def _numeral(raw: str) -> Optional[int]:
    """「12」「１２」「十二」→ 12。読めなければ None。

    見出しの番号は順序の判定にしか使わないので、位取りは十・百まで。
    それを超える条番号を持つ規程はあるが、そこは番号ではなく本文の
    並び順で足りる — 読めない番号は見出しとして扱わないだけで、行は
    直前の節に残る。
    """
    t = raw.translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    if t.isdigit():
        return int(t)
    total, unit = 0, 0
    for ch in t:
        if ch in _KANJI_DIGIT:
            unit = _KANJI_DIGIT[ch]
        elif ch == "十":
            total += (unit or 1) * 10
            unit = 0
        elif ch == "百":
            total += (unit or 1) * 100
            unit = 0
        else:
            return None
    return (total + unit) or None

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
    #: 番号の単調性は見出しの種類ごとに数える。章と条が交互に現れる規程
    #: (第1章 → 第1条 第2条 → 第2章 → 第3条) を一本の数列として読むと、
    #: 章に戻った時点で以降の条が全て節に立たなくなる。
    tops: Dict[str, int] = {}
    for line in lines:
        head = _BULLET.sub("", line).strip()
        jm = _JP_HEADING.match(head)
        if jm:
            n = _numeral(jm.group(1))
            kind = jm.group(2)
            if n is not None and n > tops.get(kind, 0):
                tops[kind] = n
                # 見出しは行そのまま。「第2条」は人が引用に使う名前で、
                # アラビア数字の「1.」とは違って落とせない。
                secs.append(Section(ordinal=n, heading=head))
                continue
        m = _HEADING.match(head)
        if m:
            n = int(m.group(1))
            top = tops.get("", 0)
            # A numbered line whose number does not continue the document's
            # own sequence is a nested list, not a new section. 「6. 提出物
            # および提出期限」 is followed by 「1. プロジェクトフォルダ一式」,
            # and reading that 1 as a section split the submission list into
            # four empty headings — so the section that should answer
            # 「提出物は」 held two lines and none of its four items.
            if n > top:
                tops[""] = n
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


def content_terms(subject: str) -> List[str]:
    """The subject's own content words — no morphology, no guessing.

    Shared by the two places that need to know what a question is about:
    the fallback probe below, and the narrowing inside a matched section.
    """
    import re as _re
    from .lang import ja_content_runs
    out = set(ja_content_runs(subject))
    out |= {m.group(0) for m in _re.finditer(r"[ァ-ヶーA-Za-z0-9]+[一-龥]{0,2}", subject)}
    out |= {m.group(0) for m in _re.finditer(r"[一-龥]{2,}", subject)}
    return [t for t in out if len(t) >= 2]


def _ngrams(text: str, n: int) -> set:
    t = _norm(text)
    return {t[i:i + n] for i in range(max(0, len(t) - n + 1))}


#: 語段・2字段・3字段。concord の階段と同じ考えで、同じ行集合を別の解像度で
#: 読む。段を分割にしてはならない（語長で仕分けると各語が1段にしか入らず、
#: 一致は低いのではなく起こり得ない）ので、全段が全行を自分の粒度で見る。
_GRAINS = (0, 2, 3)


def _independent_in(probe: str, text: str) -> bool:
    """探査語 probe は text の中で、複合語の一部ではなく独立して立つか。

    実測 2026-08-18、経費精算規程で「飛行機のチケット代は精算できますか」
    に対し、より特定的な探査語(飛行機・チケット・食事代)は文書のどこにも
    無いので順に空振りし、最後に残った探査語「精算」が見出し「第7条
    不正精算」の部分文字列として一致した。「精算」は文書中に4回現れる
    一般語で、「不正精算」の中では「不正」に直接続く複合語の一部 —
    独立した主題ではない。無関係な条文(懲戒)が「精算できますか」の答え
    として返っていた。

    独立とは、text の中で probe の両隣が文字列の端か、内容字でない
    ことを言う。「不正精算」の「精算」は左隣が内容字(正)なので不合格、
    「精算方法」の「精算」も右隣が内容字(方)なので不合格。「経費精算
    規程」全体を探査語として引くのとは違う話 — ここは一片の話。
    """
    i = text.find(probe)
    while i != -1:
        left_ok = i == 0 or not _is_content_char(text[i - 1])
        j = i + len(probe)
        right_ok = j == len(text) or not _is_content_char(text[j])
        if left_ok and right_ok:
            return True
        i = text.find(probe, i + 1)
    return False


def _is_content_char(c: str) -> bool:
    return ("一" <= c <= "龥" or "ぁ" <= c <= "ん" or "ァ" <= c <= "ヶ"
            or c == "ー" or c.isalnum())


def _stair_pick(lines: List[str], subject: str) -> Optional[str]:
    """節の中で、問いが実際に指している行。段が一致したときだけ返す。

    見出しに一致すると節が丸ごと返る。それは問いが節そのもののとき
    （「第5条は」）は正しく、一行が答えるときは危ない。実測 2026-08-18、
    就業規則20問で17答のうち6答が節ごとで、最悪の形がこれだった:

        「時間外労働の割増賃金は」→ 第3条の3行
          時間外労働を命じる場合は、事前に所属長の承認を得るものとする。
          時間外労働の割増賃金は、通常賃金の25パーセント増しとする。  ← 答え
          深夜労働の割増賃金は、通常賃金の50パーセント増しとする。    ← 別の問いの答え

    25と50が同じ重さで並び、どちらが問いの答えかを読み手が選ぶ。

    先に手製の数え上げ（問いの語を最も多く含む行）を書いたが、これは誤り
    が二重だった。一つ、測定済みの機構があるのに手で書いた。二つ、直した
    分岐がこれらの問いの通る分岐ではなく、一度も発火せずに測定値が変わら
    なかった — にもかかわらず同じ数字を見て先へ進んだ。

    採るのは concord の階段の規律そのもの:

    * 段は分割ではなく解像度。全段が全行を、自分の粒度で見る
    * 同点の段は棄権する。全段が同点なら全段が同じ最小要素を選ぶので、
      証拠と無関係な一致が作られる（実測 全一致 73.3% → 辞書順で 23.7%、
      3対1 より悪い）
    * 2段以上が答えたうえで全一致したときだけ行を指す。割れたら節のまま

    合算はしない。段は独立に投票し、一致だけが信号になる — 重みを一つの
    数に畳むと、なぜその行かが数字に潰れて追試できなくなる。
    """
    if len(lines) < 2:
        return None
    votes: List[int] = []
    for g in _GRAINS:
        if g == 0:
            probe = set(content_terms(subject))
            units = lambda line: set(content_terms(line))  # noqa: E731
        else:
            probe = _ngrams(subject, g)
            units = lambda line, g=g: _ngrams(line, g)     # noqa: E731
        if not probe:
            continue
        scored = [(len(probe & units(l)), i) for i, l in enumerate(lines)]
        best = max(scored)[0]
        if best == 0:
            continue
        winners = [i for n, i in scored if n == best]
        if len(winners) != 1:
            continue
        votes.append(winners[0])
    if len(votes) < 2 or len(set(votes)) != 1:
        return None
    return lines[votes[0]]


def _doc_vocab(book: Dict[str, Any]) -> set:
    """文書が実際に書いている語だけ。外の辞書は持ち込まない。"""
    v: set = set()
    for d in book.get("documents", []):
        for sec in d.get("sections", []):
            v |= set(content_terms(sec.get("heading") or ""))
            for line in sec.get("lines") or []:
                v |= set(content_terms(line))
        for name, val in (d.get("labels") or {}).items():
            v |= set(content_terms(name)) | set(content_terms(val))
    return {w for w in v if len(w) >= 2}


def typo_hint(book: Dict[str, Any], term: str) -> List[str]:
    """この語は、文書が書いている別の語の打ち損じか。訂正はしない。

    誤字回復は測定済み(回復@5 84.8%、実在語への誤発火 0/500)なのに、
    文書の経路からは一度も呼ばれていなかった。実測 2026-08-18、
    「時間害労働の割増賃金は」は拒否文で「時間害労働」という非語を名指す
    だけで、文書が 時間外労働 を持っていることには触れなかった。

    語彙は文書自身のものに限る。外の辞書を積むと、文書に無い語を「在る」
    側に数えてしまう。3規程133語で 時間害労働→時間外労働、割増賃銀→
    割増賃金、宿泊日→宿泊費 を回復し、実在語4件への誤発火は 0。
    片仮名は位置単位を持たないので棄権する(インシデソト)。

    返すのは候補だけで、問いは書き換えない。どれを意味していたかを決める
    のは読み手であって、この関数ではない。
    """
    try:
        from . import lattice as _lat, typo_recovery as _tr
    except Exception:
        return []
    vocab = book.get("_vocab")
    if vocab is None:
        vocab = _doc_vocab(book)
        try:
            book["_vocab"] = vocab
            book["_lattice"] = _lat.build(vocab)
        except Exception:
            return []
    lat = book.get("_lattice")
    if lat is None or not vocab:
        return []
    try:
        r = _tr.recover(term, lattice=lat, vocab=vocab)
    except Exception:
        return []
    if r.get("verdict") != "TYPO_CANDIDATE":
        return []
    return [c.get("word") for c in (r.get("candidates") or [])[:3] if c.get("word")]


def _title_descent(book: Dict[str, Any], subject: str):
    """題名で文書へ降り、別の語で節へ降りる。二語とも文書自身の言葉。

    棚が一件のうちは要らない。増えた途端に効く — 実測 2026-08-18、3規程・
    7問で「出張の申請はいつまでに」が就業規則の第5条 出張へ降りた。正しい
    文書は出張旅費規程だが、その見出しは 申請/上限/期限 で、出張 はどこにも
    無い。題名にしかない。見出しだけを見る読み手には、文書の名前が見えて
    いなかった。

    conduct_tree を先に当てたが、この規模では移らなかった(7問中5問が
    UNKNOWN_NO_ROUTE)。あれは36法令の連合で測った器官で、核が13〜35しか
    ない文書には伝導の流れる先が無い。検証済みとは「その規模・その形で」
    であって、移送は無料ではない。だからここは器官を持ち込まず、book が
    既に持っている source を経路に載せるだけにしてある。

    二語を要求するのは、題名一致だけで文書を決め打ちしないため。出張 が
    題名に当たっても、節を選ぶ語が別に要る。片方しか無ければ何も返さず、
    従来の順路がそのまま動く。

    実測: 複数文書 6/7 → 7/7、単一文書の回帰 誤答 0。
    """
    terms = content_terms(subject)
    if len(terms) < 2:
        return None
    for doc in book.get("documents", []):
        stem = _norm(str(doc.get("source", "")).rsplit(".", 1)[0])
        if not stem:
            continue
        by_title = [t for t in terms if _norm(t) and _norm(t) in stem]
        if not by_title:
            continue
        rest = [t for t in terms if t not in by_title]
        for sec in doc.get("sections", []):
            head = _norm(sec.get("heading") or "")
            if not head or not sec.get("lines"):
                continue
            for t in rest:
                if _norm(t) and _norm(t) in head:
                    lines = list(sec["lines"])
                    one = _stair_pick(lines, subject)
                    return {"verdict": "DOCUMENT_LINE" if one else "DOCUMENT_SECTION",
                            "subject": sec.get("heading"),
                            "text": one or "\n".join(lines),
                            "lines": lines,
                            "source": doc.get("source"), "quoted": True,
                            "reached_via": "%s→%s" % (by_title[0], t),
                            "note": "題名「%s」で文書に降り、「%s」で節に降りた"
                                    % (doc.get("source"), t)}
    return None


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
                one = _stair_pick(lines, subject)
                if one is not None:
                    return {"verdict": "DOCUMENT_LINE",
                            "subject": sec.get("heading"),
                            "text": one,
                            "section": sec.get("heading"),
                            "source": doc.get("source"),
                            "quoted": True,
                            "note": "節の中で、問いの語を最も多く含む行。"
                                    "同数の行があれば節のまま返す"}
                return {"verdict": "DOCUMENT_SECTION",
                        "subject": sec.get("heading"),
                        "text": "\n".join(lines),
                        "lines": lines,
                        "ordinal": sec.get("ordinal"),
                        "source": doc.get("source"),
                        "quoted": True,
                        "note": "文書の該当節をそのまま引用。並び順は原文のまま"}
    # 棚が複数になった時点で、文書の名前が経路の一部になる。
    descended = _title_descent(book, subject)
    if descended is not None:
        return descended

    # ── ここから後退。すべて閉じた規則で、置換は必ず名乗る ──────────────
    #
    # Measured 2026-08-18 on a company-regulation probe: 「宿泊費の上限は」
    # fell through — 宿泊費 is a line under the 精算上限 heading, not a
    # heading itself — and 「グリーン車は使えますか」 has no heading at all,
    # yet the honest answer exists in the document: the governing section,
    # plus the checkable fact that the term is never mentioned. An engine
    # whose claim is "the document's own words or a typed refusal" owes
    # exactly those two shapes, and owed them since 2026-08-16, when
    # 「指定された要件は」 was recorded as the modified-noun-phrase hole.

    from .lang import ja_content_runs
    import re as _re

    # 探査語。ja_content_runs は文字種境界で割る（グリーン車 → グリーン+車）
    # ので、複合語の形も原文から直接拾う。閉じた2形だけ:
    # カタカナ/ラテン+漢字尾（グリーン車・エコノミークラス）と漢字連。
    cand = set(ja_content_runs(subject))
    cand |= {m.group(0) for m in _re.finditer(r"[ァ-ヶーA-Za-z0-9]+[一-龥]{0,2}", subject)}
    cand |= {m.group(0) for m in _re.finditer(r"[一-龥]{2,}", subject)}
    # ひらがなの内容語（はがき・きっぷ）。ja_content_runs は機能語を避ける
    # ためにひらがな連を捨てるが、主題の位置に立つ3字以上は内容語である
    # ことが多い。閉じた除外集合だけを引く。
    _HIRA_STOP = {"について", "ください", "ですか", "でしょう", "および",
                  "または", "ならびに", "こんにちは", "ありがとう"}
    # 疑問語で始まる連は問いの側であって主題ではない。実測: 「申出はいつ
    # までに」が探査語「はいつま」を作り、拒否文がその非語を名指した。
    _HIRA_ASK = ("いつ", "どこ", "どの", "どう", "どちら", "どれ",
                 "なに", "なぜ", "いく", "だれ")
    for m in _re.finditer(r"[ぁ-ん]{3,}", subject):
        run = m.group(0)
        # 内容字（漢字・カタカナ・ラテン）の直後に立つ先頭のひらがなは、
        # その語に付いた助詞である。文頭の「はがき」の は とは違う —
        # 位置がその二つを分ける。
        if m.start() > 0 and not ("ぁ" <= subject[m.start() - 1] <= "ん") \
           and run[0] in "のはがをにでへとも":
            run = run[1:]
        # 末尾の助詞を剥がす（はがきの → はがき）。主題位置のひらがな連は
        # 助詞まで一続きで取れてしまう。
        while len(run) > 3 and run[-1] in "のはがをにでへとも":
            run = run[:-1]
        if len(run) < 3:
            continue
        # 動詞語尾で終わる連（された・している）は形態であって主題ではない
        if run[-1] in "たてる":
            continue
        # ひらがなだけで か に終わる連は問いの尾（できますか・でしょうか）。
        if run[-1] == "か":
            continue
        if run.startswith(_HIRA_ASK):
            continue
        if run not in _HIRA_STOP:
            cand.add(run)
    probes: List[str] = []
    for run in sorted(cand, key=lambda r: (-len(r), subject.find(r))):
        # 主題自身も探査語に含める。以前は _norm(run) != q で除外して
        # いた — 完全一致の段が先に引いたから、という理由だが、その段が
        # 引くのはラベルと見出しだけで、行は一度も主題そのもので引かれて
        # いなかった。実測 2026-08-18: lookup("繰越上限は") は行に届き、
        # lookup("繰越上限") は UNKNOWN。engine は助詞を剥いでから引く
        # ので、裸の名詞主題が全部この穴に落ちていた。問い形の主題は
        # どの run とも一致しない(qに助詞が残る)ので、この変更は裸形
        # だけに効く。
        if len(run) >= 2:
            probes.append(run)

    # 不在の昇格: 最も特定的な探査語が文書のどこにも無いなら、より汎用の
    # 語の行で「答えたふり」をしてはならない。グリーン車で入れた型の一般化 —
    # 「書留の料金は」に対し、書留の不在を名乗った上で料金の定めを支配則
    # として引用する。不在は部分文字列検査で追試できる。
    _all_text = "\n".join(
        "\n".join(ln for sec in doc.get("sections", []) for ln in (sec.get("lines") or []))
        + "\n" + "\n".join("%s %s" % (k, v) for k, v in (doc.get("labels") or {}).items())
        + "\n" + "\n".join(sec.get("heading", "") for sec in doc.get("sections", []))
        for doc in book.get("documents", []))
    _absent = {pr for pr in probes if pr not in _all_text}

    # 1. 内容語で、ラベル→見出し→本文行の順に照合（主辞抽出を包含する形）
    for _pi, probe in enumerate(probes):
        pn = _norm(probe)
        for doc in book.get("documents", []):
            for name, value in (doc.get("labels") or {}).items():
                if _norm(name) == pn:
                    return {"verdict": "DOCUMENT_LABEL", "subject": name,
                            "text": "%s: %s" % (name, value), "value": value,
                            "source": doc.get("source"), "quoted": True,
                            "reached_via": probe,
                            "note": "「%s」の記載は無いが、内容語「%s」がラベルに一致。引用のみ"
                                    % (subject, probe)}
            for sec in doc.get("sections", []):
                head = _norm(sec.get("heading", ""))
                # 独立性は空白を保持した元の見出しで判定する。_norm は
                # PDFの余分な空白を吸収するために全空白を消すが、それは
                # 「第5条 交通費の上限」の条番号とタイトルの間という
                # "意味のある区切り" も一緒に消していた。実測: 独立性を
                # 正規化後の head に対して判定すると「第5条」の右隣が
                # 「交」(内容字)に見えて不合格になり、正しい節到達
                # (第5条は)まで巻き添えで壊れた。空白を残した見出しで
                # 見れば、区切りは区切りのまま残る。
                raw_head = sec.get("heading", "") or ""
                if pn and pn in head and sec.get("lines") \
                        and _independent_in(pn, raw_head):
                    one = _stair_pick(list(sec.get("lines") or []), subject)
                    if one is not None:
                        return {"verdict": "DOCUMENT_LINE",
                                "subject": sec.get("heading"),
                                "text": one, "section": sec.get("heading"),
                                "source": doc.get("source"), "quoted": True,
                                "reached_via": probe,
                                "note": "内容語「%s」で見出しに到達し、節の中で"
                                        "全段が一致した行。割れれば節のまま返す"
                                        % probe}
                    return {"verdict": "DOCUMENT_SECTION",
                            "subject": sec.get("heading"),
                            "text": "\n".join(sec.get("lines") or []),
                            "lines": list(sec.get("lines") or []),
                            "source": doc.get("source"), "quoted": True,
                            "reached_via": probe,
                            "note": "内容語「%s」で見出しに到達。該当節をそのまま引用" % probe}
        # 2. 本文の行そのもの。行は文書の最小の定めで、引用に要約は要らない
        for doc in book.get("documents", []):
            for sec in doc.get("sections", []):
                hit = [ln for ln in (sec.get("lines") or []) if probe in ln]
                _earlier_absent = [probes[j] for j in range(_pi)
                                   if probes[j] in _absent]
                if hit and _earlier_absent:
                    # 特定語（書留）は不在で、汎用語（料金）だけが当たった。
                    # 汎用の行を答えとして出せば、聞かれていない物の値段で
                    # 聞かれた物に答える捏造になる。
                    return {"verdict": "DOCUMENT_NOT_SPECIFIED",
                            "subject": _earlier_absent[0],
                            "section": sec.get("heading"),
                            "term_absent": True,
                            "governing": hit[:3],
                            "source": doc.get("source"), "quoted": True,
                            "text": "文書は「%s」を明記していない。「%s」の定め（%s）: %s"
                                    % (_earlier_absent[0], probe,
                                       sec.get("heading"), " / ".join(hit[:3])),
                            "typo_candidates": typo_hint(book, _earlier_absent[0]),
                            "note": "不在は部分文字列検査で追試可能。近い語の定めを"
                                    "引用しただけで、「%s」の答えではない" % _earlier_absent[0]}
                if hit:
                    return {"verdict": "DOCUMENT_LINE",
                            "subject": probe,
                            "section": sec.get("heading"),
                            "text": "\n".join(hit),
                            "lines": hit,
                            "source": doc.get("source"), "quoted": True,
                            "note": "「%s」を含む行をそのまま引用（節: %s）"
                                    % (probe, sec.get("heading"))}

    # 3. 明記なしの型。語そのものは文書のどこにも無い（部分文字列検査 —
    #    機械で追試できる主張）が、文字種境界の尾単位（グリーン車→車）を
    #    共有する定めがあるなら、それが「最も近い定め」。可否は言わない —
    #    沈黙の判定と、支配する条文の引用だけが、この構造に言える全部です。
    tails: List[str] = []
    for probe in probes or [q]:
        m = _re.search(r"[ァ-ヶーA-Za-z]+([一-龥]{1,2})$", probe)
        if m:
            tails.append(m.group(1))
    for tail in tails:
        for doc in book.get("documents", []):
            for sec in doc.get("sections", []):
                hit = [ln for ln in (sec.get("lines") or []) if tail in ln]
                if hit:
                    term = probes[0] if probes else subject
                    return {"verdict": "DOCUMENT_NOT_SPECIFIED",
                            "subject": term,
                            "section": sec.get("heading"),
                            "term_absent": True,
                            "governing": hit,
                            "source": doc.get("source"), "quoted": True,
                            "text": "文書は「%s」を明記していない。最も近い定め（%s）: %s"
                                    % (term, sec.get("heading"), " / ".join(hit)),
                            "note": "不在は部分文字列検査で追試可能。可否の判断はしていない — "
                                    "明記が無いという事実と、支配する定めの引用のみ"}

    # 最終の型付き沈黙にも誤字候補を添える。実測: 「経比精算とは」
    # (経費精算の誤字)は、不在昇格の対象になる行が無いのでここまで
    # 落ちてきて、以前は候補なしの沈黙だった。文書自身の語彙で候補が
    # 引ければ添える — 訂正はしない、答えたことにもしない。
    # subject そのもの("経比精算とは")ではなく、探査語(内容語だけを
    # 切り出したもの)を渡す。lattice は助詞付きの生文では回復できない。
    hint: List[str] = []
    for pr in probes:
        hint = typo_hint(book, pr)
        if hint:
            break
    out = {"verdict": "UNKNOWN_NOT_IN_DOCUMENTS", "subject": subject}
    if hint:
        out["typo_candidates"] = hint
    return out


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
