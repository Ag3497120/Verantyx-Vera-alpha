"""Sentences, by recombining forms and contents that were both written down.

I claimed a read-out could not become prose because the particles and verb
endings a sentence needs are not in the store, and emitting one would break
the closure that makes fabrication impossible. That was wrong. The particles
are in the corpus — `events.runs_with_particles` already keeps them — they
were simply never indexed. What could not be produced was not unproducible;
it was discarded at ingest.

So a sentence here is a recombination of two things that were each written
by somebody:

    form      a sentence with its content words punched out
              「未遂罪も同様に処する」 -> 「<0>も<1>に<2>する」
    content   the cores and facets a walk passed through

Neither is invented. The closure holds; what changed is that the connective
tissue is now stored instead of thrown away.

## Forms and contents have different provenance, and must stay apart

A form's source wrote that shape, not the claim the shape ends up carrying.
Merged into one store, 「〜のようなものがある」 would arrive as something a
document asserted. Kept apart, a generated sentence can say both: this
content came from here, this shape came from there, and only the first is
evidence.

## Slots carry a case, or the output is grammatical noise

The first version punched holes without recording what fits them, and
produced 「放送開始な2週間としては…」 — 〜な wants an adjectival stem and got
a noun. The particle after a hole says what the hole is: を marks an object,
に a target, な an adjectival stem, は a topic. Filling a hole with a term
whose case cannot match is how a generator stops being a recombination and
starts being a scramble.

## Measured, including where it is still bad

    3,264 prose sentences   ->  127 sentence-initial forms, 2-3 holes
    1,271 verbal nouns learned from the corpus (terms it puts before する)
    300 subjects sampled    ->  300 produce a sentence, typed or untyped

That last line is the honest one. Typing the slots does NOT reject forms; it
changes WHICH term enters a hole. It removed the visible breakages —
放送開始な2週間 (a noun in an adjectival slot), 借主を地位する (a noun that
is not a verb), 安全:危険 (a polarity key, not a word) — and left the
sentences grammatical more often than not:

    消防法第十六条に削除がある。
    刑事訴訟法第三百五十条にも判決ができるようになっていたものもある。
    市販がインターネットにサービスされている。

and left plenty that are not:

    詳細っこいエルシーブイである。
    犯罪係数はアドバンテージをエリミネーターとしている。

The remaining failures are semantic, not grammatical: the walk supplies
terms that co-occur in the store, and co-occurrence is not selectional fit.
Nothing here can tell that アドバンテージ is a poor object for 犯罪係数, and
the store does not contain the fact that would say so.

## What this is not

It does not know whether the sentence is true. A form says "X does Y to Z"
and the walk supplies an X, a Y and a Z that were near each other in the
store — that is a plausible-looking sentence assembled from real parts, and
the assembly is not evidence for it. Anything downstream must treat the
output as a draft with two citations attached, never as a claim the corpus
made. It is deliberately NOT wired into `gather`, `descend` or any verdict
path; a generated sentence must never be able to arrive where a reader
expects a citation.
"""
from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

#: A content word: kanji, katakana, or the marks that bind them.
_CONTENT = re.compile(r"[㐀-䶿一-鿿ァ-ヺー々〆]+")

#: Markup, formulae and latin that are not Japanese prose. A form harvested
#: from 「{\displaystyle X}」 is not a sentence and filling it produces
#: nothing a reader would accept.
_NOT_PROSE = re.compile(r"[=\{\}\[\]\\<>|#*A-Za-z0-9]|displaystyle")

#: A sentence ends on a predicate. Without this the harvest is dominated by
#: headings and list fragments, which have no verb to carry a claim.
_PREDICATE = re.compile(
    r"(である|でない|する|しない|した|ある|ない|いる|られる|れる|になる|となる|できる)$")

#: What the character after a hole says the hole must hold.
#:
#:   な   an adjectival stem — 静かな, 主な. A plain noun breaks it, and
#:        this is the case that produced 放送開始な2週間 before slots were
#:        typed at all.
#:   を   an object          に  a target        へ  a direction
#:   は   a topic            が  a subject       の  a modifier
#:   で   a means or place   と  a companion or quotation
CASE_OF: Dict[str, str] = {
    "な": "adjstem", "を": "object", "に": "target", "へ": "target",
    "は": "topic", "が": "subject", "の": "modifier", "で": "means",
    "と": "with", "も": "topic", "から": "source", "まで": "limit",
}

#: Suffixes that mark a term as an adjectival stem rather than a plain noun.
_ADJ_STEM = ("的", "式", "様", "急", "重要", "主", "新た", "明らか")

#: A term that is mostly digits, a counter, or a bare fragment is not
#: something to put in a sentence. Untyped, the first fills were 10日間,
#: 1980アイコ and 17時枠 — every one a real facet and none of them a word a
#: sentence wants.
_NUMERISH = re.compile(r"[0-9０-９]|^[〇一二三四五六七八九十百千万]+")
_COUNTER = re.compile(
    r"(年|月|日|時|分|秒|回|人|件|個|台|本|枚|度|割|歳|代|周年|時台|時枠|番)$")

MIN_LEN, MAX_LEN = 8, 40
MIN_HOLES, MAX_HOLES = 2, 3


@dataclass
class Form:
    """One sentence shape, with a case per hole and where it came from."""

    template: str
    cases: List[str] = field(default_factory=list)
    count: int = 0
    example: str = ""
    source: str = ""

    @property
    def holes(self) -> int:
        return len(self.cases)


#: Terms observed immediately before する / した / される in the corpus.
#: Learned, not listed: which nouns take する is a fact about Japanese that
#: the corpus already demonstrates thousands of times, and guessing it
#: produced 「借主を地位する」 — 地位 is a noun and never a verb.
VERBAL_NOUNS: Set[str] = set()

_SURU = re.compile(r"(する|した|される|されている|しない)")


def _case_after(text: str, at: int) -> str:
    """The case a hole imposes, read from what follows it."""
    tail = text[at:at + 2]
    for k in ("から", "まで"):
        if tail.startswith(k):
            return CASE_OF[k]
    if _SURU.match(text[at:at + 4] or ""):
        return "verbalnoun"
    return CASE_OF.get(text[at:at + 1], "free")


def harvest(
    texts: Iterable[Tuple[str, str]],
    *,
    head_only: bool = True,
) -> Dict[str, Form]:
    """Read documents into sentence forms. ``texts`` is (source, body).

    ``head_only`` keeps forms whose first hole is at the start of the
    sentence. Mid-sentence fragments — 「が<0>の<1>となる」 — are clauses cut
    out of a longer sentence and read as broken when used alone.
    """
    forms: Dict[str, Form] = {}
    # First pass: learn which nouns the corpus actually conjugates with する.
    for _source, body in texts:
        for m in _CONTENT.finditer(body or ""):
            if _SURU.match((body or "")[m.end():m.end() + 4]):
                w = m.group(0)
                if 2 <= len(w) <= 8:
                    VERBAL_NOUNS.add(w)
    for source, body in texts:
        for raw in re.split(r"[。\n]", body or ""):
            s = raw.strip()
            if not (MIN_LEN <= len(s) <= MAX_LEN):
                continue
            if _NOT_PROSE.search(s) or not _PREDICATE.search(s):
                continue
            parts: List[str] = []
            cases: List[str] = []
            pos = 0
            for m in _CONTENT.finditer(s):
                parts.append(s[pos:m.start()])
                parts.append(f"<{len(cases)}>")
                cases.append(_case_after(s, m.end()))
                pos = m.end()
            parts.append(s[pos:])
            tpl = "".join(parts)
            if not (MIN_HOLES <= len(cases) <= MAX_HOLES):
                continue
            if head_only and not tpl.startswith("<0>"):
                continue
            f = forms.get(tpl)
            if f is None:
                forms[tpl] = Form(template=tpl, cases=cases, count=1,
                                  example=s, source=source)
            else:
                f.count += 1
    return forms


def fits(term: str, case: str) -> bool:
    """Can this term stand in a hole of this case?

    Only the checks that can be made from the string are made. A noun in an
    object slot is accepted because nothing here can tell a good object from
    a bad one; an adjectival slot is checked because 〜な after a plain noun
    is wrong in a way that is visible without knowing the meaning.
    """
    t = (term or "").strip()
    if not t or len(t) > 12 or len(t) < 2:
        return False
    if _NUMERISH.search(t) or _COUNTER.search(t):
        return False
    if ":" in t or "：" in t:
        # A polarity-keyed facet (安全:危険) is a claim about an aspect, not
        # a word. It leaked into a sentence before this check existed.
        return False
    if case == "adjstem":
        return t.endswith(_ADJ_STEM)
    if case == "verbalnoun":
        return t in VERBAL_NOUNS
    if case == "modifier":
        return len(t) <= 8
    return True


@dataclass
class Draft:
    """A generated sentence, and the two places it came from."""

    text: str
    template: str
    fills: List[str]
    content_from: List[str] = field(default_factory=list)
    form_from: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {"text": self.text, "template": self.template,
                "fills": self.fills, "content_from": self.content_from,
                "form_from": self.form_from,
                "note": "content and form have separate sources; neither "
                        "makes this sentence true"}


def compose(
    forms: Dict[str, Form],
    subject: str,
    facets: Sequence[str],
    *,
    limit: int = 3,
    content_from: Optional[Sequence[str]] = None,
) -> List[Draft]:
    """Sentences about ``subject`` using ``facets``, best-supported first.

    The subject takes hole 0 and the facets fill the rest, each checked
    against its slot's case. A form no available term fits is skipped rather
    than filled anyway — that skip is the whole difference between this and
    the untyped version.
    """
    pool = [f for f in facets if f and f != subject]
    out: List[Draft] = []
    # Rotate the form order per subject. Taking the most frequent form that
    # fits gave every step of a walk the same shape — eight sentences all
    # reading 「<0>も<1>に<2>する」, which is a template being recited rather
    # than a text. The offset is derived from the subject, so it is stable.
    ranked = sorted(forms.items(), key=lambda kv: (-kv[1].count, kv[0]))
    if ranked:
        off = sum(ord(c) for c in subject) % len(ranked)
        ranked = ranked[off:] + ranked[:off]
    for tpl, form in ranked:
        if not fits(subject, form.cases[0]):
            continue
        picks: List[str] = [subject]
        used: Set[str] = {subject}
        ok = True
        for case in form.cases[1:]:
            cand = [t for t in pool if t not in used and fits(t, case)]
            if not cand:
                ok = False
                break
            picks.append(cand[0])
            used.add(cand[0])
        if not ok:
            continue
        text = tpl
        for i, p in enumerate(picks):
            text = text.replace(f"<{i}>", p)
        out.append(Draft(text=text + "。", template=tpl, fills=picks,
                         content_from=list(content_from or []),
                         form_from=form.source))
        if len(out) >= limit:
            break
    return out


def compose_walk(
    store: Any,
    forms: Dict[str, Form],
    trace: Any,
    *,
    per_step: int = 1,
) -> List[Dict[str, Any]]:
    """One or more sentences per step of a recorded walk.

    The walk supplies the order and the subject; the forms supply the shape.
    Both are recordings, so the same walk over the same store and forms
    produces the same text every time.
    """
    labels = getattr(store, "source_labels", set()) or set()
    out: List[Dict[str, Any]] = []
    for core in getattr(trace, "seen", []):
        facets = [f for f in (store.crosses.get(core) or {})
                  if f not in labels and 2 <= len(f) <= 10]
        drafts = compose(forms, core, sorted(facets), limit=per_step,
                         content_from=[core])
        for d in drafts:
            out.append(d.as_dict())
    return out
