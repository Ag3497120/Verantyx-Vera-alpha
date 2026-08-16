"""Build a sentence from a frame, a pattern and its fillers. No model.

What this is
------------
Three tables answer three different questions, and a sentence needs all
three:

    frames    which cases this verb takes at all
    patterns  which cases occurred TOGETHER in one clause
    fillers   which nouns were observed in each slot

Composing from `frames` alone is what produced
「父がそれぞれ当該各号に父と期間を定める。」 — every case above a
threshold, filled independently, in combinations the corpus never held.
Measured afterwards: the mean pattern is 1.22 cases while the mean frame
is 3.20, so a threshold over the frame invents roughly two arguments per
sentence.

So the pattern chooses the shape and the fillers only fill it.

What it is not
--------------
Not fluency. Sentences come out one clause long, in canonical order, with
no paragraph, no development and no connective — `connective_render`
already owns joining, licensed by an edge.

Not knowledge. A composed sentence is a construction, and it says so:
`constructed: True` travels with every draft, the same mark
`meaning_descent` puts on its own. 「権利を有する」 being well-formed is
not a claim that anyone has a right to anything.

Register
--------
The tables are swappable. Grammar transfers — measured across encyclopedia,
civil code and labour law at 0.735–0.857 agreement against a 0.28 control
— so the frames and patterns are a thin shared map, while fillers belong
to whichever corpus was read. Compose with a domain's fillers and the
sentence speaks that domain: 場合を除く。規定に従う。責任を負う。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

#: Japanese is head-final and its case order is conventional rather than
#: free. This is the order a reader expects; the pattern decides WHICH of
#: these appear, never this list.
ORDER: Tuple[str, ...] = ("が", "に", "で", "から", "へ", "と", "を")

NO_VERB = "UNKNOWN_VERB_NOT_IN_FRAMES"
NO_PATTERN = "UNKNOWN_NO_OBSERVED_PATTERN"
NO_FILLER = "UNKNOWN_SLOT_UNFILLED"


@dataclass(frozen=True)
class Draft:
    """A composed sentence and everything it stands on."""

    text: str
    verb: str
    pattern: Tuple[str, ...]
    slots: Tuple[Tuple[str, str], ...]      # (case, noun)
    pattern_count: int
    constructed: bool = True

    def as_dict(self) -> Dict[str, Any]:
        return {"verdict": "DRAFT", "text": self.text, "verb": self.verb,
                "pattern": list(self.pattern),
                "slots": [{"case": c, "noun": n} for c, n in self.slots],
                "pattern_observed": self.pattern_count,
                "constructed": self.constructed,
                "note": "構成された文であり、証言ではない"}


class Tables:
    """Read-through access to frames / patterns / fillers.

    Backed by the meaning index by default so a composer never holds the
    12.6 MB of fillers resident — the lesson `meaning_index` was built for.
    A plain dict of the same shape works too, which is what lets a caller
    compose with a single document's fillers without indexing anything.
    """

    def __init__(self, conn=None, *,
                 frames: Optional[Dict[str, Any]] = None,
                 patterns: Optional[Dict[str, Any]] = None,
                 fillers: Optional[Dict[str, Any]] = None) -> None:
        self._conn = conn
        self._frames, self._patterns, self._fillers = frames, patterns, fillers

    @classmethod
    def indexed(cls) -> Optional["Tables"]:
        from .meaning_index import connection
        conn = connection()
        return cls(conn) if conn is not None else None

    def _get(self, table: str, key: str, override) -> Optional[Dict[str, Any]]:
        if override is not None:
            v = override.get(key)
            return v if isinstance(v, dict) else None
        if self._conn is None:
            return None
        row = self._conn.execute(
            "SELECT v FROM %s WHERE k = ?" % table, (key,)).fetchone()
        return json.loads(row[0]) if row else None

    def frame(self, verb: str):
        return self._get("frames", verb, self._frames)

    def pattern(self, verb: str):
        return self._get("patterns", verb, self._patterns)

    def filler(self, verb: str, case: str):
        return self._get("fillers", "%s\t%s" % (verb, case), self._fillers)


def _top_pattern(raw: Dict[str, Any]) -> Tuple[Tuple[str, ...], int]:
    best, n = (), 0
    for key, count in raw.items():
        if count > n:
            best, n = tuple(k for k in key.split("|") if k), int(count)
    return best, n


def compose(verb: str, tables: Tables, *,
            given: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """One clause for `verb`, or a typed refusal naming what was missing.

    `given` pins particular slots — the caller knows the subject, the
    tables know the shape. A pinned case that the observed pattern does
    not contain is added rather than dropped: the person asking for it is
    better evidence than the corpus's silence about it.
    """
    if tables.frame(verb) is None:
        return {"verdict": NO_VERB, "verb": verb,
                "note": "この動詞の枠が無い。推測した項構造は出さない"}

    raw = tables.pattern(verb)
    if not raw:
        return {"verdict": NO_PATTERN, "verb": verb,
                "note": "格が同時に現れた記録が無い。枠から閾値で組むと、"
                        "コーパスに一度も無い組み合わせを作る"}

    cases, seen = _top_pattern(raw)
    pinned = {k: v for k, v in (given or {}).items() if v}
    wanted = tuple(c for c in ORDER if c in set(cases) | set(pinned))

    slots: List[Tuple[str, str]] = []
    unfilled: List[str] = []
    for c in wanted:
        if c in pinned:
            slots.append((c, pinned[c]))
            continue
        f = tables.filler(verb, c)
        if not f:
            unfilled.append(c)
            continue
        slots.append((c, max(f.items(), key=lambda kv: kv[1])[0]))

    if not slots:
        return {"verdict": NO_FILLER, "verb": verb,
                "pattern": list(cases), "unfilled": unfilled,
                "note": "枠も型もあるが、この語彙で埋まる項が無い"}

    text = "".join(n + c for c, n in slots) + verb + "。"
    return Draft(text=text, verb=verb, pattern=cases, slots=tuple(slots),
                 pattern_count=seen).as_dict() | (
        {"unfilled": unfilled} if unfilled else {})
