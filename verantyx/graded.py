"""Quantize the VERDICT, not only the index.

The constellation varies grain across its members and reads the spread as
doubt. The verdict never did: one gate, whole terms, and a question one
character off the corpus refused with nothing said about how close it came.

The refusal is not hypothetical. 刑法第百九十九条 carries the facet 殺人 —
from its own caption, so the statute does name the offence — and the
doctrinal term is 殺人罪. One character apart, and the whole-grain gate
reports `query_terms_not_addressed:殺人罪` exactly as it reports a word the
corpus has never held. The two are not alike and nothing told them apart.

So the same three axes the sovereigns vary are turned on the judgment, over
CORES rather than leaves:

    grain     whole / 3 / 2         windows the term is matched through
    grammar   raw / nosuffix        傷害罪 -> 傷害, per `ja_morph`
    depth     how many facets of a core are indexed at all

`resolution.Ladder` already does this and abstains on ties; the first
version of this module reimplemented it by hand and reproduced every failure
the ladder was written to avoid. Scanning all 54,244 cores per query took
900ms, requiring every term to be covered and then preferring the core with
the most facets picked a junk hub — 「殺人罪の刑は」 came back 至 — and
coarsening answered 「超伝導とは」, a term the corpus does not hold, out of
a two-character collision. Reusing the calibrated machinery is the fix.

## Does coarsening generalize to a word the store does not hold

Measurably, partly, and by composition rather than by meaning. 120 cores
with eight or more facets were REMOVED from the index and then asked for.
57% were refused outright. Of the 43% answered, the core that came back
shared 19.0% of the removed core's facets against 1.6% for a random core of
the same richness — 12.1x, and only 12% landed with no overlap at all
against 84% at random.

Most of that survives the obvious confound. A held-out 電波法第二十九条 can
be answered by 電波法施行規則第二十九条, which shares facets because it
shares a name, not because anything was inferred. Splitting on a shared
three-character prefix: siblings 21.0%, genuinely different strings 17.9%,
still 11.4x over chance.

The mechanism is visible in what it returns. アバター lands on 人工知能ホロ
アバター and 火山活動 on 活動火山対策特別措置法 — Japanese is head-final and
its compounds are transparent, so a coarse window finds the compound that
CONTAINS the unheld term and inherits its subject. It also finds
鉱業法第百一条 for 漁業法第百一条, sharing a numeral and 0% of the facets.

So this reaches unseen WORDS and not unseen MEANINGS. 17.9% overlap is a
lead, not an answer, which is the whole reason a coarse reading is typed
`ANSWER_BY_COARSENING` and never promoted. The 0.0% generalization measured
on held-out terms elsewhere in this package is unchanged: nothing here
infers what a word means, it finds a longer word the corpus already held.

## Coarsening may add a reading and may never overturn one

A verdict that only a coarse member reached is typed `ANSWER_BY_COARSENING`
and carries the count. It is not promoted to `ANSWER`, and the strict
reading stays in the report for a caller that wants only that. What the
bands are worth on a given corpus is a measurement, not a property of this
code — see `calibrate`.
"""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .resolution import Ladder, ask as rung_ask

#: One member per SETTING, one axis at a time. The cross product would be a
#: bigger census and a worse experiment: when members disagree, nothing
#: would say which axis did it.
#: `depth=1` indexes a core by its NAME alone; without it every setting
#: abstains on a one-term question. A ladder scores one point per matching
#: term, so every core that merely MENTIONS 傷害罪 ties with the one that
#: IS it, and a tie abstains — correctly, for the question the ladder was
#: written for. "Which document mentions this" and "which core is this" are
#: different questions and the store already tells them apart: name against
#: facet. Preferring the name is not a tie-break, it is reading the
#: distinction the store records.
DEFAULT_SETTINGS: Tuple[Tuple[str, Dict[str, Any]], ...] = (
    ("whole",       {"rungs": (("whole", 0),), "grammar": "raw", "depth": 1}),
    ("g3",          {"rungs": (("g3", 3),), "grammar": "raw", "depth": 1}),
    ("g2",          {"rungs": (("g2", 2),), "grammar": "raw", "depth": 1}),
    ("nosuffix",    {"rungs": (("whole", 0),), "grammar": "nosuffix", "depth": 1}),
    ("nosuffix.g2", {"rungs": (("g2", 2),), "grammar": "nosuffix", "depth": 1}),
    ("mentions",    {"rungs": (("whole", 0),), "grammar": "raw"}),
)


def cores_as_items(store: Any, *, depth: Optional[int] = None) -> Dict[str, List[str]]:
    """core -> the terms that identify it: itself plus its facets.

    Source labels are dropped. A label is provenance, and leaving it in lets
    a question be answered by the name of the file it would be answered
    from — which is the document citing itself.
    """
    labels = getattr(store, "source_labels", set()) or set()
    items: Dict[str, List[str]] = {}
    for core, cross in store.crosses.items():
        if core in labels:
            continue
        terms = [core] + sorted(f for f in (cross or ()) if f not in labels)
        items[core] = terms[:depth] if depth else terms
    return items


class GradedJudge:
    """One ladder per setting, over the cores of a store."""

    def __init__(self, settings: Sequence[Tuple[str, Dict[str, Any]]] = DEFAULT_SETTINGS):
        self.settings = list(settings)
        self.ladders: Dict[str, Ladder] = {}

    def build(self, store: Any) -> "GradedJudge":
        cache: Dict[Optional[int], Dict[str, List[str]]] = {}
        for name, cfg in self.settings:
            depth = cfg.get("depth")
            if depth not in cache:
                cache[depth] = cores_as_items(store, depth=depth)
            self.ladders[name] = Ladder(
                rungs=cfg["rungs"], grammar=cfg.get("grammar", "raw"),
            ).build(cache[depth])
        return self

    def ask(self, query: str) -> Dict[str, Any]:
        """Every setting's reading, and the count. Never a promotion."""
        from .lang import ja_content_runs

        terms = ja_content_runs(query)
        if not terms:
            return {"verdict": "UNKNOWN_UNPARSED", "query": query}

        readings: Dict[str, Optional[str]] = {}
        for name, _cfg in self.settings:
            r = rung_ask(self.ladders[name], terms)
            readings[name] = r["item"] if r["verdict"] == "ANSWER" else None

        spoke = [v for v in readings.values() if v]
        if not spoke:
            return {"verdict": "UNKNOWN_NOT_PRESENT", "item": None,
                    "terms": terms, "agreeing": 0, "of": len(self.settings),
                    "readings": readings}
        tally = Counter(spoke)
        top = max(tally.values())
        leaders = sorted(k for k, v in tally.items() if v == top)
        if len(leaders) > 1:
            return {"verdict": "AMBIGUOUS", "item": None, "terms": terms,
                    "agreeing": top, "of": len(self.settings),
                    "leaders": leaders[:4], "readings": readings}
        item = leaders[0]
        strict = readings.get("whole")
        return {
            # Typed apart on purpose. A caller that wants the old contract
            # reads ANSWER and treats the other as a refusal.
            "verdict": "ANSWER" if strict == item else "ANSWER_BY_COARSENING",
            "item": item, "terms": terms,
            "agreeing": top, "of": len(self.settings),
            "spoke": len(spoke),
            "concord": round(top / len(spoke), 3),
            "readings": readings,
            "note": "a coarser setting may add a reading and may never "
                    "overturn a finer one",
        }
