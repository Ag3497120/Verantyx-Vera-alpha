"""Where independently-built fields meet — and how much a change moves it.

Each field is modelled on its own: its own store, its own geometry, its own
placement. Federating them does not merge those models, and it is not
supposed to. What federation creates is a set of POINTS where two or three
fields turn out to be talking about the same thing:

    家庭裁判所   刑事 38 entities ｜ 民事 54     one institution, two procedures
    存続期間     民事 15          ｜ 知財 26     a lease term and a patent term
    救助         民事 15          ｜ 防災 20     salvage at sea and disaster relief
    関数         数学 68          ｜ 経済 10

Those are the joins. A question that arrives at one of them can be answered
from either side, and the two sides can disagree — which is the thing this
whole package exists to surface.

## The band, and why it is not a tuning knob

A concept present in EVERY field carries no field. Measured over twelve
fields (17,407 concepts, 9,370 cores): 18.7% span more than one field, and
the ones spanning all twelve are 作成, 必要, 定義, 目的, 規定, 情報 —
function words of legal and technical prose. Reading them as fusion would
say every field is fused with every other, which is true and useless.

The informative band is small: concepts held by BAND_LOW..BAND_HIGH fields.
Two fields sharing a term is a bridge; twelve fields sharing it is grammar.

## One field per SOURCE, or the measurement reads the prose style

Built at scale the first time with twenty-five fields — statutes split by
legal area, and ja.wikipedia split three ways by category — the largest
join was 全_文学 x 引用_法 at 202 concepts. Literature and jurisprudence had
not converged; both slices were Wikipedia, and what they shared was its
prose. Slicing one source into several fields makes the source's own style
arrive as fusion.

Re-cut so that each SOURCE is one field (e-Gov statutes by law, all of
ja.wikipedia as one), on nearly the same leaves:

    25 fields, source mixed   1,223 joins   top pair 全_文学 x 引用_法  202
     9 fields, one per source   858 joins   top pair 法令_民事 x 百科   240

Fewer joins and every top pair is now primary-source-against-commentary,
which is a relation that exists. Abstention in the resolution ladder fell
too, 24.9% to 20.3%: leaves cut from one source were more alike than the
field labels claimed.

## Why a baseline is computed before anything changes

The number that matters is not the fusion index — it is the DELTA when a
document arrives. An amendment that connects two fields which were separate,
or one that dissolves a bridge, is a structural event, and it is only
visible against a figure taken beforehand. That is what makes this a
pre-simulation rather than a report: `index` is run at build time so
`delta` has something to subtract from.

Nothing here merges stores or writes to a field. A fusion point is an
observation ABOUT two models, held outside both.
"""
from __future__ import annotations

import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from .cross_store import CrossStore
from .lex_filters import is_junk_core

#: Fields a concept may span and still say something about which field it is.
#: The upper bound is the load-bearing one — see the module docstring.
BAND_LOW, BAND_HIGH = 2, 3

#: Both sides must actually talk about it. A bridge one entity wide is a
#: coincidence of vocabulary, not a place two models meet.
MIN_ENTITIES = 2

_NUMERAL = re.compile(r"^[〇一二三四五六七八九十百千万0-9０-９]+$")


def usable(term: str) -> bool:
    """Is this a concept, or is it bookkeeping?

    Article enumerations (一, 十二, 第三) dominate any raw count over
    statutes — they were 85% of the cross-field cores the first pass found,
    and none of them meant anything.
    """
    t = str(term)
    return (2 <= len(t) <= 10
            and not _NUMERAL.match(t)
            and not is_junk_core(t)
            and not t.endswith("txt"))


@dataclass
class Point:
    """One concept, and which entity of each field carries it."""

    concept: str
    by_field: Dict[str, Set[str]] = field(default_factory=dict)

    @property
    def fields(self) -> List[str]:
        return sorted(self.by_field)

    @property
    def width(self) -> int:
        """The thinner side. A bridge is only as wide as its narrow end."""
        return min((len(v) for v in self.by_field.values()), default=0)

    @property
    def mass(self) -> int:
        return sum(len(v) for v in self.by_field.values())

    def as_dict(self) -> Dict[str, Any]:
        return {"concept": self.concept, "fields": self.fields,
                "width": self.width, "mass": self.mass,
                "entities": {d: sorted(v)[:4] for d, v in sorted(self.by_field.items())}}


def concept_map(fields: Dict[str, Dict[str, CrossStore]]) -> Dict[str, Dict[str, Set[str]]]:
    """concept -> field -> the entities of that field which carry it."""
    holds: Dict[str, Dict[str, Set[str]]] = defaultdict(lambda: defaultdict(set))
    for name, leaves in fields.items():
        for store in leaves.values():
            labels = getattr(store, "source_labels", set()) or set()
            for core, cross in store.crosses.items():
                for facet in cross:
                    if facet in labels or not usable(facet):
                        continue
                    holds[facet][name].add(core)
    return {c: dict(v) for c, v in holds.items()}


def points(
    fields: Dict[str, Dict[str, CrossStore]],
    *,
    band: Tuple[int, int] = (BAND_LOW, BAND_HIGH),
    min_entities: int = MIN_ENTITIES,
    only: Optional[Iterable[str]] = None,
) -> List[Point]:
    """The places two or three fields turn out to mean the same thing."""
    keep = set(only) if only else None
    out: List[Point] = []
    for concept, by in concept_map(fields).items():
        if keep is not None:
            by = {d: e for d, e in by.items() if d in keep}
        if not (band[0] <= len(by) <= band[1]):
            continue
        p = Point(concept=concept, by_field=by)
        if p.width < min_entities:
            continue
        out.append(p)
    out.sort(key=lambda p: (-p.width, -p.mass, p.concept))
    return out


def index(
    fields: Dict[str, Dict[str, CrossStore]],
    **kw: Any,
) -> Dict[str, Any]:
    """The baseline. Run at build time so `delta` has something to subtract.

    Reports the pair census as well as the total, because "these two fields
    became connected" is the event worth noticing and a single number hides
    which pair moved.
    """
    pts = points(fields, **kw)
    pair: Dict[str, int] = defaultdict(int)
    for p in pts:
        fs = p.fields
        for i in range(len(fs)):
            for j in range(i + 1, len(fs)):
                pair[f"{fs[i]}×{fs[j]}"] += 1
    return {
        "fields": sorted(fields),
        "n_points": len(pts),
        "total_mass": sum(p.mass for p in pts),
        "pairs": dict(sorted(pair.items(), key=lambda kv: (-kv[1], kv[0]))),
        "concepts": {p.concept: p.width for p in pts},
        "top": [p.as_dict() for p in pts[:20]],
    }


def delta(before: Dict[str, Any], after: Dict[str, Any]) -> Dict[str, Any]:
    """What an external change did to the joins.

    Three outcomes, and the first two are what a reader wants:

      opened   a concept that now bridges fields and did not before
      closed   a bridge that stopped being one
      widened  a bridge whose narrow end grew

    A change that only adds entities to existing bridges is reported as
    widening rather than as new fusion, because a field learning more about
    something it already shared is not a new connection.
    """
    b, a = before.get("concepts", {}), after.get("concepts", {})
    opened = sorted(set(a) - set(b))
    closed = sorted(set(b) - set(a))
    widened = sorted(c for c in (set(a) & set(b)) if a[c] > b[c])
    narrowed = sorted(c for c in (set(a) & set(b)) if a[c] < b[c])
    pb, pa = before.get("pairs", {}), after.get("pairs", {})
    pair_delta = {k: pa.get(k, 0) - pb.get(k, 0)
                  for k in set(pb) | set(pa) if pa.get(k, 0) != pb.get(k, 0)}
    return {
        "verdict": "CHANGED" if (opened or closed or widened or narrowed)
                   else "UNCHANGED",
        "n_points": {"before": before.get("n_points", 0),
                     "after": after.get("n_points", 0)},
        "opened": opened, "closed": closed,
        "widened": widened, "narrowed": narrowed,
        "pairs_moved": dict(sorted(pair_delta.items(),
                                   key=lambda kv: (-abs(kv[1]), kv[0]))),
    }


def read_at(
    fields: Dict[str, Dict[str, CrossStore]],
    concept: str,
    *,
    limit: int = 6,
) -> Dict[str, Any]:
    """What each side of a join says, side by side, without merging them.

    The point of keeping the models apart is that this can show a
    disagreement instead of averaging it away. Nothing here chooses.
    """
    from .consensus_store import ja_consensus_ask

    out: Dict[str, List[Dict[str, Any]]] = {}
    for name, leaves in fields.items():
        rows: List[Dict[str, Any]] = []
        for leaf, store in leaves.items():
            if not any(concept in c for c in store.crosses.values()):
                continue
            res = ja_consensus_ask(store, concept)
            rows.append({"leaf": leaf, "verdict": res.get("verdict"),
                         "core": res.get("core"), "text": res.get("text", "")})
            if len(rows) >= limit:
                break
        if rows:
            out[name] = rows
    return {"concept": concept, "fields": sorted(out), "by_field": out}
