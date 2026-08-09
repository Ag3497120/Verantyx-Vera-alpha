"""A ladder of resolutions over one corpus — agreement as calibrated doubt.

The same text read at several grain sizes gives several readings, and how
many of them agree is a confidence signal that no single reading carries.
Measured on 10,222 statute articles, asked by their own captions, 600
probes:

    best single rung (g2), answering everything      29.8%

    four rungs answer, all agree      39 probes         100.0%
    three rungs answer, all agree     38                100.0%
    one rung answers alone            36                 75.0%
    two rungs answer, both agree     103                 31.1%
    no rung had grounds              383                  --

Unanimity among three or four rungs was right every time, and 64% of the
probes got no answer at all. That is the trade this makes: a deterministic
system cannot report a probability, but it can report how much of its own
structure concurred, and refuse when the answer would rest on nothing.

## Ties are ABSTENTIONS, and getting that wrong twice is the story

Breaking ties by insertion order made a rung's answer depend on which
article was ingested first. Breaking them lexicographically instead was
WORSE, and the measurement says so plainly: every tied rung then chose the
same smallest item, unanimity rose from 86 probes to 321 and its accuracy
fell from 73.3% to 23.7% — below the three-to-one majority. Determinism at
the tie manufactures agreement, because the rungs concur for a reason that
has nothing to do with evidence.

A rung with no single leader abstains. Unanimity then means what it says:
every rung that had grounds said the same thing.

## The rungs must be NESTED, not partitioned

The first attempt binned each term by its length: one-character terms to q1,
two to q2, and so on. That is a partition — a term lands in exactly one
rung, so no two rungs ever see it, and 507 of 600 probes had a single rung
answer at all. Agreement was not low; it was impossible.

A rung is a GRAIN SIZE applied to every term. 損害賠償 is one item at the
word rung, 損害/害賠/賠償 at the two-character rung, and four items at the
character rung. Every rung sees every term, at its own resolution — which is
what the quantisation analogy actually means.

## What it does not do

It does not make a wrong answer right. The ladder reranks nothing: each rung
votes with what it indexed, and the report says how much they concurred. An
answer all four rungs agree on is still wrong a quarter of the time here,
and the number is printed rather than rounded away.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

#: Grain sizes, coarse to fine. 0 means "the whole term, uncut".
DEFAULT_RUNGS: Tuple[Tuple[str, int], ...] = (
    ("whole", 0), ("g3", 3), ("g2", 2), ("g1", 1),
)


def grains(term: str, size: int) -> List[str]:
    """One term seen at one grain size.

    Sliding, not chunked: 損害賠償 at size 2 gives 損害/害賠/賠償, so a
    boundary the corpus draws differently is still covered. Chunking would
    make the rung sensitive to where a compound happens to start.
    """
    t = term or ""
    if size <= 0 or len(t) <= size:
        return [t] if t else []
    return [t[i:i + size] for i in range(len(t) - size + 1)]


@dataclass
class Ladder:
    """rung -> grain -> the items that contain it."""

    rungs: Tuple[Tuple[str, int], ...] = DEFAULT_RUNGS
    index: Dict[str, Dict[str, Set[str]]] = field(default_factory=dict)

    def build(self, items: Dict[str, Iterable[str]]) -> "Ladder":
        """``items`` maps an id (an article, a leaf) to the terms it holds."""
        self.index = {name: defaultdict(set) for name, _ in self.rungs}
        for item, terms in items.items():
            for term in terms:
                for name, size in self.rungs:
                    for g in grains(term, size):
                        self.index[name][g].add(item)
        return self

    def report(self) -> Dict[str, Any]:
        return {"rungs": [n for n, _ in self.rungs],
                "grains_per_rung": {n: len(self.index.get(n, {}))
                                    for n, _ in self.rungs}}

    def vote(self, query_terms: Sequence[str]) -> Dict[str, Optional[str]]:
        """Each rung's own best item for this query, or None if it has none."""
        out: Dict[str, Optional[str]] = {}
        for name, size in self.rungs:
            score: Counter = Counter()
            for term in query_terms:
                for g in grains(term, size):
                    for item in self.index.get(name, {}).get(g, ()):
                        score[item] += 1
            # A tied rung ABSTAINS. It does not pick.
            #
            # most_common breaks ties by insertion order, so which article
            # was ingested first decided what a rung said. Breaking them
            # lexicographically instead was worse: every tied rung then
            # chose the same smallest item, unanimity rose from 86 probes to
            # 321 and its accuracy fell from 73.3% to 23.7% — below the 3-1
            # majority. Determinism at the tie MANUFACTURES agreement,
            # because the rungs concur for a reason unrelated to evidence.
            #
            # Abstaining keeps unanimity meaning what it claims: every rung
            # that had grounds said the same thing.
            if not score:
                out[name] = None
                continue
            best = max(score.values())
            leaders = [k for k, v in score.items() if v == best]
            out[name] = leaders[0] if len(leaders) == 1 else None
        return out


def ask(ladder: Ladder, query_terms: Sequence[str]) -> Dict[str, Any]:
    """The ladder's answer, with how much of it concurred.

    ``concord`` is answered-and-agreeing over answered — the fraction of
    rungs that could speak and said the same thing. It is a measurement of
    this structure, not a probability of being right, and the two are only
    related by the calibration table in the module docstring.
    """
    votes = ladder.vote(query_terms)
    answered = [v for v in votes.values() if v]
    if not answered:
        return {"verdict": "UNKNOWN_NOT_PRESENT", "item": None,
                "votes": votes, "answered": 0, "distinct": 0,
                "majority": 0, "concord": 0.0}
    tally = Counter(answered)
    # Ties are broken lexicographically, not by insertion order. Counter's
    # most_common keeps first-seen order among equal counts, so simply
    # reordering the rungs changed the answer on every split vote — the same
    # arbitrary tie-break that placement was measured to fabricate through.
    top = max(tally.values())
    item = sorted(k for k, v in tally.items() if v == top)[0]
    return {
        # A 2-2 split has no majority at all, and calling it AMBIGUOUS
        # separates it from 3-1, which does. Reporting only `distinct`
        # merged the two and hid that the tie-break was deciding.
        "verdict": "ANSWER" if len(tally) == 1 else
                   ("MAJORITY" if top * 2 > len(answered) else "AMBIGUOUS"),
        "item": item,
        "votes": votes,
        "answered": len(answered),
        "distinct": len(tally),
        "majority": top,
        "concord": round(top / len(answered), 3),
    }


def calibrate(
    ladder: Ladder,
    probes: Sequence[Tuple[str, Sequence[str]]],
) -> Dict[str, Any]:
    """Group probes by how the rungs split, and report accuracy per group.

    This is the table that gives `concord` its meaning, and it has to be
    rebuilt per corpus: 74.2% at full agreement is a fact about statute
    captions over these articles, not a constant.
    """
    buckets: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    for want, terms in probes:
        r = ask(ladder, terms)
        if not r["answered"]:
            buckets["none answered"][0] += 1
            continue
        key = (f"{r['answered']}答 / 最大派{r['majority']} "
               f"[{r['verdict'][:8]}]")
        buckets[key][0] += 1
        buckets[key][1] += int(r["item"] == want)
    return {
        "probes": len(probes),
        "buckets": {k: {"n": v[0], "correct": v[1],
                        "accuracy": round(v[1] / v[0], 4) if v[0] else 0.0}
                    for k, v in sorted(buckets.items())},
    }
