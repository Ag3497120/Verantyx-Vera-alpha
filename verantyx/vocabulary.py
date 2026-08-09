"""Which facets are words — a separate layer, for a separate job.

A facet is whatever the reader cut out of a sentence. That is the right
thing for retrieval and the wrong thing for generation, and the gap is not
small: of 2,000 facets sampled from an 88,789-facet federation, 7.4% appear
three or more times as free-standing words in prose. エリミネーター,
コンキスタドール, 1980アイコ are real facets and none of them is a word a
sentence can use.

So generation needs a vocabulary, and a vocabulary is a different structure
from an index:

    index        every string that identifies something. 88,789 entries,
                 answers retrieval at 100% when the ladder speaks
    vocabulary   the subset attested as words, with the corpus that attested
                 them and how often. Feeds composition; never feeds a verdict

## Attested, not judged

A word here is one that occurs NOT flanked by further kanji, at least
`MIN_ATTEST` times, in a corpus that did not produce the facet. That test
already earned its place in `granularity`, where it separated real coinages
(自動, 定理, 人権) from substrings (事訴 inside 民事訴訟法, 法上 inside
憲法上) and raised the advantage over chance from 4.9x to 15x by dropping
the fragments.

Nothing here decides what a word MEANS, and nothing marks a facet wrong for
failing — 1980アイコ is a perfectly good retrieval key. It is only unfit for
one job, and this layer records which.

## Why the attesting corpus must be named

A vocabulary is a claim about usage, so it inherits the usage it was
measured on. Statutes attest 拘禁刑 and not コンキスタドール; an
encyclopedia attests the reverse. Recording which corpus attested each term
is what stops a vocabulary built from one register being used to judge
another — the same discipline that keeps `placement` from carrying one
domain's demand into a different one.
"""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

#: Occurrences unflanked by further kanji before a facet counts as a word.
#: One is a line break or a typo; three is a usage.
MIN_ATTEST = 3

#: Below and above this, a run is not a lexical item worth composing with.
MIN_LEN, MAX_LEN = 2, 12

_FLANK = "㐀-䶿一-鿿"
_NUMERISH = re.compile(r"[0-9０-９]")


@dataclass
class Vocabulary:
    """Terms attested as free-standing words, and who attested them."""

    #: term -> corpus label -> standalone occurrences
    attested: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def add(self, term: str, corpus: str, n: int) -> None:
        self.attested.setdefault(term, {})[corpus] = n

    def __contains__(self, term: str) -> bool:
        return term in self.attested

    def support(self, term: str) -> int:
        return sum((self.attested.get(term) or {}).values())

    def sources(self, term: str) -> List[str]:
        return sorted(self.attested.get(term) or ())

    def report(self) -> Dict[str, Any]:
        by: Counter = Counter()
        for v in self.attested.values():
            for c in v:
                by[c] += 1
        return {"terms": len(self.attested), "by_corpus": dict(by.most_common())}

    def save(self, path: Path) -> None:
        Path(path).write_text(
            json.dumps(self.attested, ensure_ascii=False, sort_keys=True),
            encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Vocabulary":
        v = cls()
        v.attested = json.loads(Path(path).read_text(encoding="utf-8"))
        return v


def standalone(term: str, text: str) -> int:
    """How often ``term`` appears not flanked by further kanji.

    The flanking test is the whole filter: 事訴 occurs thousands of times
    inside 民事訴訟法 and never on its own.
    """
    return len(re.findall(
        f"(?<![{_FLANK}])" + re.escape(term) + f"(?![{_FLANK}])", text))


#: A maximal run of the characters a Japanese content word is made of.
#: A term "stands alone" exactly when it IS one of these, so counting runs
#: once answers the question for every candidate at the same time.
_RUN = re.compile(r"[㐀-䶿一-鿿ァ-ヺー々〆]+")


def runs(text: str) -> Counter:
    """Maximal content runs and their counts. One pass over the corpus."""
    c: Counter = Counter()
    for m in _RUN.finditer(text or ""):
        c[m.group(0)] += 1
    return c


def attest(
    candidates: Iterable[str],
    corpora: Sequence[Tuple[str, str]],
    *,
    min_attest: int = MIN_ATTEST,
) -> Vocabulary:
    """Which candidates a corpus uses as words. ``corpora`` is (label, text).

    Counts maximal runs once per corpus rather than searching per candidate.
    The first version did the latter — a flanking regex for every candidate
    against every corpus — and did not finish 20,000 candidates against 3MB
    in ten minutes. The two are equivalent by construction: a term is
    unflanked exactly when it is a maximal run.
    """
    vocab = Vocabulary()
    keep = {c for c in candidates
            if c and MIN_LEN <= len(c) <= MAX_LEN and not _NUMERISH.search(c)}
    for label, text in corpora:
        counts = runs(text)
        for term in keep:
            n = counts.get(term, 0)
            if n >= min_attest:
                vocab.add(term, label, n)
    return vocab


def from_stores(
    stores: Iterable[Any],
    corpora: Sequence[Tuple[str, str]],
    *,
    min_attest: int = MIN_ATTEST,
    limit: Optional[int] = None,
) -> Vocabulary:
    """Sift a federation's facets into the ones that are words."""
    facets: Counter = Counter()
    for st in stores:
        labels = getattr(st, "source_labels", set()) or set()
        for cross in st.crosses.values():
            for f in cross:
                if f not in labels:
                    facets[f] += 1
    ranked = [t for t, _n in facets.most_common(limit)] if limit else list(facets)
    return attest(ranked, corpora, min_attest=min_attest)


def filter_terms(
    terms: Iterable[str],
    vocab: Vocabulary,
    *,
    min_support: int = 1,
) -> List[str]:
    """The terms of ``terms`` that are words, best-attested first.

    Order is by attestation then alphabetical, so a caller taking the first
    fill gets the most-used word rather than the alphabetically luckiest —
    which is what put 1980アイコ in a sentence before this existed.
    """
    got = [(vocab.support(t), t) for t in terms if vocab.support(t) >= min_support]
    got.sort(key=lambda st: (-st[0], st[1]))
    return [t for _s, t in got]
