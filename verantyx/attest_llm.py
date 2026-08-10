"""Check a sentence somebody else wrote against what the store holds.

The architecture this exists for puts an LLM in the generation layer and
Vera underneath it as verification, citation and structure. That only means
anything if the verification actually catches an unsupported sentence — a
citation layer that passes everything is decoration, and a fluent wrong
answer with a citation stapled to it is worse than a fluent wrong answer.

So this scores a sentence it did not write, against a SUBJECT:

    terms    content runs the sentence uses
    linked   the ones the subject's own cross holds
    support  linked / terms

## Presence is not support, and measuring presence fails backwards

The first version asked whether the corpus held each term anywhere, and
whether adjacent pairs were crossed. Measured against a local 4B model over
14 subjects it ranked FREE generation above grounded generation — 95.7%
term presence against 85.5%, 25.3% pair support against 15.0% — because a
fluent answer about Japanese law is built from 法律, 制定, 原則, 国民, and a
federation of 54,244 legal cores holds every one of them. In a large corpus
presence is nearly free, so a checker built on it passes everything.

What is not free is the link to the SUBJECT. Asked about 第37条 the model
wrote 「国家の権限を保障し、国家が法律を制定する権利を確認する条文である」 —
fluent, plausible, and sharing nothing with what the store records under
第37条. Same 14 subjects, scored that way:

    grounded   64.1% of terms in the subject's cross
    free        6.4%

    at a 30% threshold: 0 of 14 grounded flagged, 14 of 14 free flagged

## What it can and cannot say

It says "this corpus does not support that", never "that is false". 超伝導
is absent from a federation of Japanese statute and encyclopedia articles
and is not thereby untrue. The distinction is the whole point of putting a
store under a generator rather than asking the generator to grade itself:
one of them can be wrong about the world and the other can only be wrong
about the corpus, and the second failure is the one a reader can check.

A sentence that passes is not verified true. It is verified to be about
things this corpus holds, joined the way this corpus joins them — which is
exactly the claim a citation makes and no more.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

_RUN = re.compile(r"[㐀-䶿一-鿿ァ-ヺー々〆]{2,}")


@dataclass
class Report:
    subject: str
    sentence: str
    terms: List[str] = field(default_factory=list)
    linked: List[str] = field(default_factory=list)
    unlinked: List[str] = field(default_factory=list)

    @property
    def support(self) -> float:
        return len(self.linked) / max(len(self.terms), 1)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "subject": self.subject, "sentence": self.sentence,
            "terms": len(self.terms), "support": round(self.support, 3),
            "linked": self.linked[:8], "unlinked": self.unlinked[:8],
            "note": "unsupported by THIS corpus for THIS subject; "
                    "not a claim about truth",
        }


#: Below this, the sentence is about something other than what the store
#: records under the subject. Measured, not chosen: grounded output does not
#: reach down to it and free output does not reach up to it.
THRESHOLD = 0.30


def check(store: Any, subject: str, sentence: str) -> Report:
    labels = getattr(store, "source_labels", set()) or set()
    cross = {f for f in (store.crosses.get(subject) or {}) if f not in labels}
    terms: List[str] = []
    for t in _RUN.findall(sentence or ""):
        if t != subject and t not in terms and t not in labels:
            terms.append(t)
    r = Report(subject=subject, sentence=sentence, terms=terms)
    for t in terms:
        (r.linked if t in cross else r.unlinked).append(t)
    return r


def check_all(store: Any, subject: str, text: str) -> Dict[str, Any]:
    """Split on 。 and score every sentence against the subject."""
    sents = [s.strip() for s in re.split(r"[。\n]", text or "") if s.strip()]
    reps = [check(store, subject, s) for s in sents]
    reps = [r for r in reps if r.terms]
    if not reps:
        return {"verdict": "UNKNOWN_EMPTY", "sentences": 0}
    sup = sum(r.support for r in reps) / len(reps)
    return {
        # The typed refusal is the point. A generation layer that gets
        # UNSUPPORTED back has been told something a fluent draft cannot
        # tell it, and a reader who sees ANSWER has a link to check.
        "verdict": "ANSWER" if sup >= THRESHOLD else "UNSUPPORTED_BY_CORPUS",
        "subject": subject, "sentences": len(reps),
        "support": round(sup, 3), "threshold": THRESHOLD,
        "reports": [r.as_dict() for r in reps],
    }
