"""Documents in, disagreement out — multi-source ingestion for deep search.

The use case this exists for: several articles about one event, and the
question "what is actually going on". An LLM summarises them into one fluent
story, and the disagreement between sources dissolves into that fluency —
which is exactly the information a person in a disaster needs most. Here the
disagreement is PRESERVED, because each source's claims land on their own
poles of the same aspect and the store's contradiction detection fires on
its own.

The output is three separated things, never blended:

    settled    every source that spoke agrees
    disputed   sources disagree — with WHICH source said WHICH side
    missing    no source answered a question the arms say should have one

`missing` is the part that turns this into DEEP search rather than
summarisation: an unanswered arm is a typed gap, and a typed gap is a search
query for the next round ("nobody wrote why — go find why"). That loop is
not invented here; it is the acquisition loop the GapGraph already runs,
pointed at news instead of at the system's own failures.

No LLM is used anywhere in this file. Sentence splitting, polarity, and arm
assignment are all deterministic, so the same documents always produce the
same report — which is what makes a disputed claim citable.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .arm_schema import ArmIndex, classify_arm
from .cross_store import CrossStore
from .polarity import ANTONYM_PAIRS, detect, ingest_polar

_SENT = re.compile(r"(?<=[.!?。])\s+")
#: Sentences shorter than this carry no claim worth storing (headlines
#: fragments, bylines). Cheap filter, tuned to be permissive.
_MIN_SENT_CHARS = 12


@dataclass
class Document:
    source: str           # citable label: outlet, URL, agency
    text: str
    published: str = ""   # free-form; kept for display, never parsed


@dataclass
class IngestReport:
    documents: int = 0
    sentences: int = 0
    cores: List[str] = field(default_factory=list)
    polar_claims: int = 0
    per_source: Dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {"documents": self.documents, "sentences": self.sentences,
                "cores": sorted(set(self.cores)), "polar_claims": self.polar_claims,
                "per_source": self.per_source}


def ingest_documents(store: CrossStore, docs: List[Document],
                     arms: Optional[ArmIndex] = None) -> IngestReport:
    """Split each document into sentences and place them with source and
    polarity attached.

    Source attribution is appended to the sentence text so it reaches the
    store's own provenance (which records the originating snippet per
    facet). That is deliberately not a parallel bookkeeping structure: a
    citation that lives somewhere other than the evidence tends to drift
    away from it.
    """
    rep = IngestReport()
    for doc in docs:
        rep.documents += 1
        count = 0
        for raw in _SENT.split(doc.text or ""):
            s = raw.strip()
            if len(s) < _MIN_SENT_CHARS:
                continue
            tagged = f"{s} (reported by {doc.source})"
            core = ingest_polar(store, tagged)
            if core is None:
                continue
            count += 1
            rep.sentences += 1
            rep.cores.append(core)
            if detect(s):
                rep.polar_claims += 1
            if arms is not None and classify_arm(s):
                arms.arms.setdefault(core, {}).setdefault(
                    classify_arm(s), []).append(f"{s[:180]} — {doc.source}")
        rep.per_source[doc.source] = count
    return rep


def _sources_for(store: CrossStore, core: str, facet: str) -> List[str]:
    """Which reports backed one facet, read out of the store's provenance."""
    prov = store.provenance.get(core, {}) if store.track_provenance else {}
    slot = prov.get(facet)
    if not slot or len(slot) < 3:
        return []
    m = re.search(r"reported by ([^)]+)\)", str(slot[2]))
    return [m.group(1)] if m else []


def deep_report(store: CrossStore, core: str,
                arms: Optional[ArmIndex] = None) -> Dict[str, Any]:
    """Settled / disputed / missing for one topic.

    The three lists are the answer to "what is going on", kept apart on
    purpose. Blending them is what a summary does and what makes a summary
    unusable for a decision: the reader cannot tell which parts are agreed,
    which are contested, and which nobody checked.
    """
    aspect_keys = {p for p, _ in ANTONYM_PAIRS}
    disputed: List[Dict[str, Any]] = []
    for entry in store.contradictions(core):
        if entry["key"] not in aspect_keys:
            continue
        sides = []
        for value in entry["values"]:
            sides.append({
                "claim": value.split(":", 1)[1],
                "weight": entry["counts"].get(value, 0),
                "sources": _sources_for(store, core, value),
            })
        disputed.append({"aspect": entry["key"], "sides": sides})

    # Both the keyed form (open:closed) and the bare word (closed) must be
    # excluded. The bare word is also an ordinary facet of the sentence, so
    # without this a contested claim appears in BOTH lists — settled AND
    # disputed — which is the one thing this report exists to prevent.
    disputed_values = set()
    for e in store.contradictions(core):
        if e["key"] not in aspect_keys:
            continue
        for v in e["values"]:
            disputed_values.add(v)
            disputed_values.add(v.split(":", 1)[1])
            disputed_values.add(e["key"])
    # The attribution suffix is part of the ingested text, so its own words
    # become facets too ("city", "office", "reported"). They are provenance,
    # not claims, and a settled-facts list that includes the newspaper's name
    # as a fact about the shelter is worse than useless to a responder.
    attribution_words = {"reported", "by"} | _attribution_vocabulary(store, core)

    settled = []
    for facet, count in store.top_facets(core, k=12):
        if facet in disputed_values or ":" in facet:
            continue
        if facet in attribution_words:
            continue
        settled.append({"claim": facet, "weight": count,
                        "sources": _sources_for(store, core, facet)})

    missing: List[Dict[str, str]] = []
    if arms is not None:
        report = arms.report(core)
        for arm, verdict in zip(
            [a for a in report["empty"]], report["gap_verdicts"]
        ):
            missing.append({"arm": arm, "verdict": verdict,
                            "next_query": _gap_query(core, arm)})

    return {
        "core": core,
        "settled": settled,
        "disputed": disputed,
        "missing": missing,
        # The headline number a responder actually reads first.
        "confidence": ("contested" if disputed
                       else "supported" if settled else "unknown"),
    }


def _attribution_vocabulary(store: CrossStore, core: str) -> set:
    """Words that entered only through the "(reported by X)" suffix.

    Read back from provenance rather than guessed: every source label seen
    for this core is split into its words, and those words are excluded from
    the settled list. Doing it from the store's own record means a new
    outlet name never needs adding to a hardcoded list.
    """
    out: set = set()
    if not store.track_provenance:
        return out
    for slot in (store.provenance.get(core, {}) or {}).values():
        if not slot or len(slot) < 3:
            continue
        m = re.search(r"reported by ([^)]+)\)", str(slot[2]))
        if m:
            out |= {w for w in re.split(r"[^a-z0-9]+", m.group(1).lower()) if w}
    return out


_ARM_QUESTION = {
    "cause+": "why",
    "cause-": "what happens because of",
    "support+": "what evidence confirms",
    "support-": "what contradicts",
    "kind-": "an example of",
    "kind+": "what kind of thing is",
}


def _gap_query(core: str, arm: str) -> str:
    """A typed gap turned back into a search string — the step that makes
    this deep search instead of one-shot summarisation."""
    return f"{_ARM_QUESTION.get(arm, 'about')} {core}"
