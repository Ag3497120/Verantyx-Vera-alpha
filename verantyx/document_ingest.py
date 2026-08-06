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
from .polarity import (ANTONYM_PAIRS, ANTONYM_PAIRS_JA, detect,
                       detect_ja, ingest_polar, ingest_polar_ja)

#: Japanese does not put a space after 。, so a splitter that requires
#: trailing whitespace treats a whole article as one sentence — and then the
#: minimum-length filter and the English decomposer between them dropped it
#: entirely. Measured before this was fixed: a two-source Japanese corpus
#: ingested as zero sentences, silently, which is the worst way for the
#: disaster-information path to fail given who needs it.
_SENT = re.compile(r"(?<=[.!?。！？])\s*")

_CJK = re.compile(r"[぀-ヿ㐀-䶿一-鿿]")

#: Sentence-length floor, by script. Twelve characters of Latin text is about
#: two words and carries nothing; twelve characters of Japanese is a full
#: claim — 「避難所は開いています」 is eleven. Holding both to one number is
#: how the Japanese path lost its content.
_MIN_SENT_CHARS = 12
_MIN_SENT_CHARS_CJK = 6


def _min_chars(text: str) -> int:
    return _MIN_SENT_CHARS_CJK if _CJK.search(text) else _MIN_SENT_CHARS


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
    #: core → {lang: sentence count}. Kept per core rather than per document
    #: because one document can mix languages (this corpus does), and the
    #: catalogue's question is "what language is this TOPIC discussed in".
    core_lang: Dict[str, Dict[str, int]] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, Any]:
        return {"documents": self.documents, "sentences": self.sentences,
                "cores": sorted(set(self.cores)), "polar_claims": self.polar_claims,
                "per_source": self.per_source, "core_lang": self.core_lang}


def _place(store: CrossStore, sentence: str,
           detect_on: Optional[str] = None) -> tuple:
    """Put one sentence in the store, using the right language's segmenter.

    `ingest_polar` goes through CrossStore.ingest_sentence, whose decomposer
    is `en_decompose` — its word pattern is `[A-Za-z0-9']+`, so a Japanese
    sentence yields no words at all and the whole string falls through as one
    core with zero facets. Nothing then links, contradicts, or answers.

    The Japanese segmenter already existed in `lang`; it simply was not on
    this path. Routing by script is the whole fix. Polarity still runs on the
    English path only, so a Japanese corpus gets cores and facets but not yet
    the open/closed contradiction pairs — stated here rather than discovered
    later, because deep_report's `disputed` list will be thinner in Japanese
    and that is a limit, not a finding about the sources.
    """
    from .lang import detect, ja_ingest_sentence

    # Detect on the ORIGINAL sentence, not the one carrying the attribution
    # suffix. `detect` compares Japanese characters against Latin ones, and
    # "(reported by f.docx)" adds enough Latin to outvote a short Japanese
    # sentence — which routed it to the English decomposer and turned the
    # whole sentence into a single core with no facets. The suffix is
    # bookkeeping; it must not get a vote on what language the claim is in.
    lang = detect(detect_on or sentence)
    if lang == "ja":
        return ingest_polar_ja(store, sentence), "ja"
    if lang == "zh":
        # Chinese gets segmentation (the ideograph-run scanner is script-,
        # not language-specific) but NOT the Japanese polarity pass: the
        # shared kanji make terms like 安全 match, while the negation and
        # predicate grammar around them is a different language's. Facts
        # without poles is the honest depth here — the alternative, routing
        # to the English decomposer, dropped Chinese sentences entirely,
        # which is the same silent-zero failure Japanese had.
        return ja_ingest_sentence(store, sentence), "zh"
    return ingest_polar(store, sentence), "en"


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
    # Provenance defaults to off on CrossStore, which is fine for a store
    # built from one source. Here it is the whole contract: "which report
    # backed this claim" is what separates this from summarisation, and with
    # tracking off every `sources` list came back empty — indistinguishable
    # from sources that simply had nothing to say. Turned on rather than
    # required, because a caller who assembled the store elsewhere should not
    # have to know this flag exists to get attributed answers.
    store.track_provenance = True

    rep = IngestReport()
    for doc in docs:
        rep.documents += 1
        count = 0
        for raw in _SENT.split(doc.text or ""):
            s = raw.strip()
            if len(s) < _min_chars(s):
                continue
            tagged = f"{s} (reported by {doc.source})"
            core, lang = _place(store, tagged, detect_on=s)
            if core is None:
                continue
            by_lang = rep.core_lang.setdefault(core, {})
            by_lang[lang] = by_lang.get(lang, 0) + 1
            count += 1
            rep.sentences += 1
            rep.cores.append(core)
            # Counted with the same detector that placed the poles;
            # the English one alone reported zero polar claims for a
            # Japanese corpus whose poles were in fact placed.
            if detect(s) or detect_ja(s):
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
    # Both languages' aspect keys. English-only here meant a Japanese
    # corpus produced poles that were placed correctly and then never
    # read, so every report came back "supported" no matter how hard
    # the sources disagreed.
    aspect_keys = ({p for p, _ in ANTONYM_PAIRS}
                   | {p for p, _ in ANTONYM_PAIRS_JA})
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
        if not m:
            continue
        label = m.group(1)
        out |= {w for w in re.split(r"[^a-z0-9]+", label.lower()) if w}
        # The Latin split treats every CJK character as a separator, so a
        # source called 「A新聞」 contributed only "a" and 新聞 leaked into the
        # settled facts as though the outlet's name were a fact about the
        # shelter. Same leak that was fixed for English labels; it came back
        # the moment Japanese reached this path, because the fix was written
        # in a pattern that could not see Japanese.
        out |= set(_CJK_RUN.findall(label))
        for run in _CJK_RUN.findall(label):
            # The segmenter emits sub-runs of a compound label, so exclude
            # those too — otherwise 新聞 is filtered but 新 or 聞 is not.
            out |= {run[i:j] for i in range(len(run))
                    for j in range(i + 1, len(run) + 1)}
    return out


#: Kanji/katakana runs — the same shape `lang.ja_content_runs` extracts, so
#: what a source label contributes here matches what ingestion put in.
_CJK_RUN = re.compile(r"[゠-ヿ]+|[㐀-䶿一-鿿]+")


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
