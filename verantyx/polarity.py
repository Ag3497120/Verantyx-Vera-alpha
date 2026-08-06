"""Evidence polarity — the opposition axis, awake.

The cross has had opposite poles since the first sketch and nothing has ever
lived on them: no fact carries a sign, so "the shelter is open" and "the
shelter is closed" land in the same facet bag and the store cannot see that
they fight. This module gives facts a pole, and it does it by PLACEMENT
rather than by new detection machinery: a polar fact is stored as a keyed
facet `aspect:value`, and `CrossStore.contradictions()` — which has detected
multi-valued keys all along — fires on its own. The flow's claim was that
only the placement was missing; that turned out to be literally true.

Polarity detection is a closed vocabulary, deterministic, and deliberately
small: antonym pairs whose two members really are mutually exclusive states
of one aspect, plus negators. No embedding similarity, no sentiment model —
"open" vs "closed" is an opposition; "open" vs "large" is not, and a fuzzy
detector that thinks otherwise would manufacture contradictions, which is
worse than missing them. Words outside the vocabulary simply carry no pole,
exactly as before.

Contradiction becomes an O(facets) LOOKUP on the answered core, not a
search: both poles of one aspect holding mass IS the contradiction, with
per-value provenance when the store tracks it. The consensus gate downgrades
an ANSWER to UNKNOWN_UNRESOLVED_CONTRADICTION only when the query actually
asks about the contradicted aspect — a store may hold a dozen disputes about
a core and still answer questions the disputes do not touch.
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from .cross_store import CrossStore

#: (positive, negative) — the positive member names the aspect key. Mutually
#: exclusive states only; near-synonym gradations (warm/cool) are excluded on
#: purpose because they can both be true enough to not be a contradiction.
ANTONYM_PAIRS: List[Tuple[str, str]] = [
    ("open", "closed"),
    ("safe", "dangerous"),
    ("alive", "dead"),
    ("on", "off"),
    ("full", "empty"),
    ("working", "broken"),
    ("available", "unavailable"),
    ("wet", "dry"),
    ("hot", "cold"),
    ("occupied", "vacant"),
    ("connected", "disconnected"),
    ("passable", "blocked"),
]

_NEGATORS = re.compile(r"\b(not|never|no longer|isn't|aren't|wasn't|weren't)\s+(\w+)",
                       re.IGNORECASE)

_ASPECT_OF: Dict[str, Tuple[str, str]] = {}
for _pos, _neg in ANTONYM_PAIRS:
    _ASPECT_OF[_pos] = (_pos, "+")
    _ASPECT_OF[_neg] = (_pos, "-")


def detect(sentence: str) -> List[Tuple[str, str, str]]:
    """(aspect_key, value_word, polarity) for every polar word in the
    sentence. `not <positive>` flips to the negative pole with a distinct
    value, so 'not open' and 'open' collide on the same key with different
    values — which is what makes the store's multi-value detection fire."""
    out: List[Tuple[str, str, str]] = []
    text = (sentence or "").lower()
    negated: Set[str] = {m.group(2) for m in _NEGATORS.finditer(text)}
    for word in re.findall(r"[a-z']+", text):
        hit = _ASPECT_OF.get(word)
        if hit is None:
            continue
        aspect, pol = hit
        if word in negated:
            pol = "-" if pol == "+" else "+"
            out.append((aspect, f"not_{word}", pol))
        else:
            out.append((aspect, word, pol))
    return out


def ingest_polar(store: CrossStore, sentence: str) -> Optional[str]:
    """Normal ingest plus pole placement. The plain facets stay exactly as
    they were (composition and retrieval are untouched); the polar reading is
    ADDED as keyed facets on the same core."""
    core = store.ingest_sentence(sentence)
    if core is None:
        return None
    keyed = {f"{aspect}:{value}": None for aspect, value, _pol in detect(sentence)}
    if keyed:
        store.add(core, keyed, source=sentence.strip())
    return core


def query_aspects(query: str) -> Set[str]:
    """Aspect keys the query mentions — via either pole's word."""
    return {aspect for aspect, _v, _p in detect(query)}


def bipolar_evidence(store: CrossStore, core: str,
                     aspects: Optional[Set[str]] = None) -> List[Dict[str, Any]]:
    """The O(facets) contradiction lookup: aspects of this core holding mass
    on both poles. Restricted to `aspects` when given, so a dispute the query
    never asked about does not block an unrelated answer."""
    out = []
    for entry in store.contradictions(str(core)):
        if aspects is not None and entry["key"] not in aspects:
            continue
        if entry["key"] in {p for p, _ in ANTONYM_PAIRS}:
            out.append(entry)
    return out


def apply_polarity_gate(store: CrossStore, out: Dict[str, Any], query: str) -> None:
    """Downgrade an ANSWER whose core carries both poles of an aspect the
    query asks about. The evidence rides on the verdict — which values, what
    mass, and (when tracked) which sources said each side — because 'it is
    contested' without the sides named is barely better than a shrug."""
    if out.get("verdict") != "ANSWER":
        return
    core = out.get("core_key") or out.get("core")
    if not core:
        return
    aspects = query_aspects(query)
    if not aspects:
        return
    disputes = bipolar_evidence(store, str(core), aspects)
    if disputes:
        out["verdict"] = "UNKNOWN_UNRESOLVED_CONTRADICTION"
        out["contradictions"] = disputes
        out["reason"] = ("both poles of " +
                         ", ".join(sorted(d["key"] for d in disputes)) +
                         " hold evidence")
