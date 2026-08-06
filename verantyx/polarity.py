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

#: Japanese pairs. Held to a stricter rule than the English ones, because
#: Japanese has no word boundaries and matching is therefore by substring:
#: every term here must be long and distinctive enough that it cannot appear
#: inside an unrelated word. That is why 「開」 is absent — it sits inside
#: 開始, 公開, 展開 and would manufacture a contradiction every time a
#: document mentioned a meeting starting. Missing an opposition costs one
#: undetected disagreement; inventing one costs the reader's trust in every
#: contested claim the report shows.
#:
#: Verified against a control corpus of unrelated sentences containing the
#: near-miss words — see polarity_ja_eval in settings-adjacent evals.
ANTONYM_PAIRS_JA: List[Tuple[str, str]] = [
    ("通行可能", "通行止"),
    ("開設", "閉鎖"),
    ("営業中", "休業"),
    ("安全", "危険"),
    ("実施", "中止"),
    ("稼働", "停止"),
    ("使用可能", "使用不可"),
    ("復旧", "断水"),
    ("受付中", "受付終了"),
]

#: Inflected forms that mean the same pole as a listed term. Kept separate so
#: the pair table stays readable as pairs.
_JA_ALIASES: Dict[str, str] = {
    "開いています": "開設", "開いてい": "開設", "利用できます": "使用可能",
    "閉まっています": "閉鎖", "閉鎖されました": "閉鎖",
    "通行できません": "通行止", "通行できます": "通行可能",
    "使用できません": "使用不可",
}

#: 〜ない / 〜ません flips the preceding term. Japanese negation is a suffix,
#: not a preceding word, so the English negator pattern cannot see it.
_JA_NEGATED = re.compile(r"(.{2,6}?)(?:では)?(?:あり)?(?:ませ|な)ん?[^あ-ん]*?(?:ない|ません)")

#: English terms that are also ordinary prepositions or parts of hyphenated
#: words, and so need a copula in front to be a claim about a state.
#:
#: Measured, not guessed. Ingesting this project's own 12 documents produced
#: exactly one contradiction across 251 cores, and it was false: "pour a
#: second corpus ON top" against "trade-OFF", neither of which says anything
#: is on or off. Precision was 0 of 1. The Japanese side already had a guard
#: for the same class of error (a polar term swallowed by a compound); the
#: English side had none.
_REQUIRES_COPULA = {"on", "off"}
_COPULA_BEFORE = re.compile(
    r"\b(?:is|are|was|were|be|been|being|stays?|remains?|turned)\s+$",
    re.IGNORECASE)

_NEGATORS = re.compile(r"\b(not|never|no longer|isn't|aren't|wasn't|weren't)\s+(\w+)",
                       re.IGNORECASE)

_ASPECT_OF: Dict[str, Tuple[str, str]] = {}
for _pos, _neg in ANTONYM_PAIRS:
    _ASPECT_OF[_pos] = (_pos, "+")
    _ASPECT_OF[_neg] = (_pos, "-")

_ASPECT_OF_JA: Dict[str, Tuple[str, str]] = {}
for _pos, _neg in ANTONYM_PAIRS_JA:
    _ASPECT_OF_JA[_pos] = (_pos, "+")
    _ASPECT_OF_JA[_neg] = (_pos, "-")

#: Longest first, so 使用可能 is matched before 使用不可's shorter neighbours
#: and 受付終了 is never read as 受付中 plus noise.
_JA_TERMS: List[str] = sorted(
    list(_ASPECT_OF_JA) + list(_JA_ALIASES), key=len, reverse=True)


def detect_ja(sentence: str) -> List[Tuple[str, str, str]]:
    """The Japanese half of `detect`, matched by substring.

    Substring matching is forced by the language — there are no spaces to
    split on — and it is why the vocabulary above is restricted to long,
    distinctive terms. Scanning longest-first and blanking each hit stops one
    stretch of text from being counted twice, which would otherwise let
    使用可能 also register as a 使用不可 near-miss.
    """
    out: List[Tuple[str, str, str]] = []
    text = sentence or ""
    negated = {m.group(1) for m in _JA_NEGATED.finditer(text)}
    seen: Set[str] = set()
    for term in _JA_TERMS:
        start = _standalone_index(text, term)
        if start < 0:
            continue
        canonical = _JA_ALIASES.get(term, term)
        hit = _ASPECT_OF_JA.get(canonical)
        if hit is None or canonical in seen:
            continue
        seen.add(canonical)
        aspect, pol = hit
        # Blank the matched span so a shorter term inside it cannot match
        # separately — 「受付終了」must not also report 「受付中」.
        text = text[:start] + "　" * len(term) + text[start + len(term):]
        if any(canonical.startswith(n) or n.startswith(canonical)
               for n in negated):
            pol = "-" if pol == "+" else "+"
            out.append((aspect, f"not_{canonical}", pol))
        else:
            # Same convention as the English half: the value is the word, and
            # `not_` appears only for an explicit negation. Naming the
            # negative pole `not_危険` would have said "not dangerous" about a
            # sentence that said dangerous.
            out.append((aspect, canonical, pol))
    return out


def _standalone_index(text: str, term: str) -> int:
    """Where `term` occurs as its own word, or -1.

    Japanese compounds are formed by butting kanji together, so a substring
    match alone reads 停止線 (a painted stop line) as "stopped" and 危険物
    (hazardous materials) as "dangerous" — both were produced by the first
    version of this, and both would have manufactured a contradiction out of
    a sentence that made no claim at all.

    The rule: reject when the next character is a kanji, because that is a
    compound. Hiragana and katakana after the term are inflection or
    particles and leave the meaning intact — 通行止です, 開設されました,
    復旧し通行可能に all survive it.

    Deliberately one-sided. Testing the preceding character too would reject
    大変危険です, where the kanji before is an intensifier and the claim is
    real. Missing a compound that leads with a polar term is the cheaper
    error, and the vocabulary is small enough to inspect.
    """
    at = 0
    while True:
        at = text.find(term, at)
        if at < 0:
            return -1
        after = text[at + len(term):at + len(term) + 1]
        if not after or not _KANJI.match(after):
            return at
        at += 1


_KANJI = re.compile(r"[㐀-䶿一-鿿]")


def ingest_polar_ja(store: CrossStore, sentence: str) -> Optional[str]:
    """Japanese ingest plus pole placement.

    Mirrors `ingest_polar`, but segments with `lang.ja_ingest_sentence`
    because the English decomposer's word pattern is `[A-Za-z0-9']+` and
    returns nothing at all for Japanese. Keyed facets land on the same core
    in the same `aspect:value` shape, so `CrossStore.contradictions()` fires
    without knowing which language produced them.
    """
    from .lang import ja_ingest_sentence

    core = ja_ingest_sentence(store, sentence)
    if core is None:
        return None
    keyed = {f"{aspect}:{value}": None
             for aspect, value, _ in detect_ja(sentence)
             if subject_is_core(sentence, core, value.replace("not_", ""), "ja")}
    if keyed:
        store.add(core, keyed, source=sentence.strip())
    return core


def detect(sentence: str) -> List[Tuple[str, str, str]]:
    """(aspect_key, value_word, polarity) for every polar word in the
    sentence. `not <positive>` flips to the negative pole with a distinct
    value, so 'not open' and 'open' collide on the same key with different
    values — which is what makes the store's multi-value detection fire."""
    out: List[Tuple[str, str, str]] = []
    text = (sentence or "").lower()
    negated: Set[str] = {m.group(2) for m in _NEGATORS.finditer(text)}
    for m in re.finditer(r"[a-z']+", text):
        word = m.group(0)
        hit = _ASPECT_OF.get(word)
        if hit is None:
            continue
        if word in _REQUIRES_COPULA and not _COPULA_BEFORE.search(text[:m.start()]):
            # "on top of", "trade-off" — the word is present, the claim is not.
            continue
        aspect, pol = hit
        if word in negated:
            pol = "-" if pol == "+" else "+"
            out.append((aspect, f"not_{word}", pol))
        else:
            out.append((aspect, word, pol))
    return out


def subject_is_core(sentence: str, core: str, word: str,
                    lang: str = "en") -> bool:
    """Is `word` predicated of `core`, or of something else in the sentence?

    This gate exists because of a measurement. Catalogued across 2,633 real
    documents, the polarity detector produced 39 contradictions and the four
    inspected were all false, in the same way every time:

        "The gateway surfaces one installer (brew when available)"
        "If sandbox mode is enabled but Docker is unavailable"

    `available` describes brew, `unavailable` describes Docker — but a facet
    bag has no attachment, so both landed on the sentence's core and the two
    poles then looked like a disagreement about the gateway. Precision was
    0 of 4 on a corpus with no real disputes in it, which is the worst
    possible ratio: the system was confidently wrong and nothing was right.

    So a pole is placed only when the core is the SUBJECT of the predicate
    carrying it. Approximated, not parsed: English wants the core, then a
    copula, then the word, within a short span; Japanese wants the core
    marked by は or が before the term. Both miss real claims phrased around
    the pattern. That is the correct trade for a catalogue whose whole value
    is that a listed disagreement is worth investigating.
    """
    text = (sentence or "")
    if not core:
        return False
    if lang == "ja":
        pat = re.compile(re.escape(core) + r"[^。]{0,12}?[はがも][^。]{0,24}?"
                         + re.escape(word))
        return bool(pat.search(text))
    # The gap between the core and its copula may not cross a clause
    # boundary. Without this, "If sandbox mode is enabled but Docker is
    # unavailable" still matched: `sandbox` … `is unavailable`, with a whole
    # other subject in between. Commas, conjunctions and subordinators are
    # where a new subject gets introduced, so the span stops at them.
    pat = re.compile(
        r"\b" + re.escape(core.replace("_", " "))
        + r"\b(?:(?!\b(?:but|and|or|if|when|while|unless|because|though|"
          r"although|which|that|where)\b)[^.,;:()\[\]])"
          r"{0,32}?\s*\b"
        r"(?:is|are|was|were|be|been|being|becomes?|became|remains?|stays?|"
        r"turned|seems?|appears?)\s+(?:not\s+|no\s+longer\s+)?"
        + re.escape(word) + r"\b", re.IGNORECASE)
    return bool(pat.search(text))


def ingest_polar(store: CrossStore, sentence: str) -> Optional[str]:
    """Normal ingest plus pole placement. The plain facets stay exactly as
    they were (composition and retrieval are untouched); the polar reading is
    ADDED as keyed facets on the same core — but only for poles the core is
    actually the subject of."""
    core = store.ingest_sentence(sentence)
    if core is None:
        return None
    keyed = {f"{aspect}:{value}": None
             for aspect, value, _pol in detect(sentence)
             if subject_is_core(sentence, core, value.replace("not_", ""), "en")}
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
