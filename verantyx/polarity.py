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
# The Japanese vocabulary is DATA, shipped as lang_data/ja_grammar.json
# and extensible by overlay — the design rationale (2-char floor, one
# pole per term, aspect joins) is enforced by ja_grammar.validate().
from .ja_grammar import ALIASES as _JA_ALIASES
from .ja_grammar import ANTONYM_PAIRS as ANTONYM_PAIRS_JA
from .ja_grammar import ASPECT_OF as _ASPECT_OF_JA
from .ja_grammar import TERMS as _JA_TERMS

#: Negation that immediately FOLLOWS the polar term. Japanese negation is a
#: suffix on the predicate, so it is read from the characters after the
#: matched term, not scanned for elsewhere in the sentence.
#:
#: The previous version tried to pre-scan the sentence for anything shaped
#: like 〜ない and match it back to terms by prefix. Measured result:
#: 「この道は安全ではありません」 was stored as 安全(+) — the pole INVERTED,
#: the store asserting the road is safe where the source said it is not.
#: For the fields this is meant to serve, a detector that silently flips
#: negated claims is worse than no detector: every other error here loses
#: information, this one manufactured the opposite claim with a citation.
_JA_NEG_AFTER = re.compile(
    r"^(?:さ)?(?:では|じゃ)?(?:あり)?ません"
    r"|^(?:では|じゃ)?ない"
    r"|^(?:して|されて|できて)?(?:い|おり)?(?:ない|ません)"
    r"|^できない|^できません"
)

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

# JA derived tables (ASPECT_OF, TERMS) come from ja_grammar above.


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
        # Read the suffix directly after the term — that is where Japanese
        # puts the negation, and the only place it can belong to THIS term.
        negated_here = bool(_JA_NEG_AFTER.match(text[start + len(term):]))
        # Blank the matched span so a shorter term inside it cannot match
        # separately — 「受付終了」must not also report 「受付中」.
        text = text[:start] + "　" * len(term) + text[start + len(term):]
        if negated_here:
            pol = "-" if pol == "+" else "+"
            out.append((aspect, f"not_{canonical}", pol))
        else:
            # Same convention as the English half: the value is the word, and
            # `not_` appears only for an explicit negation. Naming the
            # negative pole `not_危険` would have said "not dangerous" about a
            # sentence that said dangerous.
            out.append((aspect, canonical, pol))
    return out


def _place_poles(store: CrossStore, sentence: str, core: str,
                 hits: List[Tuple[str, str, str]], lang: str) -> None:
    """Attach each detected pole to the noun it is actually predicated of.

    The recall half of the subject gate. The gate alone asked "is the CORE
    the subject?" and threw the pole away otherwise — correct for precision,
    and it cost every quotative claim: "Staff confirmed that the lighthouse
    has been open" cores under staff, so the lighthouse's state vanished.
    Now the pole follows its subject: on the core when the core is the
    subject (the common case, unchanged), on the subject's own cross when it
    is some other noun, and nowhere when no noun passes — a hypothetical
    (「使用不可の場合」, "if the elevator is unavailable") still places
    nothing, because a pole from a supposition is a manufactured fact.
    """
    for aspect, value, _pol in hits:
        word = value.replace("not_", "")
        if subject_is_core(sentence, core, word, lang):
            target = core
        else:
            target = subject_of(sentence, word, lang)
            if target is None:
                continue
        store.add(target, {f"{aspect}:{value}": None},
                  source=sentence.strip())


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
    _place_poles(store, sentence, core, detect_ja(sentence), "ja")
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


def subject_of(sentence: str, word: str, lang: str = "en") -> Optional[str]:
    """The noun `word` is predicated of, or None if there is no clean subject.

    This generalizes the subject gate from a yes/no about the core into
    finding the actual subject — the recall half of the same coin. Measured
    need: "Staff confirmed that the lighthouse has been open" files its core
    under staff, so a gate that only asks "is the core the subject?" throws
    the claim away. The claim is real; it just belongs to the lighthouse.
    Now the pole is placed on whichever noun IS the subject.

    Two refusals, both learned from planted traps:

      conditionals   "If the elevator is unavailable, …" and 「使用不可の
                     場合は…」 assert nothing — they suppose. A subject whose
                     clause opens with if/when/unless (EN) or whose predicate
                     is followed by 場合/なら/れば (JA) is rejected, because a
                     pole from a hypothetical is a manufactured fact.
      no subject     when no noun passes the anchored test, the answer is
                     None and no pole is placed anywhere — same as before.
    """
    text = sentence or ""
    if lang == "ja":
        from .lang import ja_content_runs
        for run in reversed(ja_content_runs(text)):
            if run == word or not _anchored_ok(text, run, word, "ja"):
                continue
            return run
        # Enumerated subjects: 「九州自動車道、南九州自動車道など通行止めが発生」.
        # A real miss, found by reading the one government release in the
        # corpus that actually reported a closure — a person reads it as
        # "九州自動車道 is closed" and the gate saw no subject at all,
        # because the road names sit in a list and the polar term is what
        # carries が.
        #
        # Structurally distinct from the enumeration FALSE positive that the
        # guard above rejects: there, the polar term is INSIDE the list
        # (「開設、運営等については」— 開設 is one of the things being listed);
        # here it FOLLOWS a list closed by など/等 and is predicated of every
        # item in it. Inside versus after is the whole distinction, and it is
        # visible in the characters.
        m = re.search(r"((?:[^、，。\s]+[、，]){1,6}[^、，。\s]*?)(?:など|等)\s*"
                      + re.escape(word), text)
        if m:
            items = [x for x in re.split(r"[、，]", m.group(1)) if x.strip()]
            if items:
                runs = ja_content_runs(items[0])
                if runs:
                    return runs[-1]
        return None

    # Lookahead, not consumption: in "that the lighthouse", a consuming
    # scan eats "that the" and captures "the" — the first run of this did
    # exactly that. Overlapping matches let "the lighthouse" be seen too.
    for m in reversed(list(re.finditer(
            r"(?=\b(?:the|a|an|this|that|its|their|our)\s+([A-Za-z][\w-]*))",
            text, re.IGNORECASE))):
        cand = m.group(1)
        if cand.lower() in {"the", "a", "an", "this", "that", "these",
                            "those", "same", "other", "first", "last",
                            word.lower()}:
            continue
        if not _anchored_ok(text, cand, word, "en"):
            continue
        return cand.casefold()
    return None


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
    return _anchored_ok(text, core if lang == "ja" else core.replace("_", " "),
                        word, lang)


def _anchored_ok(text: str, noun: str, word: str, lang: str) -> bool:
    """Anchor match plus the conditional guard AT the matched position.

    One helper for both the core route and the subject search, because the
    two drifted: the conditional check lived only in subject_of, so a
    sentence whose CORE happened to be the noun inside a when-clause placed
    a pole from a hypothetical. Found on a real document — "The group
    policy values (when group access is available…)" cored as group, the
    core route anchored inside the parenthetical, and a supposition met a
    genuine claim from the same file as a manufactured dispute.
    """
    if lang == "ja":
        m = _ja_anchor_match(text, noun, word)
        if m is None:
            return False
        at = text.rfind(word, m.start(), m.end())
        after = text[at + len(word):at + len(word) + 10] if at >= 0 else ""
        # A hypothetical asserts nothing.
        if re.match(r"^[のでにと]?(場合|とき|なら|れば|たら)", after):
            return False
        # A past incident is history, not a current state. Found on a real
        # government case-study PDF: 「避難所が閉鎖した後にPCR検査を実施した」
        # is one shelter's story, and reading it as "this shelter is closed"
        # put it against a sentence elsewhere in the SAME document.
        if re.match(r"^(した|され(た|て)|してい(た|る)?)?(後|際|時|直後|とき)", after):
            return False
        # A noun inside an enumeration is a topic, not a predicate:
        # 「避難所の開設、運営等については」 lists what the guidance covers.
        # A polar term followed by a list separator, or preceded by の, is
        # being named rather than asserted.
        if re.match(r"^[、，・]", after):
            return False
        if at >= 1 and text[at - 1] == "の":
            return False
        return True
    m = _en_anchor_match(text, noun, word)
    if m is None:
        return False
    clause_start = max(text.rfind(ch, 0, m.start()) for ch in ".,;:(")
    clause = text[clause_start + 1:m.start()]
    return not re.search(r"\b(if|when|unless|while|whether|suppose|assuming)\b",
                         clause, re.IGNORECASE)
    # The gap between the core and its copula may not cross a clause
    # boundary. Without this, "If sandbox mode is enabled but Docker is
    # unavailable" still matched: `sandbox` … `is unavailable`, with a whole
    # other subject in between. Commas, conjunctions and subordinators are
    # where a new subject gets introduced, so the span stops at them.
    # Between the copula and the state word, two kinds of material are
    # grammatical and claim-preserving, and both come from the tier B
    # measurement (recall was 0% on English passives and adverbs while
    # Japanese passed):
    #
    #   adverbs        "remains fully open", "is still closed", "was
    #                  completely unavailable" — closed by FORM (-ly) plus
    #                  the handful of common adverbs that lack the suffix.
    #   small-clause   "was reported closed", "is considered dangerous" —
    #   participles    verbs of report and judgement whose complement is a
    #                  predicate over the SUBJECT. A closed class: adding an
    #                  arbitrary verb here would let "was painted closed"
    #                  through, so only verbs whose grammar guarantees the
    #                  attachment are listed.
def _ja_anchor(text: str, noun: str, word: str) -> bool:
    return _ja_anchor_match(text, noun, word) is not None


def _ja_anchor_match(text: str, noun: str, word: str):
    pat = re.compile(re.escape(noun)
                     + r"(?:に関して|について|につきまして)?"
                     + r"[^。]{0,12}?[はがも][^。]{0,24}?" + re.escape(word))
    return pat.search(text)


def _en_anchor(text: str, noun: str, word: str) -> bool:
    return _en_anchor_match(text, noun, word) is not None


def _en_anchor_match(text: str, noun: str, word: str):
    _mid = (r"(?:(?:\w+ly|still|now|again|already|once|long|almost|fully)\s+)*"
            r"(?:(?:reported|declared|confirmed|considered|deemed|marked|"
            r"found|kept|left|ruled|judged|presumed)\s+)?"
            r"(?:(?:\w+ly|still|now|again)\s+)*")
    pat = re.compile(
        r"\b" + re.escape(noun)
        + r"\b(?:(?!\b(?:but|and|or|if|when|while|unless|because|though|"
          r"although|which|that|where)\b)[^.,;:()\[\]])"
          r"{0,32}?\s*\b"
        r"(?:is|are|was|were|be|been|being|has\s+been|have\s+been|"
        r"becomes?|became|remains?|stays?|"
        r"turned|seems?|appears?)\s+(?:not\s+|no\s+longer\s+)?" + _mid
        + re.escape(word) + r"\b", re.IGNORECASE)
    return pat.search(text)


def ingest_polar(store: CrossStore, sentence: str) -> Optional[str]:
    """Normal ingest plus pole placement. The plain facets stay exactly as
    they were (composition and retrieval are untouched); the polar reading is
    ADDED as keyed facets on the same core — but only for poles the core is
    actually the subject of."""
    core = store.ingest_sentence(sentence)
    if core is None:
        return None
    _place_poles(store, sentence, core, detect(sentence), "en")
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
