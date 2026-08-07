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
from .ja_grammar import COMPLETION_SUFFIXES as _JA_COMPLETION
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
#: The last alternative is the label-value form a table uses:
#: 「ア 被災による通行止め：なし」 means there are NO closures, and without it
#: the sentence was stored as 通行止 — the claim inverted, with a government
#: citation attached, which is the one failure this module calls worse than
#: silence. Found in 内閣府's 令和8年熊本地震 reports while checking what tabular
#: reading had started to admit. The separator is optional because 「通行止め
#: なし」 occurs bare, and 「通行止め：あり」 is untouched and stays positive.
_JA_NEG_AFTER = re.compile(
    r"^(?:さ)?(?:では|じゃ)?(?:あり)?ません"
    r"|^(?:では|じゃ)?ない"
    r"|^(?:して|されて|できて)?(?:い|おり)?(?:ない|ません)"
    r"|^できない|^できません"
    r"|^[ぁ-ん]?[ 　]*[：:・][ 　]*(?:なし|無し)"
    r"|^[ぁ-ん]?[ 　]*(?:なし|無し)(?![ぁ-ん])"
    r"|^[ぁ-ん]?(?:が|を|の)?[ 　]*解除"
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
                 hits: List[Tuple[str, str, str]], lang: str,
                 claim: Optional[str] = None,
                 context: Optional[str] = None) -> None:
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
    read = claim or sentence
    # One sentence saying both poles about one subject is a change being
    # narrated, not a disagreement, and the later state is the current one:
    # 「合志市 断水あり（復旧済）」 records an outage that is over, and
    # 「一時的に断水があったが、21:40 時点で復旧が確認された」 the same. Storing
    # both would have the store report a municipality as contradicting itself
    # inside a single row of a single document. Later cancels earlier, which
    # also reads 「復旧したが再び断水」 the right way round.
    #
    # Scoped to one subject deliberately: 「A市は断水、B市は復旧」 is two claims,
    # and each still lands on its own noun.
    latest: Dict[Tuple[str, str], Tuple[int, str]] = {}
    for aspect, value, _pol in hits:
        word = value.replace("not_", "")
        if subject_is_core(read, core, word, lang):
            target = core
        else:
            target = subject_of(read, word, lang)
            if target is None:
                # The row asserts a value and names no subject — the normal
                # shape of an official document, where the subject is in the
                # heading above. Only for THAT case: a header row or an
                # enumeration returns (False, None) and still places nothing,
                # which is what keeps the heading from collecting every stray
                # word in its section.
                asserted, _ = (tabular_claim_ja(read, word)
                               if lang == "ja" else (False, None))
                if not (asserted and context):
                    continue
                target = context
        seen_at = read.rfind(word)
        key = (target, aspect)
        if key not in latest or seen_at > latest[key][0]:
            latest[key] = (seen_at, value)

    for (target, aspect), (_seen_at, value) in latest.items():
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

    The one exception is a closed set of completion suffixes shipped as
    grammar data (`ja_grammar.COMPLETION_SUFFIXES`). 済 is a kanji but not a
    compound head: 復旧作業 is an effort and 復旧済 is the state itself, and
    government damage tables record a municipality's water coming back with
    exactly that word.
    """
    at = 0
    while True:
        at = text.find(term, at)
        if at < 0:
            return -1
        rest = text[at + len(term):]
        after = rest[:1]
        # A kanji after the term means a compound — 停止線, 危険物 — and the
        # term is being named rather than asserted. Two kinds of kanji are
        # grammar instead: a completion suffix (復旧済) and a NEGATION
        # (閉鎖解除). Without the second, 「滑走路閉鎖解除済」 was read as a
        # compound and silently dropped: the guard was right that 閉鎖解 is not
        # 閉鎖, and wrong about why. The runway had reopened.
        # 〜中 says the state is still running, and it is safe on exactly one
        # side. For a term that names a STATE — every negative pole here is
        # one — 「操業停止中」 is stopped and 「断水中」 is still out. For a term
        # that names a TRANSITION, 中 means the opposite of the word:
        # 「復旧中」 is being restored and is NOT restored, and reading it as
        # 復旧 would tell a reader the water is back while the source says
        # crews are still working. Restricting it to the negative pole makes
        # that inversion impossible by construction rather than by care.
        ongoing = (after == "中"
                   and (_ASPECT_OF_JA.get(_JA_ALIASES.get(term, term))
                        or ("", ""))[1] == "-")
        if (not after or not _KANJI.match(after)
                or after in _JA_COMPLETION
                or ongoing
                or _JA_NEG_AFTER.match(rest)):
            return at
        at += 1


_KANJI = re.compile(r"[㐀-䶿一-鿿]")

#: The column separator in text extracted from a laid-out page: one or more
#: spaces. 「天草市 断水あり・漏水あり」 is TWO values in ONE cell — ・ joins
#: them — and treating the term as needing to be the last thing on the line
#: read that as a further column and dropped the row.
_JA_COLUMN_GAP = re.compile(r"[ \t\u3000]")

_JA_HIRA_TAIL = re.compile(r"[぀-ゟ]")

#: 〜による / 〜に伴う mark what CAUSED a state, and a cause is not a subject.
_JA_CAUSE_MARK = re.compile(r"^(?:による|によって|により|に伴う|に因る|のため)")

#: ア イ ウ …, the enumerator official Japanese documents label list items
#: with. A one-character katakana run is never a noun — loanwords are two
#: characters or more — so it can be excluded by shape rather than by list.
_JA_ENUMERATOR = re.compile(r"[ァ-ヿ]")

#: A date or a clock time. 「７月29日」 is not a topic and not a subject, and
#: it was becoming both: 「７月29日（水）に開設した熊本刑務所の避難所につき、
#: ８月３日（月）をもって閉鎖」 cored under ７月29日 and lost the shelter.
#: Anchored at a digit, so 国道4号 and 第3条 — where the digit is INSIDE the
#: name — are untouched.
_JA_DATE_RUN = re.compile(r"[0-9０-９]+(?:[年月日時分秒][0-9０-９]*)+$")

#: （水）（月）— the weekday, and the general shape of a parenthesised label.
#: In the same sentence these were chosen as the subjects of 開設 and 閉鎖:
#: Wednesday and Monday read as "water" and "moon", each a perfectly good
#: one-character noun with nothing to do with the claim.
_JA_PAREN_LABEL = re.compile(r"[（(].?[）)]")

#: Case particles. Their presence is what separates prose from a table row,
#: and it decides whether a bare state word at the end is a claim or a column
#: heading: 「熊本刑務所を避難所として開設」 ends in a bare 開設 and is a
#: sentence; 「建物被害 停電 断水」 ends in a bare 断水 and is a heading. That
#: one shelter opening on 8/3 and closing on 8/6 was the only miss the full
#: candidate sweep of this corpus turned up.
_JA_CASE_PARTICLE = re.compile(r"[をにでとへ]|から|まで|より|として")

#: A DEEMING clause defines when something COUNTS as X; it does not say that
#: anything is X. 「火災の予防に危険であると認める物件」 is a statute naming a
#: category. Both of the statute corpus's detections were this shape.
#:
#: Tested on both paths. The prose path applies it through `_anchored_ok`;
#: the tabular path does not go through that, and a legal sentence often has
#: no は or が at all — which is exactly how the first version of this guard
#: was bypassed by the very sentence it was written for.
_JA_DEEMING = re.compile(r"^[ぁ-ん]{0,8}(?:と|であると|でないと)?(認め|みなす|見なす)")


def is_state_word_ja(word: str) -> bool:
    """Is this word a state rather than a thing that can be in one?

    A state word must never be chosen as a subject or as a core. When it is,
    the subject gate asks "is this claim about 断水?" and the answer is
    trivially yes, so every guard downstream stops guarding. Measured twice:
    once as a core (「４県において断水が発生」) and once as a subject
    (「合志市 断水あり（復旧済）」, where 復旧 took 断水 for its subject and the
    municipality's restoration was filed under the outage).

    The vocabulary lists stems and the segmenter emits one okurigana with
    them, so 通行止 must also match 通行止め. Only a trailing hiragana is
    stripped — 開設準備 keeps its 準備 and stays an ordinary noun.
    """
    if word in _ASPECT_OF_JA or word in _JA_ALIASES:
        return True
    return bool(len(word) > 1 and _JA_HIRA_TAIL.match(word[-1])
                and (word[:-1] in _ASPECT_OF_JA or word[:-1] in _JA_ALIASES))

#: What a table cell puts after its state word when the cell holds a VALUE:
#: kana (断水あり, 通行止め) or a completion suffix (復旧済). A bare noun is a
#: column heading and punctuation is a list, and neither asserts anything.
_JA_CELL_VALUE = re.compile(r"^[ぁ-ヿ]|^[" + "".join(sorted(_JA_COMPLETION)) + r"]"
                            if _JA_COMPLETION else r"^[ぁ-ヿ]")


def ingest_polar_ja(store: CrossStore, sentence: str,
                    claim: Optional[str] = None,
                    context: Optional[str] = None) -> Optional[str]:
    """Japanese ingest plus pole placement.

    Mirrors `ingest_polar`, but segments with `lang.ja_ingest_sentence`
    because the English decomposer's word pattern is `[A-Za-z0-9']+` and
    returns nothing at all for Japanese. Keyed facets land on the same core
    in the same `aspect:value` shape, so `CrossStore.contradictions()` fires
    without knowing which language produced them.

    `claim` is the sentence WITHOUT its attribution suffix, when the caller
    has it. Structure is read from the claim and provenance is stored with
    the citation, which had been the same string until a rule needed to know
    where the sentence ENDS: the tabular-row reading requires the state to be
    the last column, and 「熊本市 … ・復旧済 (reported by 内閣府 8/6)」 has the
    citation sitting in that column. Detection already had its own version of
    this problem — the Latin in the suffix outvoted a short Japanese sentence
    and sent it to the English decomposer — so the suffix now stays out of
    both votes.
    """
    from .lang import ja_ingest_sentence

    core = ja_ingest_sentence(store, sentence)
    if core is None:
        return None
    _place_poles(store, sentence, core, detect_ja(sentence), "ja",
                 claim=claim, context=context)
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


#: A section heading: a numbering marker, then a name. 内閣府 writes road
#: closures as 「①高速道路」 followed by 「ア 被災による通行止め：２路線１３区間」,
#: and the row alone cannot say which network it is about. Measured on four
#: revisions: two of the five networks genuinely changed polarity across them
#: — 有料道路 gained a closure on 8/6, 直轄国道 cleared after 7/29 — and both
#: were unreachable, the entire remaining recall gap on that corpus.
#:
#: Deliberately narrow. Only markers that cannot be a data cell are accepted:
#: ① 〜 ⑳, （2）, 3., 【福岡県】. 「7 0 7/28 ・復旧済」 also begins with a digit
#: and a space, so that form is excluded — a heading must be unmistakable,
#: because it is about to be handed a claim that has no subject of its own.
#: An article number. In a statute this is not decoration — it is the
#: citation key, the only way anyone refers to a provision, and the unit a
#: version diff compares. e-Gov emits it as its own line, so it arrived as a
#: separate segment, fell under the minimum length, and every provision in
#: 334,330 characters of 災害対策基本法 and 消防法 was stored with no way to
#: say which article it came from.
#:
#: 「第六十条の二」 and 「第六十条第二項」 are distinct provisions and keep
#: their full form; 「２」 and 「３」 alone are paragraph numbers WITHIN an
#: article, so they refine the current heading rather than replacing it.
_JA_ARTICLE = re.compile(r"^第[一二三四五六七八九十百千0-9０-９]+条"
                         r"(?:の[一二三四五六七八九十0-9０-９]+)?"
                         r"(?:第[一二三四五六七八九十0-9０-９]+項)?$")

_JA_HEADING_BRACKET = re.compile(r"^【([^】]{1,24})】$")
_JA_HEADING_MARK = re.compile(
    r"^(?:[①-⑳]|[（(]\s*[0-9０-９]{1,2}\s*[）)]|[0-9０-９]{1,2}[\.．])\s*(.+)$")


def heading_subject_ja(text: str) -> Optional[str]:
    """The thing a section heading names, or None if this is not a heading."""
    from .lang import ja_content_runs

    t = (text or "").strip()
    if not t or len(t) > 40:
        return None
    # An article number is its own heading, and unlike the others it IS the
    # subject rather than naming one.
    if _JA_ARTICLE.match(t):
        return t
    m = _JA_HEADING_BRACKET.match(t) or _JA_HEADING_MARK.match(t)
    if not m:
        return None
    rest = re.split(r"[（(【\[：:]", m.group(1))[0].strip()
    # A heading that names a STATE is a column header, and handing rows to it
    # would file every value under the word for the value.
    if not rest or detect_ja(rest):
        return None
    runs = [r for r in ja_content_runs(rest)
            if not is_state_word_ja(r) and not _JA_DATE_RUN.match(r)
            and not _JA_ENUMERATOR.fullmatch(r)]
    return runs[-1] if runs else None


def tabular_claim_ja(text: str, word: str) -> Tuple[bool, Optional[str]]:
    """Read a table row: (is this an asserted cell value, whose is it).

    Official damage reports keep their per-place facts in tables, and a table
    row has no particle at all — nothing marks a subject, so the prose paths
    find nothing and the row's claim is dropped. Japanese tabular notation
    reads head-final like the rest of the language: the state in the last
    column is predicated of the nearest noun before it.

    This is the reading convention, not an inference about one ministry's
    layout, but a row is also the exact shape a heading has, so it is guarded:

      no particle    a sentence has は or が; a row does not. This is what
                     keeps the rule away from prose entirely.
      last column    the state must sit in the row's final field, so a mid-row
                     mention (「停止 断水」— 停止 is not the status) is
                     excluded. Columns are separated by WHITESPACE, not by the
                     term being last on the line: 「天草市 断水あり・漏水あり」 is
                     two values in one cell, and requiring nothing after the
                     term cost half the recall this rule exists to buy — 3 of
                     6 restorations, measured against a water table read by
                     hand, the three missed all having a compound cell.
      not a quantity a subject may not end in a bare digit. 「約20,970」 is how
                     many households lost water, not who did. 国道4号 and 第3条
                     end in 号/条 and stay eligible.
      value marked   a data cell says what it holds, in kana or with a
                     completion suffix — 断水あり, 復旧済, 通行止め. A HEADER
                     names columns with bare nouns (「建物被害 停電 断水」 is the
                     damage table's heading; across four revisions 断水 appears
                     bare 26 times and as 断水あり 14, the bare ones headings),
                     and an ENUMERATION lists topics (「害、 停電、 断水、」).
                     Prose is exempt from the marker, because a sentence may
                     end in a bare verbal noun — 「熊本刑務所を避難所として開設」
                     — and case particles are what tell the two apart.

    The two return values are separate on purpose. `(False, None)` means this
    is not a claim and nothing should be placed. `(True, None)` means it IS a
    claim whose subject is not in the row — which is the normal shape of an
    official document, 「①高速道路 / ア 被災による通行止め：２路線１３区間」, and
    the caller can supply the heading. Collapsing both into None is what made
    those rows unreachable.
    """
    from .lang import ja_content_runs

    if re.search(r"[はが]", text) or re.search(
            r"(場合|とき|なら|れば|たら|予定|見込)", text):
        return False, None
    at = _standalone_index(text, word)
    if at < 0:
        return False, None
    tail = text[at + len(word):]
    # A negation is a value too. 「滑走路閉鎖解除済」 says the runway REOPENED,
    # and the marker test looked only at the character after 閉鎖 — 解 — which
    # is neither kana nor a completion suffix, so the row asserted nothing and
    # the reopening was dropped.
    marked = bool(_JA_CELL_VALUE.match(tail) or _JA_NEG_AFTER.match(tail)
                  or _JA_CASE_PARTICLE.search(text))
    # A term followed by a list separator is being NAMED, not asserted —
    # 「避難所の開設、運営等について」 lists what the guidance covers.
    # `_anchored_ok` applies the same rule on the prose path; stated again
    # because this branch does not go through it, and because relaxing the
    # marker for particle-bearing prose reopened exactly this hole.
    enumerated = bool(re.match(r"^[、，]", tail))
    if (not marked or enumerated or _JA_COLUMN_GAP.search(tail)
            or _JA_DEEMING.match(tail)):
        return False, None

    head = text[:at]
    for run in reversed(ja_content_runs(head)):
        if (run == word or is_state_word_ja(run)
                or run[-1] in "0123456789０１２３４５６７８９"
                or _JA_ENUMERATOR.fullmatch(run)
                or _JA_DATE_RUN.match(run)):
            continue
        # A cause is not a subject. 「ア 被災による通行止め：なし」 and
        # 「ア 被災による通行止め：２県６区間」 are two road networks' rows in ONE
        # 8/6 report, each headed the same way, and taking 被災 for the subject
        # made the document contradict itself.
        where = head.rfind(run) + len(run)
        if _JA_CAUSE_MARK.match(head[where:]):
            continue
        return True, run
    return True, None


def _around(text: str, run: str) -> str:
    """The run with the character on each side — enough to see （水）."""
    at = text.rfind(run)
    if at < 0:
        return run
    return text[max(0, at - 1):at + len(run) + 1]


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
            if (run == word or is_state_word_ja(run)
                    or _JA_DATE_RUN.match(run)
                    or (len(run) == 1
                        and _JA_PAREN_LABEL.match(_around(text, run)))
                    or not _anchored_ok(text, run, word, "ja")):
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

        asserted, found = tabular_claim_ja(text, word)
        return found if asserted else None

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
        # A DEEMING clause defines when something counts as X; it does not say
        # that anything is X. 「火災の予防に危険であると認める物件」 is a statute
        # naming a category, and reading it as a claim that a物件 is dangerous
        # inverts what a regulation is for. Found blind on 災害対策基本法 and
        # 消防法, where both of the corpus's detections were this shape.
        #
        # The window is bounded and stops at a clause boundary, so a real
        # claim followed later by an unrelated 認める is untouched.
        if _JA_DEEMING.match(after):
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
