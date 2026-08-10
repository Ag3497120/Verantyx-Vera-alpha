"""What the user already settled, and whether the reply still honours it.

Not comprehension. Two mechanical checks against things somebody already
said, and a proposal to paste them back in:

    covenant   a rule the user set — required terms, forbidden terms, and
               the topic it applies to. Violated when the reply is on that
               topic and the requirement is missing or a prohibition is used
    collapse   the reply is ABOUT something the conversation already
               constrained, and shares nothing with what was said about it

Both are answers to "has the model forgotten", and neither needs to know
what anything means. A covenant is a string test with a scope; a collapse is
the `attest_llm` linkage test with the CONVERSATION as the corpus instead of
a document store.

## The output is a proposal, never a verdict on the reply

A guard that blocks is wrong here. The model may be departing from an
earlier instruction because the user just changed it, and this layer cannot
tell those apart — it sees text, not intent. So a finding carries the exact
sentence to re-inject and the turn it came from, and a human or the calling
agent decides. The failure this prevents is the silent one: a window slides,
an instruction from turn 3 falls out of it, and nothing anywhere says so.

## Why the conversation is a different corpus from the knowledge store

`attest_claim` asks whether a claim matches a body of documents. This asks
whether a reply matches THIS conversation. A reply can be perfectly true and
still have forgotten what the user asked for, and that is the failure worth
catching in a long session. Same linkage test, different store, and mixing
them would let a well-attested falsehood pass as consistency.

## What it cannot do, stated because it will be deployed

It matches strings against scopes. 「TypeScriptを使う」 is caught by naming
TypeScript required and JavaScript forbidden; it cannot infer that
prohibition from the requirement. A covenant nobody wrote down is not
checked, and a paraphrase that avoids every registered term is not caught.
The register is the contract — this reports what was registered, exactly.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

#: Hiragana is excluded from the CONTINUATION class on purpose. Including
#: it let one run swallow its own particles — 「プロジェクトはPythonで実装
#: されています」 came back as a single term, so no subject was ever
#: recognised and `collapse` reported every reply consistent.
_RUN = re.compile(r"[㐀-䶿一-鿿ァ-ヺー々〆A-Za-z][㐀-䶿一-鿿ァ-ヺー々〆A-Za-z0-9.+#-]*")


def terms_of(text: str) -> List[str]:
    """Content runs, latin words included — a rule often names a tool."""
    out: List[str] = []
    for m in _RUN.finditer(text or ""):
        t = m.group(0)
        if len(t) >= 2 and t not in out:
            out.append(t)
    return out


@dataclass
class Covenant:
    """One thing the user settled, in the form a machine can check."""

    name: str
    requires: List[str] = field(default_factory=list)
    forbids: List[str] = field(default_factory=list)
    #: Terms that put a reply in this covenant's scope. Empty means always,
    #: which is deliberately hard to get: an always-on rule fires on replies
    #: that were never about it, and a guard that cries every turn is turned
    #: off by the second day.
    topic: List[str] = field(default_factory=list)
    said_at_turn: int = -1
    said_by: str = "user"
    quote: str = ""

    def in_scope(self, text: str, asked: str = "") -> bool:
        """Scope is the EXCHANGE, not the reply's wording.

        A rule about the implementation language did not fire on the reply
        「Python。」 — one word, on topic, naming no scope term. Checking the
        question that prompted it fixes exactly that case, and it is the
        right reading anyway: a covenant binds what was asked and answered,
        not the vocabulary the answer happened to use.
        """
        if not self.topic:
            return True
        return any(t in text or (asked and t in asked) for t in self.topic)

    def check(self, text: str, asked: str = "",
              store: Any = None, infer_top: int = 6) -> Optional[Dict[str, Any]]:
        """``store`` lets the covenant infer prohibitions it never listed.

        Registered by hand, a rule catches only the substitution somebody
        anticipated. Given a store, the siblings of each required term are
        the alternatives the geometry already knows about — measured over
        four legal alternative sets, 11 of 14 terms recovered another member
        of their own set, most at rank one.

        Inferred hits are reported SEPARATELY from registered ones. A
        registered prohibition is what the user said; an inferred one is
        what the corpus suggests they meant, and a reader deciding whether
        to re-inject needs to know which is which.
        """
        if not self.in_scope(text, asked):
            return None
        used = [f for f in self.forbids if loosely_in(f, text)]
        missing = [r for r in self.requires if not loosely_in(r, text)]
        inferred: List[Dict[str, Any]] = []
        if store is not None and missing:
            for r in missing:
                for w, s in siblings(store, r, limit=infer_top):
                    if loosely_in(w, text):
                        inferred.append({"instead_of": r, "used": w,
                                         "sibling_score": s})
        if not used and not missing:
            return None
        return {
            "covenant": self.name,
            "forbidden_used": used,
            "required_missing": missing,
            "substituted": inferred,
            "said_at_turn": self.said_at_turn,
            "said_by": self.said_by,
            # What to paste back, verbatim, rather than a paraphrase of it.
            "inject": self.quote or self.name,
        }

    def as_dict(self) -> Dict[str, Any]:
        return {"name": self.name, "requires": self.requires,
                "forbids": self.forbids, "topic": self.topic,
                "said_at_turn": self.said_at_turn, "said_by": self.said_by,
                "quote": self.quote}


@dataclass
class Register:
    """The covenants in force, the turns they came from, and their history.

    ## Which harness, and when

    Re-injecting every rule every turn is what a system prompt already does,
    and it is why long sessions drift anyway: the model has seen the rule so
    often it has stopped carrying information. What carries information is a
    rule that has JUST started being broken.

    So each covenant keeps its own record of checks and breaks, and `fading`
    reports the ones whose recent behaviour differs from their history — a
    rule kept for twenty turns and broken twice in the last three is the one
    worth spending context on. A rule never broken needs no reminder, and a
    rule broken from the beginning was probably never understood and needs
    rewriting rather than repeating.
    """

    covenants: List[Covenant] = field(default_factory=list)
    #: covenant name -> the check results, oldest first, True = kept
    history: Dict[str, List[bool]] = field(default_factory=dict)

    def add(self, c: Covenant) -> Covenant:
        self.covenants.append(c)
        self.history.setdefault(c.name, [])
        return c

    def fading(self, window: int = 5, min_history: int = 4) -> Dict[str, Any]:
        """Covenants whose recent compliance is worse than their own past.

        Compared against ITS OWN history, not against the other rules. A
        rule that is hard to keep and always half-kept is not degrading; a
        rule kept perfectly for twenty turns and broken twice just now is,
        and only the second one is news.
        """
        rows: List[Dict[str, Any]] = []
        for c in self.covenants:
            h = self.history.get(c.name) or []
            if len(h) < min_history:
                continue
            recent, past = h[-window:], h[:-window]
            if not past:
                continue
            r_keep = sum(recent) / len(recent)
            p_keep = sum(past) / len(past)
            rows.append({
                "covenant": c.name, "checks": len(h),
                "kept_before": round(p_keep, 3), "kept_recently": round(r_keep, 3),
                "delta": round(r_keep - p_keep, 3),
                "quote": c.quote,
            })
        rows.sort(key=lambda r: r["delta"])
        fade = [r for r in rows if r["delta"] < 0]
        return {
            "verdict": "FADING" if fade else "HELD",
            "window": window,
            "fading": fade,
            "stable": [r for r in rows if r["delta"] >= 0],
            "advise": ([f["quote"] for f in fade[:2]] if fade else []),
            "note": "re-inject the fading ones only; a rule repeated every "
                    "turn stops carrying information, which is how a long "
                    "session drifts with the system prompt still in place",
        }

    def check(self, text: str, asked: str = "", store: Any = None) -> Dict[str, Any]:
        """``asked`` is the turn that prompted this reply, if there was one.

        ``store`` turns on sibling inference — see `Covenant.check`.
        """
        hits = []
        for c in self.covenants:
            h = c.check(text, asked, store=store)
            # Only in-scope checks are recorded. Counting a turn that was
            # never about the rule as a "keep" makes every rule look healthy.
            if c.in_scope(text, asked):
                self.history.setdefault(c.name, []).append(h is None)
            if h:
                hits.append(h)
        return {
            # BROKEN is a finding about the REPLY, never about the user. The
            # user may have changed their mind one turn ago and this layer
            # cannot see intent, only text.
            "verdict": "BROKEN" if hits else "KEPT",
            "in_force": len(self.covenants),
            "violations": hits,
            "note": "a proposal to re-inject, not a judgment; the rule may "
                    "have been superseded and this cannot tell",
        }

    def save(self, path: Path) -> Dict[str, Any]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(
            json.dumps([c.as_dict() for c in self.covenants],
                       ensure_ascii=False), encoding="utf-8")
        return {"verdict": "ANSWER", "path": str(path),
                "covenants": len(self.covenants)}

    @classmethod
    def load(cls, path: Path) -> "Register":
        r = cls()
        p = Path(path)
        if p.is_file():
            for d in json.loads(p.read_text(encoding="utf-8")):
                r.covenants.append(Covenant(**d))
        return r


def siblings(store: Any, term: str, *, limit: int = 24,
             min_shared: int = 2, max_fanout: int = 60,
             max_common: float = 0.02) -> List[Tuple[str, float]]:
    """Terms that occupy the same slot as ``term``, by shared parents.

    Structural, not similar. Two terms are siblings here when the same cores
    hold both — 拘禁刑 and 罰金 are both facets of the articles that set a
    penalty, 過失 and 故意 of the ones that turn on intent. That is a fact
    about the store's geometry: no embedding, no nearest neighbour, no
    notion of meaning.

    It is what lets a covenant infer its own prohibitions. Registered by
    hand, 「TypeScriptを使う」 catches JavaScript only because somebody
    listed JavaScript, and the substitution nobody anticipated goes through.

    ## Raw co-occurrence returns hubs, and hubs are not siblings

    Unweighted, 拘禁刑 came back beside 法学, 百科, 日本, 規定 — the domain
    labels and the words every article uses. Two corrections, both the same
    idea `hierarchy.distinctive_terms` already applies:

      max_fanout   a parent holding hundreds of facets witnesses nothing;
                   a sentence that names a choice is small
      max_common   a term that is a facet of more than this share of all
                   cores is furniture, whatever it co-occurs with

    Scored by 1/fanout summed over shared parents, so agreement between two
    narrow articles outweighs agreement between two indexes.
    """
    labels = getattr(store, "source_labels", set()) or set()
    common = _too_common(store, max_common)
    # The store lowercases latin and a covenant is registered the way the
    # user wrote it, so 「TypeScript」 found no parents at all under
    # 実装言語 -> typescript. Everything downstream then worked perfectly on
    # an empty list.
    low = term.lower()
    parents = [(c, len(cr or ())) for c, cr in store.crosses.items()
               if c not in labels
               and (term in (cr or ()) or low in {f.lower() for f in (cr or ())})]
    parents = [(c, n) for c, n in parents if 0 < n <= max_fanout]
    if not parents:
        return []
    need = min(min_shared, len(parents))
    from collections import Counter
    score: Counter = Counter()
    seen: Counter = Counter()
    for c, n in parents:
        for f in (store.crosses.get(c) or ()):
            if f == term or f.lower() == low or f in labels or f in common:
                continue
            score[f] += 1.0 / n
            seen[f] += 1
    out = [(w, round(s, 4)) for w, s in score.most_common(limit * 3)
           if seen[w] >= need]
    return out[:limit]


_COMMON_CACHE: Dict[Tuple[int, float], Set[str]] = {}


#: Below this many cores, a SHARE threshold means nothing: on four cores a
#: facet in one of them is already 25%, so every candidate is furniture and
#: the sibling list comes back empty. Measured the hard way — the fixture
#: fork returned no siblings at all while the 54,244-core federation
#: returned 罰金 first.
_MIN_CORES_FOR_SHARE = 200


def _too_common(store: Any, share: float) -> Set[str]:
    """Facets of more than ``share`` of all cores. Furniture, not evidence."""
    key = (id(store), share)
    if key in _COMMON_CACHE:
        return _COMMON_CACHE[key]
    if len(store.crosses) < _MIN_CORES_FOR_SHARE:
        _COMMON_CACHE[key] = set()
        return _COMMON_CACHE[key]
    from collections import Counter
    labels = getattr(store, "source_labels", set()) or set()
    tally: Counter = Counter()
    for cross in store.crosses.values():
        for f in cross or ():
            if f not in labels:
                tally[f] += 1
    n = max(len(store.crosses), 1)
    _COMMON_CACHE[key] = {w for w, c in tally.items() if c / n > share}
    return _COMMON_CACHE[key]


def infer_forbidden(store: Any, required: Sequence[str], *,
                    limit: int = 8) -> Dict[str, List[str]]:
    """For each required term, the siblings that would substitute for it."""
    return {r: [w for w, _s in siblings(store, r)[:limit]] for r in required}


def loosely_in(term: str, text: str) -> bool:
    """Is the term present, allowing the word-forms the corpus writes?

    Exact substring misses 傷害罪 in a reply that wrote 傷害, which is the
    same one-character gap that made 「殺人罪の刑は」 unanswerable until the
    judgment was quantized. `ja_morph.variants` is the machinery already
    measured for it — 98.7% of 500 morphological variants reached the
    original core.

    Latin is compared case-insensitively because the store lowercases it and
    a reply does not. Without this the inference found javascript as a
    sibling of typescript and then failed to see 「JavaScript」 in the reply
    it was checking — the whole chain worked except the last comparison.
    """
    if term in text:
        return True
    if any(c.isascii() and c.isalpha() for c in term):
        if term.lower() in text.lower():
            return True
    try:
        from .ja_morph import variants
    except Exception:
        return False
    return any(v and v in text for v in variants(term, add=True, split=False))


#: Below this share of a reply's terms linked to what the conversation said
#: about the same subject, the reply has drifted off what was established.
#: The same floor `attest_llm` measured, for the same reason — it separated
#: grounded from free generation at 64.1% against 6.4%.
LINK_FLOOR = 0.30


def collapse(
    conversation: Any,
    reply: str,
    *,
    subjects: Optional[Sequence[str]] = None,
    floor: float = LINK_FLOOR,
) -> Dict[str, Any]:
    """Subjects the reply addresses that the conversation settled otherwise.

    ``conversation`` is a `conversation.Conversation`. For each subject the
    reply and the conversation share, the reply's terms are checked against
    what the CONVERSATION recorded under that subject — so a reply that has
    quietly reverted to generic knowledge about a subject the user already
    pinned down comes back with the turn to re-inject.
    """
    mem = getattr(conversation, "memory", None)
    levels = list(getattr(mem, "levels", []) or [])
    if not levels:
        return {"verdict": "UNKNOWN_NO_CONVERSATION", "subjects": 0}

    # Every layer, not just the top one. A frozen layer still holds what was
    # said — that is the whole reason overflow freezes instead of dropping —
    # and checking only level 0 would report the model as consistent with a
    # conversation whose relevant turn has aged out of it.
    labels: Set[str] = set()
    merged: Dict[str, Set[str]] = {}
    for lv in levels:
        labels |= getattr(lv, "source_labels", set()) or set()
        for c, cr in lv.crosses.items():
            merged.setdefault(c, set()).update(cr or ())

    rterms = terms_of(reply)
    # Case-fold for latin: the store ingests TypeScript as typescript, and a
    # reply that names it exactly must still match what the user set.
    fold = {c.lower(): c for c in merged}
    cand = list(subjects) if subjects else [
        fold[t.lower()] for t in rterms if t.lower() in fold]

    found: List[Dict[str, Any]] = []
    for s in cand:
        cross = {f for f in merged.get(s, ()) if f not in labels}
        if not cross:
            continue
        others = [t for t in rterms if t != s]
        if not others:
            continue
        lower = {f.lower() for f in cross}
        linked = [t for t in others if t in cross or t.lower() in lower]
        share = len(linked) / len(others)
        if share >= floor:
            continue
        loc = conversation.locate(s)
        mentions = loc.get("mentions", [])[:4]
        # Inject the TURNS, verbatim. Rebuilding a sentence from the facets
        # gave 「プロジェクト: 使い」 — the ingest keeps what a fact is about,
        # not how it was put, so anything reassembled from a cross is a
        # paraphrase this module has no business writing. The turn text is
        # what the user actually said, and it is what should go back in.
        turns = getattr(conversation, "turns", []) or []
        quotes = [turns[m["turn"]].text for m in mentions
                  if isinstance(m.get("turn"), int) and 0 <= m["turn"] < len(turns)]
        found.append({
            "subject": s,
            "link": round(share, 3),
            "conversation_said": sorted(cross)[:10],
            "reply_used": others[:10],
            "status": loc.get("status"),
            "mentions": mentions,
            "inject": quotes or [f"{s}: " + "、".join(sorted(cross)[:10])],
        })
    return {
        "verdict": "DRIFTED" if found else "CONSISTENT",
        "subjects": len(cand),
        "drifted": found,
        "floor": floor,
        "note": "the reply is about something this conversation already "
                "settled and does not use what was settled; re-injecting is "
                "a suggestion, not a correction",
    }
