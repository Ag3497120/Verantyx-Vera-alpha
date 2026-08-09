"""Events as cores — how a three-place fact fits a two-place store.

`CrossStore` maps core → facet. That is unary attribution, and it is the
right shape for "this town has no water". It is the wrong shape for "A
threatened B", because the sentence

    「甲は乙を脅迫した。」

lands as core=甲, facets={乙, 脅迫} — which cannot be told apart from "甲 is
associated with 乙 and with threatening". Chain two of them and the problem
compounds: 甲→乙 (threat) and 乙→丙 (injury) are two crosses with no path
between them, so a question about 甲 and 丙 reaches nothing.

The fix is not a bigger store. It is to stop pushing a three-place fact into
a two-place slot and instead **reify the event**: give the happening its own
core, and let the participants be its facets, tagged by the role they played.

    事象一 → 主体甲、対象乙、行為脅迫
    事象二 → 主体乙、対象丙、行為傷害、原因事象一

Every one of those is unary attribution about the event, which is exactly
what the store already does. No schema change; the change is at ingest.
This is Davidsonian event semantics, and naming it is more useful than
inventing a term: the point is not novelty, it is that a known solution
fits the existing geometry — the arms of a cross hold the roles.

## What this module will and will not extract

Japanese predicate-argument structure is not solved here and pretending
otherwise would put guesses into a legal record. `extract_events` handles
the plain pattern — a subject particle, an optional object particle, and a
final predicate — and returns a TYPED SKIP for everything else, with the
reason. Light-verb constructions (「暴行を加えた」, where the act is the
object and the predicate is nearly empty) are explicitly not handled: the
extractor would have to decide that 暴行 rather than 加 is the act, and a
wrong decision there mislabels who did what.

## Roles are composite tokens, and that has a sharp edge

A facet is a string, so a role-tagged participant is `対象丙` — one token.
That works for storage and retrieval, and it breaks a coverage gate that
has not been told about it: asked 「事象二 対象甲」, the store has never
seen the token `対象甲`, so a gate that treats unseen tokens as vocabulary
gaps lets the answer through. But for a composite, *unseen means the
relation does not hold* — the opposite reading. `role_refutation` is what
closes that, and it is the reason this module exports a splitter as well as
a builder.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

#: Particle → the role the preceding phrase played. Deliberately small: each
#: entry is a claim about Japanese that a lawyer could check, and a long
#: speculative table would be a liability in exactly the setting this exists
#: for. に/へ is 相手 rather than 対象 because "gave TO B" and "hit B" are
#: different relations and collapsing them would invent a fact.
ROLE_BY_PARTICLE: Dict[str, str] = {
    "は": "主体",
    "が": "主体",
    "を": "対象",
    "に": "相手",
    "へ": "相手",
    "で": "場所",
    "から": "起点",
    "まで": "終点",
}

#: The predicate's role.
ACT_ROLE = "行為"

#: Cause, written by the caller rather than extracted — a causal link
#: between two events is a legal conclusion, not a reading of one sentence.
CAUSE_ROLE = "原因"

ROLES: Tuple[str, ...] = tuple(sorted(
    set(ROLE_BY_PARTICLE.values()) | {ACT_ROLE, CAUSE_ROLE},
    key=len, reverse=True,
))

#: Content runs: kanji, katakana, latin, digits. Kana are separators AND
#: role markers, which is why this module cannot reuse `ja_content_runs` —
#: that function throws away exactly the characters the roles live in.
_CONTENT = re.compile(r"[㐀-䶿一-鿿ァ-ヺー々〆A-Za-z0-9]+")
_KANA = re.compile(r"[ぁ-ゟ]+")

#: Predicates that carry no act of their own; the act is the object. Not
#: handled, and listed so the skip reason can say WHICH construction it saw
#: rather than "unparsed".
_LIGHT_VERBS = frozenset({"加", "行", "為", "与", "負"})


@dataclass
class Event:
    """One happening, with its participants tagged by role."""

    eid: str
    roles: Dict[str, str] = field(default_factory=dict)
    sentence: str = ""

    def facts(self) -> List[str]:
        """Role-tagged tokens, in a stable order."""
        return [f"{r}{v}" for r, v in sorted(self.roles.items())]

    def sentences(self) -> List[str]:
        """Reified sentences an ordinary Japanese ingest can consume."""
        return [f"{self.eid}は{r}{v}である。" for r, v in sorted(self.roles.items())]


def runs_with_particles(text: str) -> List[Tuple[str, str]]:
    """[(content run, the kana that followed it)] — the particle kept.

    `lang.ja_content_runs` returns only the content, because for claim
    placement the particle is noise. Here it is the entire signal.
    """
    out: List[Tuple[str, str]] = []
    pos = 0
    for m in _CONTENT.finditer(text):
        if m.start() < pos:
            continue
        tail = _KANA.match(text, m.end())
        out.append((m.group(0), tail.group(0) if tail else ""))
        pos = m.end()
    return out


def extract_events(
    text: str, *, prefix: str = "事象", start: int = 1,
) -> Dict[str, Any]:
    """Reify what can be reified; say plainly what could not.

    Returns {"events": [...], "skipped": [{"sentence", "reason"}...]}.
    A skip is a first-class outcome here, not an error path — a legal
    record built from the sentences an extractor happened to understand,
    with the rest silently dropped, is worse than one that names its gaps.
    """
    events: List[Event] = []
    skipped: List[Dict[str, str]] = []
    n = start
    for raw in re.split(r"[。\n]", text or ""):
        s = raw.strip()
        if not s:
            continue
        pairs = runs_with_particles(s)
        if len(pairs) < 2:
            skipped.append({"sentence": s, "reason": "too_few_phrases"})
            continue
        act, act_tail = pairs[-1]
        if act in _LIGHT_VERBS:
            skipped.append({"sentence": s, "reason": f"light_verb:{act}"})
            continue
        if act_tail and act_tail in ROLE_BY_PARTICLE:
            # The last phrase is still an argument, so the predicate is
            # missing or was written in a form this does not read.
            skipped.append({"sentence": s, "reason": "no_predicate"})
            continue
        roles: Dict[str, str] = {}
        unread: List[str] = []
        for run, tail in pairs[:-1]:
            role = ROLE_BY_PARTICLE.get(tail)
            if role is None:
                unread.append(f"{run}{tail}")
                continue
            # First writer wins: 「甲は乙は…」 is not two subjects, it is a
            # construction this does not read, and overwriting would quietly
            # keep the wrong one.
            roles.setdefault(role, run)
        if "主体" not in roles:
            skipped.append({"sentence": s, "reason": "no_subject"})
            continue
        roles[ACT_ROLE] = act
        ev = Event(eid=f"{prefix}{n}", roles=roles, sentence=s)
        if unread:
            skipped.append({"sentence": s,
                            "reason": "partial:" + ",".join(unread)})
        events.append(ev)
        n += 1
    return {"events": events, "skipped": skipped}


def link_cause(effect: Event, cause: Event) -> None:
    """Record that one event caused another.

    Not extracted from text. Whether a threat caused an injury is a finding,
    and an extractor that inferred it from sentence order would be writing
    conclusions into what is supposed to be a record of claims.
    """
    effect.roles[CAUSE_ROLE] = cause.eid


def split_role(token: str) -> Tuple[Optional[str], str]:
    """`対象丙` → ("対象", "丙"); a plain token → (None, token).

    Longest role first, so 主体 is not shadowed by a shorter prefix if the
    table ever grows one.
    """
    for role in ROLES:
        if token.startswith(role) and len(token) > len(role):
            return role, token[len(role):]
    return None, token


def role_refutation(
    store: Any, core: str, token: str,
) -> Optional[Dict[str, Any]]:
    """Does ``core`` positively contradict the role claim in ``token``?

    Asked 「事象二 対象甲」 where the store holds 対象丙, the honest reply is
    not "I have never seen 対象甲" — it is "the object is 丙". A coverage
    gate that cannot tell those apart answers a question it has refuted.

    Returns None when the role is absent from this core, because then the
    store genuinely has nothing to say; a missing role is a gap, and only a
    role that is FILLED DIFFERENTLY is a refutation.
    """
    role, value = split_role(token)
    if role is None:
        return None
    cross = (getattr(store, "crosses", {}) or {}).get(core) or {}
    held = {f[len(role):] for f in cross
            if f.startswith(role) and len(f) > len(role)}
    if not held or value in held:
        return None
    return {"role": role, "asked": value, "held": sorted(held), "core": core}


def ingest_events(store: Any, events: List[Event], *, source: str = "") -> int:
    """Place reified events through the ordinary Japanese ingest path.

    Uses `ingest_documents` rather than writing crosses directly, so events
    get the same sentence splitting, provenance and gates as any other
    document. A second write path into the store is how two readers of the
    same corpus start disagreeing about what it said.
    """
    from .document_ingest import Document, ingest_documents

    text = "".join(s for ev in events for s in ev.sentences())
    if not text:
        return 0
    ingest_documents(store, [Document(source=source or "events", text=text)])
    return len(events)
