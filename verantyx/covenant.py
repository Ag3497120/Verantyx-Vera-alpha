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

    def check(self, text: str, asked: str = "") -> Optional[Dict[str, Any]]:
        if not self.in_scope(text, asked):
            return None
        used = [f for f in self.forbids if f in text]
        missing = [r for r in self.requires if r not in text]
        if not used and not missing:
            return None
        return {
            "covenant": self.name,
            "forbidden_used": used,
            "required_missing": missing,
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
    """The covenants in force, and the turns they came from."""

    covenants: List[Covenant] = field(default_factory=list)

    def add(self, c: Covenant) -> Covenant:
        self.covenants.append(c)
        return c

    def check(self, text: str, asked: str = "") -> Dict[str, Any]:
        """``asked`` is the turn that prompted this reply, if there was one."""
        hits = [h for h in (c.check(text, asked) for c in self.covenants) if h]
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
