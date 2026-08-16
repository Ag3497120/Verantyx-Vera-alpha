"""One boundary. The composition lives here, not in whoever is calling.

Why
---
The composition used to live in the caller. `veraModelTurn` in the IDE's
Swift decides — in Swift — whether a question is a difference, when to
re-ask with the last core as context, and which of the payload's thirty-one
fields become the reply. Measured today: the IDE knows 60 of the 99 doors
and its answering path uses three, so seventeen organs and about six
thousand lines are outside every question anyone asks. A different client
picks a different three and the same engine looks like a different
product — which is exactly what happened when one session measured the
CLI's v0 door and reported it as "the engine".

So the order moves here. A caller asks one thing and gets everything the
engine knows how to bring, and a new binding inherits the whole
composition instead of re-deriving a worse one.

Layered, never pooled
---------------------
Each stage hands the next a better question or annotates the answer.
Nothing here merges two notions of agreement into one vote — that
direction is measured at 6 failures out of 6, against 5 improvements out
of 5 for layering. `ask` still does the answering; this arranges what
reaches it and what is said about what comes back.

What is deliberately absent
---------------------------
No model is called. Nothing is written to the store. Both are properties
worth more than the reach they cost: the first keeps the same question
answering the same way forever, and the second keeps this file from ever
becoming a place where a claim quietly enters.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

#: Two subjects and a request to compare them. Closed, because a fuzzy
#: match here sends an ordinary question down the diff path and it comes
#: back as an abstention about the wrong thing.
_DIFF_MARKS = ("の違い", "の差", "と の違い", "はどう違う", "の違いは")
#: A follow-up shaped like it is about the last answer.
_DEICTIC = ("その", "それ", "この", "あの")


@dataclass
class Turn:
    """One question, and the record of what each stage did with it."""

    query: str
    verdict: str = ""
    text: str = ""
    core: str = ""
    door: str = ""
    tokens: List[str] = field(default_factory=list)
    origins: List[str] = field(default_factory=list)
    readings: Dict[str, Any] = field(default_factory=dict)
    remedy: str = ""
    stages: List[Dict[str, Any]] = field(default_factory=list)
    raw: Dict[str, Any] = field(default_factory=dict)

    def note(self, stage: str, fired: bool, note: str = "",
             changed: bool = False) -> None:
        self.stages.append({"stage": stage, "fired": fired,
                            "changed": changed, "note": note})

    def as_dict(self) -> Dict[str, Any]:
        out = dict(self.raw)
        out.update({"verdict": self.verdict, "text": self.text,
                    "core": self.core, "door": self.door,
                    "tokens": self.tokens, "origins": self.origins,
                    "readings": self.readings, "remedy": self.remedy,
                    "stages": self.stages})
        return out


def _looks_like_diff(q: str) -> Optional[Tuple[str, str]]:
    """(a, b) when the question compares two named things, else None."""
    if not any(m in q for m in _DIFF_MARKS):
        return None
    head = q
    for m in _DIFF_MARKS:
        head = head.split(m)[0]
    for sep in ("と", "、", "vs", "対"):
        if sep in head:
            a, _, b = head.partition(sep)
            a, b = a.strip(), b.strip()
            if a and b:
                return a, b
    return None


def _render(obj: Dict[str, Any]) -> str:
    """The readable answer, from the field that actually carries one.

    `text` alone is a token list — 「傷害罪 傷害 故意犯 狭義 204条」. The
    composed skeletons live under `written.sentences` and a descent's
    prose lives under `units[].definition`; a reply built from `text` was
    showing the reader the index instead of the answer.
    """
    units = obj.get("units")
    if isinstance(units, list):
        defs = [u.get("definition") for u in units
                if isinstance(u, dict) and u.get("definition")]
        if defs:
            return " ".join(defs)
    written = obj.get("written")
    if isinstance(written, dict):
        sents = written.get("sentences")
        if isinstance(sents, list):
            lines = [s.get("text") for s in sents[:3]
                     if isinstance(s, dict) and s.get("text")]
            if lines:
                return " ".join(lines)
    rendered = obj.get("rendered")
    if isinstance(rendered, dict) and rendered.get("text"):
        return str(rendered["text"])
    return str(obj.get("text") or "")


def _origins(obj: Dict[str, Any]) -> List[str]:
    """Sources, from wherever this door keeps them.

    `vera_ask` uses `facet_origin`; `vera_explain` keeps a source per unit.
    Reading only the first is why a console showed 「出典の記録なし」 over
    an answer that named 「jawiki:リンゴ ← りんご」.
    """
    out: List[str] = []
    fo = obj.get("facet_origin")
    if isinstance(fo, dict):
        for v in fo.values():
            if isinstance(v, list):
                out.extend(str(x) for x in v)
    units = obj.get("units")
    if isinstance(units, list):
        out.extend(str(u.get("source")) for u in units
                   if isinstance(u, dict) and u.get("source"))
    seen, uniq = set(), []
    for o in out:
        if o not in seen:
            seen.add(o)
            uniq.append(o)
    return uniq


def _readings(obj: Dict[str, Any]) -> Dict[str, Any]:
    """The numbers beside a verdict. A high agreement over thin evidence
    is a different thing from the same number over thick, so both travel."""
    r: Dict[str, Any] = {}
    if obj.get("agree_frac") is not None:
        r["agree"] = obj["agree_frac"]
    if obj.get("e_min") is not None:
        r["evidence_min"] = obj["e_min"]
    g = obj.get("grain")
    if isinstance(g, dict) and g.get("agree") is not None:
        r["grain"] = "%s/%s" % (g.get("agree"), g.get("of"))
    w = obj.get("witnesses")
    if isinstance(w, dict) and w.get("agree") is not None:
        r["witnesses"] = w["agree"]
    if obj.get("tier"):
        r["tier"] = obj["tier"]
    return r


def ask(query: str, vera: Any, *, last_core: str = "",
        domain: str = "", store_path: Any = None) -> Dict[str, Any]:
    """Everything the engine knows how to bring, for one question.

    `vera` is the loaded stack, passed in rather than loaded here: which
    store answers is the host's decision, and a module that chose for
    itself is how a probe measures the default store and reports it as the
    engine.
    """
    t = Turn(query=query.strip())
    q = t.query

    # ── before the store is asked ────────────────────────────────────

    from .intent_frames import parse as _intent
    framed = _intent(q)
    if framed.get("verdict") == "INTENT":
        t.note("intent", True, "op=%s" % framed["op"], changed=True)
        t.verdict, t.door = "INTENT", "intent_frames"
        t.raw = framed
        t.text = "指示として読みました。"
        return t.as_dict()
    t.note("intent", True, "問いとして扱う")

    # Repair before anything reads it. A misspelt subject fails every
    # later stage identically to a subject the store has never heard of,
    # and those are not the same problem. Never silent: the repair is a
    # named hand-off, and the original is kept beside it.
    try:
        from .meaning_assets import lattice as _lat
        from .meaning_assets import vocab as _voc
        from .typo_recovery import recover
        core_term = q.replace("とは", "").replace("？", "").strip()
        tr = recover(core_term, lattice=_lat(), vocab=_voc())
        if tr.get("verdict") == "TYPO_CANDIDATE" and tr.get("candidates"):
            best = tr["candidates"][0]["word"]
            t.note("typo", True, "%s → %s" % (core_term, best), changed=True)
            q = q.replace(core_term, best)
        else:
            t.note("typo", True, str(tr.get("verdict")))
    except Exception as exc:
        t.note("typo", False, "%s: %s" % (type(exc).__name__, str(exc)[:50]))

    try:
        from .stage_split import split as _split
        st = _split(q)
        if st.get("verdict") == "STAGED" and len(st.get("stages", [])) > 1:
            t.note("stage_split", True, st.get("chain", ""), changed=True)
        else:
            t.note("stage_split", True, str(st.get("verdict")))
    except Exception as exc:
        t.note("stage_split", False, str(exc)[:60])

    # ── a difference is a different question ─────────────────────────

    # Arithmetic is a different kind of question, answered by an exact
    # organ rather than by a census. It is asked first because a census
    # asked about 「3+4は」 would return whatever core happens to share a
    # character with it.
    try:
        import re as _re
        from .math_sim import math_ask
        # 「3+4は」 parses as UNKNOWN_UNPARSED with the particle attached.
        # Cut to the expression itself before handing over — a hand-off
        # normalisation, not an interpretation: if no run of arithmetic
        # characters is present, nothing is passed and nothing is claimed.
        _mx = _re.search(r"[0-9０-９][-+*/^=()0-9０-９xX\s.,]*", q)
        m = math_ask(_mx.group(0).strip()) if _mx else {"verdict": "UNKNOWN_NO_EXPR"}
        if not str(m.get("verdict", "UNKNOWN")).startswith("UNKNOWN"):
            t.note("math", True, str(m.get("verdict")), changed=True)
            t.raw, t.door = m, "math_sim"
            t.verdict = str(m.get("verdict"))
            t.text = str(m.get("text") or m.get("answer") or "")
            return t.as_dict()
        t.note("math", True, str(m.get("verdict")))
    except Exception as exc:
        t.note("math", False, "%s: %s" % (type(exc).__name__, str(exc)[:50]))

    # A theorem's proof status is not a census question either: the
    # answer is a Lean kernel run that happened, and 73,881 of them are
    # recorded. Absence of a witness is never a claim of falsehood.
    try:
        from .mathlib_witness import looks_like_declaration
        from .mathlib_witness import lookup as _ml
        _cand = q.replace("とは", "").replace("は証明済みですか", "").strip()
        if looks_like_declaration(_cand):
            w = _ml(_cand)
            if not str(w.get("verdict", "")).startswith("UNKNOWN"):
                t.note("mathlib", True, str(w.get("verdict")), changed=True)
                t.raw, t.door = w, "mathlib_witness"
                t.verdict = str(w.get("verdict"))
                return t.as_dict()
            t.note("mathlib", True, str(w.get("verdict")))
        else:
            t.note("mathlib", True, "宣言名ではない")
    except Exception as exc:
        t.note("mathlib", False,
               "%s: %s" % (type(exc).__name__, str(exc)[:50]))

    pair = _looks_like_diff(q)
    if pair:
        try:
            from .meaning_assets import aliases, empty_shelf, profiles, senses
            from .structural_diff import diff as _diff
            from .meaning_assets import lattice as _lat
            d = _diff(pair[0], pair[1], profiles=profiles(),
                      aliases=aliases(), lattice=_lat(),
                      shelf=empty_shelf(), senses=senses())
            t.note("diff", True, "%s / %s" % pair, changed=True)
            t.raw, t.door = d, "structural_diff"
            t.verdict = str(d.get("verdict", "UNKNOWN"))
            t.text = _render(d)
            return t.as_dict()
        except Exception as exc:
            # Recorded, never swallowed: a stage that failed silently is
            # indistinguishable from one that was never built, and that
            # is how organs stay dead.
            t.note("diff", False, "%s: %s" % (type(exc).__name__, str(exc)[:60]))
    else:
        t.note("diff", True, "比較の形ではない")

    # ── the census ───────────────────────────────────────────────────

    obj = dict(vera.ask(q))
    t.door = "vera_ask"
    t.note("ask", True, str(obj.get("verdict")))

    # Context as a VISIBLE operation. A short follow-up, or one opening
    # with a deictic, is re-asked with the last answered core added — and
    # the completion is printed, because an invisible context resolution
    # is the same shape of lie as an invisible ingest.
    if str(obj.get("verdict", "")).startswith("UNKNOWN") and last_core and (
            len(q) <= 10 or q.startswith(_DEICTIC)):
        stripped = q
        for d in _DEICTIC:
            if stripped.startswith(d):
                stripped = stripped[len(d):]
        retry = dict(vera.ask("%s %s" % (last_core, stripped)))
        if not str(retry.get("verdict", "")).startswith("UNKNOWN"):
            obj = retry
            t.note("context", True,
                   "直近の核「%s」を条件に補完" % last_core, changed=True)
        else:
            t.note("context", True, "補完しても届かない")

    # Descent when the census has nothing. Constructed, and it says so in
    # its own payload — this is the only route by which 「りんごとは」 has
    # ever answered.
    if str(obj.get("verdict", "")).startswith("UNKNOWN"):
        try:
            from .meaning_assets import aliases as _al
            from .meaning_assets import defs as _defs
            from .meaning_assets import lattice as _lat
            from .meaning_descent import descend
            term = q.replace("とは", "").replace("？", "").strip()
            d = descend(term, lattice=_lat(), defs=_defs(), aliases=_al())
            if d and not str(d.get("verdict", "")).startswith("UNKNOWN"):
                obj, t.door = d, "meaning_descent"
                t.note("descend", True, str(d.get("verdict")), changed=True)
            else:
                t.note("descend", True, "届かない")
        except Exception as exc:
            t.note("descend", False,
                   "%s: %s" % (type(exc).__name__, str(exc)[:60]))
    else:
        t.note("descend", True, "不要")

    # ── what is said about what came back ────────────────────────────

    t.raw = obj
    t.verdict = str(obj.get("verdict", "UNKNOWN"))
    t.core = str(obj.get("core") or "")
    t.text = _render(obj)
    t.tokens = list(obj.get("tokens") or [])
    t.origins = _origins(obj)
    t.readings = _readings(obj)
    t.remedy = str(obj.get("remedy") or "")

    # Facets become claims where a cue licenses it, and stay facets where
    # none does. Annotation only — nothing here changes the verdict.
    try:
        from .arm_schema import classify_arm
        claims = []
        for f in t.tokens[:8]:
            arm = classify_arm("%sは%sである" % (t.core, f)) if t.core else None
            if arm:
                claims.append("%s --%s--> %s" % (t.core, arm, f))
        t.note("arms", True, " / ".join(claims[:3]) if claims
               else "手がかり無し — 面のまま")
    except Exception as exc:
        t.note("arms", False, str(exc)[:60])

    # A refusal says what is missing; the gap graph may already hold a
    # named ticket for it. Annotation only — a gap does not become a fact
    # by being mentioned beside one.
    if t.verdict.startswith(("UNKNOWN", "ABSTAIN", "AMBIGUOUS",
                             "NOT_", "UNGROUNDED")):
        try:
            if store_path is None:
                raise ValueError("store_path 未指定（欠落台帳は店に紐づく）")
            from .gap_graph import GapGraph, gap_graph_path
            g = GapGraph.load(gap_graph_path(store_path))
            hit = g.find_by_scope_subject("meaning", t.core) if t.core else None
            t.note("gaps", True,
                   ("既知の欠落 %s (%s)" % (hit.gap_id, hit.status)) if hit
                   else "この核に登録された欠落は無い")
        except Exception as exc:
            t.note("gaps", False,
                   "%s: %s" % (type(exc).__name__, str(exc)[:50]))

    # Nothing readable came back but the store holds the subject: compose
    # one clause from its observed frame rather than showing the reader a
    # token list. Marked `constructed`, never testimony.
    if not t.text and t.core:
        try:
            from .compose_frame import Tables, compose
            tab = Tables.indexed(domain)
            if tab is not None:
                c = compose(t.core, tab)
                if c.get("text"):
                    t.text = c["text"]
                    t.raw["constructed"] = True
                    t.note("compose", True, c["text"][:60], changed=True)
                else:
                    t.note("compose", True, str(c.get("verdict")))
        except Exception as exc:
            t.note("compose", False,
                   "%s: %s" % (type(exc).__name__, str(exc)[:50]))

    if t.remedy:
        t.note("remedy", True, t.remedy[:80])
    return t.as_dict()
