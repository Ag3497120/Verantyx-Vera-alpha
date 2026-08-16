"""The engine suspends and asks for language. It never calls a model.

The inversion, and why it is built this way
-------------------------------------------
MCP puts the model in charge — it picks which door to call, and the doors
it does not know about never run. Measured today: the IDE knows 60 of the
99 doors and its answering path uses three, so seventeen organs and about
six thousand lines sit outside every question anyone asks.

Inverting that is right. Calling a model API from inside `ask` is not.
That would put a non-deterministic call in the deterministic core, and the
same question could then return different answers — which is the one
property this engine has that an LLM cannot, and it is not for sale.

So the engine stops and says what it needs:

    ask(query)            → NEEDS_LANGUAGE, a typed request, a token
       the host fulfils it   (MCP sampling, a provider API, a person)
    resume(token, text)   → the engine continues

Given the same fulfilment, the same answer. The model is outside, what it
said is data the engine received, and both are recorded.

`ask_back.py` already has this shape for people — a typed refusal becomes
a question with candidates, and an answer comes back marked as testimony.
This is the same mechanism with one more kind of receiver, and the same
rule applies: what comes back is `support+:model`, never a measurement.

What the model may and may not do
---------------------------------
It is handed one string and asked for another. It does not receive the
store, the doors, or any ability to act, and nothing it returns is
written anywhere — it may only re-form a QUERY, which the engine then
answers by its own rules.

That boundary is measured, not cautious. A 27B model given the loop
produced forty paragraphs and no tool call; the loop belongs on this side.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence

NEEDS_LANGUAGE = "NEEDS_LANGUAGE"

#: The only things the engine will ask a model for. Closed, like every
#: other table here: a request outside this set would be the engine
#: inventing a use for a model it has not measured.
WANTS = {
    "rephrase": "同じ問いの別の言い方（表記ゆれ・別名・言い換え）",
    "split": "複合的な問いを、独立した問いに割る",
}


@dataclass(frozen=True)
class Request:
    """What the engine stopped for, in terms the host can fulfil."""

    want: str
    subject: str
    because: str          # the typed refusal that caused the suspension
    n: int = 3

    @property
    def token(self) -> str:
        """Deterministic from the content, so the same suspension in the
        same state yields the same token — a random id would make a
        replayed run diverge from the original on the identifier alone."""
        raw = "%s|%s|%s|%d" % (self.want, self.subject, self.because, self.n)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()[:16]

    def as_dict(self) -> Dict[str, Any]:
        return {"verdict": NEEDS_LANGUAGE, "want": self.want,
                "want_means": WANTS.get(self.want, ""),
                "subject": self.subject, "because": self.because,
                "n": self.n, "resume": self.token,
                "note": "モデルには問いの言い換えだけを頼む。"
                        "返ってきたものは店に書かない"}


def suspend(subject: str, because: str, *,
            want: str = "rephrase", n: int = 3) -> Dict[str, Any]:
    """Stop and say what would let the engine continue.

    A refusal outside the set this mechanism can act on returns None-ish:
    `UNKNOWN_CAPTURE_EMPTY` is not a rephrasing problem, and asking a model
    to reword the query would produce a plausible answer to the wrong
    question.
    """
    if want not in WANTS:
        return {"verdict": "UNKNOWN_WANT_NOT_MAPPED", "want": want,
                "mapped": sorted(WANTS)}
    return Request(want=want, subject=subject, because=because, n=n).as_dict()


@dataclass
class Resumption:
    """One suspension, fulfilled. Kept so a run can be replayed exactly."""

    token: str
    request: Dict[str, Any]
    given: List[str] = field(default_factory=list)
    tried: List[Dict[str, Any]] = field(default_factory=list)
    settled: Optional[Dict[str, Any]] = None

    def facets(self) -> List[str]:
        """What may be recorded about this, if a caller records anything.

        The model's contribution lands as testimony sourced to `model`,
        never as a measurement — same rule as a person's answer in
        `ask_back`. The store can then always say which half an answer
        stood on.
        """
        return ["support+:model:%s" % g[:100] for g in self.given]


def _content_runs(text: str) -> List[str]:
    """Content runs of a query, script-split, particles and 「とは」 gone."""
    from .lang import ja_content_runs
    return [r for r in ja_content_runs(text or "") if r]


def _subject(text: str) -> str:
    """The subject of a question: its first content run.

    `ja_chosen_core` was tried first and does not serve here — measured
    2026-08-16, it returns 冷 for 「こおりは冷たい」 and None for
    「りんごとは」. It reads declaratives (`Xは…である`) and a question is
    not one. Japanese is topic-first, so the first content run is the
    subject; that is a fact about the language rather than a rule fitted
    to these probes, and it was registered before the numbers.
    """
    runs = _content_runs(text)
    return runs[0] if runs else ""


def _reading(word: str) -> str:
    """The word's reading, or "" — kana↔kanji is a reading identity.

    りんご/リンゴ and じこう/時効 are the same word written differently,
    and no organ here holds them: `aliases` are page redirects and
    `typo_recovery` is edit distance within one script. unidic's kanaBase
    is the same instrument the case frames and the polarity gate use.
    """
    try:
        import fugashi
        global _TAG
        try:
            _TAG
        except NameError:
            _TAG = fugashi.Tagger()
        toks = list(_TAG(word))
        if not toks:
            return ""
        return "".join(getattr(t.feature, "kanaBase", "") or t.surface
                       for t in toks)
    except Exception:
        return ""


def certify(original: str, candidate: str) -> Optional[str]:
    """Which organ vouches that `candidate` asks about `original`, if any.

    Not a similarity score and not a threshold — the same instrument used
    everywhere else here: ask the dictionary, and refuse what it cannot
    vouch for.

    The failure this exists for, measured 2026-08-16: 「こおりは冷たい」
    was rephrased 「氷 冷たい」, which answered ANSWER with core 「ブラック」
    and no 「冷」 in any facet. An honest refusal became an answer about
    something else, because `resume` read "stopped refusing" as "answered".
    """
    # ONE run on each side — the subject. Everything else in a question is
    # free to change; that is what a rephrasing is. The previous gate
    # compared whole queries and certified 「氷 冷たい」 against
    # 「こおりは冷たい」 on the shared 冷, which is the predicate.
    sa, sb = _subject(original), _subject(candidate)
    if not sa or not sb:
        return None
    a, b = [sa], [sb]

    # 1. The same subject, written the same way.
    if sa == sb:
        return "identity"

    # 1b. The same subject, written in a different script. Reading
    #     identity, and only when at least one side tags as one token —
    #     a multi-token reading match would let two unrelated compounds
    #     that happen to read alike through.
    ra, rb = _reading(sa), _reading(sb)
    if ra and ra == rb:
        return "reading"

    # 2. An alias Wikipedia itself wrote (941,604 redirects).
    try:
        from .meaning_assets import aliases
        al = aliases()
        for x in a:
            t = al.get(x)
            if t and t in b:
                return "alias"
        for y in b:
            t = al.get(y)
            if t and t in a:
                return "alias"
    except Exception:
        pass

    # 3. A typo the recovery organ can repair into the other (84.8%@5,
    #    zero false fires on in-vocabulary words).
    try:
        from .meaning_assets import lattice, vocab
        from .typo_recovery import recover
        lat, voc = lattice(), vocab()
        for x in a:
            r = recover(x, lattice=lat, vocab=voc)
            if r.get("verdict") == "TYPO_CANDIDATE":
                if any(c.get("word") in b for c in r.get("candidates", [])):
                    return "typo"
    except Exception:
        pass

    # 4. A registered sense of the same surface.
    try:
        from .sense_split import resolve
        for x in a:
            r = resolve(x)
            cores = {r.get("core", "")} | {
                s.get("core", "") for s in (r.get("other_senses") or [])}
            if cores & set(b):
                return "sense"
    except Exception:
        pass

    return None


def resume(request: Dict[str, Any], given: Sequence[str],
           answer: Callable[[str], Dict[str, Any]]) -> Dict[str, Any]:
    """Continue with what the host supplied. The engine decides, as ever.

    `answer` is the engine's own question-answering function — passed in
    rather than imported, so this module cannot reach into the store and
    cannot be the place a shortcut appears later.

    Every candidate is tried in the order given and the FIRST that stops
    refusing wins. Not the best: ranking re-formed queries would mean
    scoring the model's suggestions, and this module has no basis to do
    that. Order is the host's statement of preference and it is honoured.
    """
    r = Resumption(token=str(request.get("resume", "")), request=dict(request))
    for cand in given:
        c = (cand or "").strip()
        if not c:
            continue
        r.given.append(c)
        # Certify BEFORE asking. An uncertified rephrasing is not a worse
        # question, it is a DIFFERENT one, and its answer would be about
        # something the person never asked.
        by = certify(str(request.get("subject", "")), c)
        if by is None:
            r.tried.append({"query": c, "verdict": "UNCERTIFIED",
                            "note": "どの器官も、元の問いの変種だと請け合わない"})
            continue
        out = answer(c)
        v = str(out.get("verdict", ""))
        r.tried.append({"query": c, "verdict": v, "certified_by": by})
        if not v.startswith(("UNKNOWN", "AMBIGUOUS")):
            enriched = dict(out)
            enriched["resumed_from"] = request.get("subject")
            enriched["reformed_query"] = c
            enriched["language_by"] = "model"
            enriched["certified_by"] = by
            enriched["tried"] = r.tried
            # The mark travels with the answer: this reached the store
            # through a re-forming that a model proposed, and a reader
            # deciding how much to trust it should be able to see that
            # without opening anything.
            enriched["note"] = ("問いの言い換えで届いた答え。"
                                "言い換えはモデル、判定は店")
            r.settled = enriched
            return enriched

    # Nothing re-formed reached anything. The ORIGINAL refusal is returned,
    # not a new one: inventing a verdict for the occasion would hide that
    # the store's answer never changed.
    return {"verdict": request.get("because", "UNKNOWN"),
            "subject": request.get("subject"),
            "tried": r.tried,
            "language_by": "model",
            "note": "言い換えを試したが、どれも届かなかった。"
                    "元の型付き拒否をそのまま返す"}
