"""One entry point: a question in, a typed answer with its route out.

Everything measured this session sits in layers, and the layers only work
layered — six pooled combinations were worse, six layered ones better. This
wires them in the order the measurements put them:

    1  language     detect, and route to the sovereign built from that
                    language. Never a census across two: mixing them
                    answered superconductivity with contract
    2  deictic      「今日の天気は」 has no answer in a store with no clock.
                    Typed out before anything is looked up
    3  staircase    grain and grammar over the cores of that sovereign,
                    six settings for kanji and two for latin — character
                    windows collide at 26 letters and discriminate at 2,000
    4  core         the inference core, entered directly or seeded by the
                    staircase when it cannot enter at all. Sections converge
                    and the axis words along the agreed paths are the answer
    5  reach        when the term is not held: split it into units the
                    corpus attests (10.4% overlap) before falling back on a
                    longer word that contains it (4.5%)
    6  words        the path becomes the content and a harvested template
                    supplies only the form
    7  remedy       what an expert would register to close a refusal, and
                    which refusals should not be closed

Each stage hands the next something typed. None of them votes with another.

## Every answer says how it was reached

`ANSWER` is the core converging on the question as asked. `SEEDED` means the
staircase had to name the subject first. `UNITS` and `CONTAINMENT` are two
different ways of landing near a word the store never held, measured 16.3x
and 3.9x over chance, and reported apart because a reader discounting one
needs to know which they have.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class Vera:
    """The federation, its sovereigns, and everything built over them."""

    stores: Dict[str, Any] = field(default_factory=dict)
    judges: Dict[str, Any] = field(default_factory=dict)
    models: Dict[str, Any] = field(default_factory=dict)
    writer: Optional[Any] = None
    built: Dict[str, Any] = field(default_factory=dict)

    def add(self, lang: str, store: Any) -> "Vera":
        from .graded import GradedJudge, settings_for
        from .reach import build_model

        probe = "これは日本語です。" if lang == "ja" else "This is English."
        self.stores[lang] = store
        self.judges[lang] = GradedJudge(settings_for(probe)).build(store)
        self.models[lang] = build_model(store)
        self.built[lang] = {"cores": len(store.crosses),
                            "settings": len(self.judges[lang].settings)}
        return self

    def ask(self, query: str, *, limit: int = 3) -> Dict[str, Any]:
        from .lang import detect
        from .remedy import remedy
        from .stacked import ask as stacked_ask, in_words

        lang = detect(query)
        if lang == "latin" and "en" in self.stores:
            lang = "en"
        store = self.stores.get(lang)
        if store is None:
            out = {"verdict": "UNKNOWN_LANGUAGE_NOT_HELD", "language": lang,
                   "have": sorted(self.stores)}
            return {**out, "remedy": remedy(out)}

        judge = self.judges[lang]
        graded = judge.ask(query)
        # A question about now cannot be answered by a store with no clock,
        # and the deictic is in the QUESTION — so it is settled before any
        # lookup rather than after one succeeds.
        if graded.get("verdict") == "UNKNOWN_TIME_DEPENDENT":
            return {**graded, "language": lang, "remedy": remedy(graded)}

        core = stacked_ask(store, query, judge=judge)
        out: Dict[str, Any] = {**core, "language": lang,
                               "coverage": graded.get("coverage"),
                               "as_core": graded.get("as_core"),
                               "as_facet_only": graded.get("as_facet_only"),
                               "missing": graded.get("missing")}

        if out.get("text") and self.writer is not None:
            out["written"] = in_words(store, out, self.writer, limit=limit)

        # Nothing was held. Try to land NEAR the terms rather than refuse
        # outright, and say by which route — the two are not the same claim.
        if not out.get("text"):
            from .reach import reach

            landed = []
            for t in (graded.get("terms") or []):
                r = reach(store, t, model=self.models.get(lang), judge=judge)
                if r["verdict"] in ("UNITS", "CONTAINMENT"):
                    landed.append(r)
            if landed:
                out["reached"] = landed
        out["remedy"] = remedy(out)
        return out

    def report(self) -> Dict[str, Any]:
        return {"sovereigns": dict(self.built),
                "writer": (self.writer.report() if self.writer else None)}


def load(root: Any = None) -> Vera:
    """The built federation, the English sovereign, and the writer."""
    import pickle
    from pathlib import Path

    from .cross_store import CrossStore
    from .writer import Writer

    root = Path(root or (Path.home() / "Projects" / "vera-corpus"))
    v = Vera()

    doms = pickle.loads((root / "build" / "federation.pkl").read_bytes())
    ja = CrossStore()
    for d in doms:
        for s in doms[d].values():
            ja.source_labels |= getattr(s, "source_labels", set())
            for c, cr in s.crosses.items():
                ja.crosses.setdefault(c, {}).update(cr)
    v.add("ja", ja)

    en_path = root / "build" / "english.pkl"
    if en_path.is_file():
        v.add("en", pickle.loads(en_path.read_bytes()))

    wpath = root / "build" / "writer.json"
    if wpath.is_file():
        v.writer = Writer.load(wpath)
    return v
