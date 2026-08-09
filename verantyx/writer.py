"""The generation stack, assembled — and kept away from the answer path.

Five pieces were built and measured separately. This wires them, because a
caller assembling them by hand gets the order wrong: a walk without a
vocabulary fills sentences with 1980アイコ, a vocabulary attested on the
wrong register loses 場合 and 被保険者, and forms harvested from one corpus
write in that corpus's voice whatever the content is.

    vocabulary   which facets are words              33,126 of 123,734
    forms        sentence shapes with typed holes    478
    selection    what the corpus puts in each slot   382,884 triples
    trace        where the walk went, held outside   subject held 5.3 steps
    compose      one sentence per step

## The order is load-bearing

Attestation must come from the SAME register as the stores. 626MB of statute
raised the facet count 1.4x and the vocabulary not at all, because every
attesting corpus was encyclopedia prose; adding the statute bodies took the
vocabulary from 12,348 to 33,126 and the selection triples from 8,805 to
382,884. A `Writer` therefore takes its corpora explicitly and reports what
each contributed, so a build that judges law by Wikipedia is visible rather
than silent.

## Never on the answer path

Nothing here is imported by `gather`, `descend`, `consensus_over_store` or
any verdict. A generated sentence is a draft with two citations — one for
the content, one for the shape — and neither makes it true. The whole point
of the closure elsewhere is that a reader can trust a citation; a sentence
this module assembled must never be able to arrive where one is expected.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .compose_ja import Form, compose, harvest, learn_selection
from .trace import Trace, walk
from .vocabulary import Vocabulary, from_stores, statute_text


@dataclass
class Writer:
    """Everything composition needs, built once and reported on."""

    vocab: Vocabulary = field(default_factory=Vocabulary)
    forms: Dict[str, Form] = field(default_factory=dict)
    built: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def build(
        cls,
        stores: Iterable[Any],
        prose: Sequence[Tuple[str, str]],
        *,
        statutes: Optional[Iterable[Path]] = None,
    ) -> "Writer":
        """``prose`` is (label, text). ``statutes`` are e-Gov XML paths.

        Statutes are read separately because their prose is inside a schema,
        and because leaving them out is the exact mistake that made a
        vocabulary blind to the register it was judging.
        """
        corpora = list(prose)
        if statutes:
            law = statute_text(statutes)
            if law:
                corpora.append(("statute", law))
        w = cls()
        w.vocab = from_stores(stores, corpora)
        sel = learn_selection(corpora)
        w.forms = harvest(corpora)
        w.built = {
            "corpora": {label: len(text) for label, text in corpora},
            "vocabulary": w.vocab.report(),
            "forms": len(w.forms),
            "selection": sel,
        }
        return w

    def sentence(
        self,
        store: Any,
        subject: str,
        *,
        limit: int = 1,
    ) -> List[Dict[str, Any]]:
        """Sentences about one subject, or nothing if it is not a word."""
        if subject not in self.vocab:
            return []
        labels = getattr(store, "source_labels", set()) or set()
        facets = [f for f in (store.crosses.get(subject) or {})
                  if f not in labels]
        return [d.as_dict() for d in
                compose(self.forms, subject, sorted(facets), limit=limit,
                        content_from=[subject], vocab=self.vocab)]

    def passage(
        self,
        store: Any,
        seed: str,
        *,
        steps: int = 6,
        mode: str = "path",
        trace: Optional[Trace] = None,
    ) -> Dict[str, Any]:
        """A walk, and a sentence at each step it could write one for.

        Anchoring on the path rather than the seed is the default because
        that is what a developing text needs: measured, path-anchoring walks
        furthest and overlaps its previous step most (0.069 against 0.033),
        while seed-anchoring holds the ORIGINAL subject longest (6.9 steps
        against 1.7) and is what a question wants.

        The trace comes back with the passage. It is the only mutable thing
        here, and returning it is what makes a passage resumable.
        """
        t = walk(store, seed, mode=mode, steps=steps, trace=trace)
        out: List[Dict[str, Any]] = []
        for core in t.seen:
            got = self.sentence(store, core, limit=1)
            if got:
                out.append(got[0])
        return {
            "seed": seed,
            "path": list(t.seen),
            "sentences": out,
            "written": len(out),
            "skipped": len(t.seen) - len(out),
            "trace": t,
            "note": "each sentence carries its content source and its form "
                    "source; neither makes it true",
        }

    def report(self) -> Dict[str, Any]:
        return dict(self.built)
