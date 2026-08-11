"""Surface conduction: flow between arms, and the word where it converges.

The inference core already flows — a question's energy enters at the rim and
converges along arms toward a centre. But the flow stops at each arm's four
faces, and two measured walls stand exactly there:

    * capacity: words not on the 24 faces route 0/60 — beyond the faces,
      a term might as well not exist
    * speech: 38% of path centres are retrieval keys, not words — 相続順位
      is where the answer lives and no sentence can start there

This module lets activation cross BETWEEN axes over the node's surface, as
the conception put it, instead of only radially along one arm:

    active points   the facets the answer lit up
    conduction      an active facet is a face on OTHER arms too — every
                    core whose cross holds it is one surface step away
    convergence     the WORD whose own cross holds the most active facets

The converged word is a new kind of centre: not the retrieval centre (the
key stays the citation), but the SPEAKING centre — the word the surface
flow chose, licensed by overlap the store actually holds.

## Determinism and abstention

Scoring is overlap count, descending, then lexicographic for reporting —
but a centre is only adopted on a STRICT lead. Two words tied at the top
mean the surface does not converge, and ties must abstain: a tie broken any
other way is the same manufactured agreement the staircase measurements
caught (unanimity 86 -> 321 probes, accuracy 73.3% -> 23.7%).

A minimum overlap of two is required. One shared facet is a point, not a
flow — the same argument that made `covenant.siblings` demand two shared
parents before calling anything a sibling.

## What this does not change

Retrieval, verdicts, and the citation. The key centre found the answer and
remains the answer; the surface centre only lends the sentence a subject
the corpus writes standalone. Closure holds: the centre is a held core, the
content facets are held facets, and the overlap that chose it is in the
store for anyone to recount.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

#: One shared facet is a point, not a flow.
MIN_OVERLAP = 2


def _inverted(store: Any, vocab: Any) -> Dict[str, List[str]]:
    """facet -> word-cores holding it. Built once per store, cached on it.

    Restricted to cores that are WORDS because only words can serve as
    speaking centres — inverting the full 87k-core federation would index
    an alphabet this module can never use.
    """
    cached = getattr(store, "_surface_inv", None)
    if cached is not None:
        return cached
    labels = getattr(store, "source_labels", set()) or set()
    inv: Dict[str, List[str]] = {}
    for c, cross in store.crosses.items():
        if c in labels or c not in vocab:
            continue
        for f in cross:
            inv.setdefault(f, []).append(c)
    store._surface_inv = inv
    return inv


def word_center(
    store: Any,
    core: str,
    facets: Sequence[str],
    vocab: Any,
) -> Optional[Dict[str, Any]]:
    """The word the surface flow converges on, or None when it does not.

    ``facets`` are the answer's active surface points. Every word-core
    whose cross holds an active facet is one conduction step across the
    surface; the one holding the most active facets — strictly more than
    any rival — is the convergence.
    """
    active = [f for f in facets if f]
    if not active:
        return None
    # ANCHORED to the key, after measuring what unanchored flow does. Free
    # conduction over the whole surface was tried twice: with evidence-sum
    # as the tiebreak it converged on GENERIC words (民法第百八十七条 spoke
    # through 女性, 諸教科 through 必要 — a huge cross overlaps everything);
    # with smallest-cross as the tiebreak it converged on FRAGMENTS
    # (不二雄 through 電荷, 都市再生機構 through 助言又). Both poles are the
    # same disease: a centre chosen with no relation to the KEY drifts to
    # whatever the scoring happens to reward, and no parameter sits stably
    # between them. So the flow is anchored: the speaking centre must be
    # part of the key's own name, or on the key's own cross.
    #
    # ① Words inside the key — the unit route, the same operation `reach`
    #   runs inward. 民法第七百十五条 speaks through 民法, 相続順位 through
    #   相続. LEFTMOST first, then longest at that position: longest-first
    #   was measured to pick the bare article number (船員法第三十六条 spoke
    #   through 第三十六条, five characters beating three), and a bare
    #   article number is the recorded collapse hazard — 第一条 exists in
    #   every law. Leftmost is the topic-first rule the subject gate
    #   already defends: the law's NAME is the topic, the number is the
    #   locator.
    units = [w for w in
             {core[i:j] for i in range(len(core))
              for j in range(i + 2, len(core) + 1)}
             if w != core and w in vocab and w in store.crosses]
    if units:
        units.sort(key=lambda w: (core.find(w), -len(w)))
        best = units[0]
        rivals = [w for w in units[1:] if len(w) == len(best)
                  and core.find(w) == core.find(best)]
        if not rivals:
            shared = [f for f in active
                      if f in (store.crosses.get(best) or ())] or active
            return {"word": best, "via": "unit", "overlap": len(shared),
                    "shared": shared,
                    "note": "the key's own name decomposed to a word the "
                            "corpus writes standalone; the key centre "
                            "remains the citation"}
    # ② Words on the key's own cross — one surface step, still anchored.
    #   Strict lead on overlap with the other active facets, or abstain.
    cands = [f for f in active if f in vocab and f in store.crosses]
    if not cands:
        return None
    scored = sorted(
        ((w, sum(1 for g in active if g != w
                 and g in (store.crosses.get(w) or ())) + 1)
         for w in cands),
        key=lambda t: (-t[1], t[0]))
    if scored[0][1] < MIN_OVERLAP:
        return None
    if len(scored) > 1 and scored[1][1] == scored[0][1]:
        return None
    word, n = scored[0]
    shared = [f for f in active if f in (store.crosses.get(word) or ())
              or f == word]
    return {"word": word, "overlap": n, "shared": shared,
            "note": "surface flow across arms converged on a word the "
                    "corpus writes standalone; the key centre remains the "
                    "citation"}
