"""The staircase feeds the inference core. Layered, never pooled.

`consensus` is the original conception and it works: sections enter at the
rim, an edge query changes each node\'s energy, moves rearrange the cross,
and agreement at a stable state ends the search — then the axis words along
the agreed paths are concatenated into an answer with no language model
anywhere. Measured on the 626MB federation, 「過失 故意」 comes back ANSWER
with 「過失 法学 結果的加重犯 引 故意」.

What fails is getting there. `candidates_for_query` returned nothing at all
for 「殺人罪の刑は」, 「傷害罪とは」 and 「相続の順位は」 — three total
failures where the store holds the subject and the entry could not name it.
The staircase can: it reaches 傷害罪 and 相続順位 by coarsening, which is
what it was measured to do.

Seeding the core with what the staircase found turned two of those three
into ANSWER, generated paths included:

    傷害罪とは    -> 傷害罪 傷害 傷害致死罪 死 法学
    相続の順位は   -> 相続順位 法学 相続

## Layered, and that word is doing work

Every combination measured this session divides cleanly:

    POOLED — two signals into one vote, index or store:
      cut-varied sovereigns beside data-varied   out-of-corpus 0 -> 8 wrong
      two languages in one store                 false answers in both
      eleven grain settings instead of six       reach 464 -> 450, false 2 -> 7
      three domain sovereigns instead of one     answered 284 -> 208
      citations merged into the core ladder      0 of 387 gold links
      units and links added to a core\'s terms    385 -> 351 answers

    LAYERED — one stage\'s typed output is the next stage\'s input:
      vocabulary before composition              73% -> 100% attested words
      licence before composition                 49 -> 0 unlicensed norms
      seam test at fill time                     18% -> 0% broken joins
      coverage beside the verdict                bad answers became legible
      staircase before the inference core        3 dead questions -> 2 answers

Six pooled combinations, all worse. Five layered ones, all better. The rule
is not about which parts are good — every part above was measured good on
its own — it is that pooling asks two structures that mean different things
by "agreement" to vote in one election, and layering asks one to hand the
other something it can use.

## What this does not do

Promote. A seeded answer says it was seeded, because the entry was widened
by coarsening and that is exactly the fact a reader needs in order to
discount it. `SEEDED` is not `ANSWER`.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence


def in_words(
    store: Any,
    result: Dict[str, Any],
    writer: Any,
    *,
    limit: int = 2,
) -> Dict[str, Any]:
    """Put the answer into sentences, using the PATH as the content.

    The inference core already generates: on agreement it concatenates the
    axis words along the converged section paths, natural language
    rearranged with no model anywhere. 「過失 故意」 comes back as
    「過失 法学 結果的加重犯 引 故意」 — the answer, in the query\'s own
    terms, and not a sentence.

    `writer` composes sentences and, on its own, ignores the question: given
    the seed 過失 it walked and produced 「法律ではほとんどストーカーを規定
    している」 as its second sentence. The walk is what drifted, not the
    composition.

    So the path replaces the walk. The centre becomes the subject, the rest
    of the path becomes the available content, and the writer supplies only
    the FORM:

        過失 故意     -> 過失は故意となっている。
        正当防衛とは    -> 正当防衛は行為の成立である。
        遺言 方式     -> 遺言は法律をもつてこれをしなければならない。

    Layered, not pooled: the core decides what the answer is about and the
    writer decides only how to say it. Every draft still carries its content
    source and its form source separately, and a sentence built this way is
    still a draft — it is not the citation, which is the path.
    """
    text = result.get("text")
    if not text:
        return {"verdict": result.get("verdict"), "sentences": [],
                "note": "no converged path to speak from"}
    path = [w for w in text.split() if w]
    if not path:
        return {"verdict": result.get("verdict"), "sentences": []}
    subject, rest = path[0], path[1:]
    if subject not in writer.vocab:
        return {"verdict": "UNKNOWN_SUBJECT_NOT_A_WORD", "subject": subject,
                "path": path,
                "note": "the centre is a retrieval key and not a word the "
                        "corpus writes on its own; the path stands as the "
                        "answer"}
    from .compose_ja import compose

    drafts = compose(writer.forms, subject, rest, limit=limit,
                     content_from=[subject], vocab=writer.vocab,
                     licence=writer.licence(subject))
    return {
        "verdict": result.get("verdict"),
        "path": path,
        "sentences": [d.as_dict() for d in drafts],
        "note": "content from the converged path, form from a harvested "
                "template; neither makes it true",
    }


def ask(
    store: Any,
    query: str,
    *,
    judge: Optional[Any] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    """Ask the inference core; if it cannot enter, let the staircase in.

    ``judge`` is a built `graded.GradedJudge` over the same store. Building
    one per call would re-index 54,244 cores for a question.
    """
    from .consensus_store import candidates_for_query, consensus_over_store

    direct = consensus_over_store(store, query, **kwargs)
    if direct.get("verdict") != "UNKNOWN_NO_EVIDENCE" or judge is None:
        return direct

    g = judge.ask(query)
    if not str(g.get("verdict", "")).startswith("ANSWER"):
        # The staircase could not name a subject either. The core\'s own
        # refusal stands — a second reader that also found nothing is not a
        # reason to widen further.
        return {**direct, "staircase": g.get("verdict")}

    # The subject alone. Adding its facets to the seeded query DILUTES it:
    # measured over 120 questions the core could not enter, where the
    # staircase did name a subject —
    #
    #     subject alone        113 of 120 answered   94%
    #     + 4 by frequency      53                   44%
    #     + 4 alphabetically    57                   48%
    #     + every facet         35                   29%
    #     + 4 rarest            24                   20%
    #
    # Extra terms pull the sections apart, which is the same reason a
    # one-word question makes every rung abstain: the core is looking for a
    # centre several sections agree on, and each added term is another
    # section that has to agree. It also removes an arbitrary choice — there
    # is no ordering left to pick, so nothing here decides the answer by
    # how a list happened to be sorted.
    seed = g["item"]
    seeded_query = seed
    out = consensus_over_store(store, seeded_query, **kwargs)
    if out.get("verdict") == "ANSWER":
        out = dict(out)
        # Typed apart on purpose. The entry was widened; a reader deciding
        # whether to rely on this needs to see that it was.
        out["verdict"] = "SEEDED"
        out["seeded_from"] = {"subject": seed,
                              "staircase_verdict": g.get("verdict"),
                              "agreeing": g.get("agreeing"),
                              "query": seeded_query}
        out["note"] = ("the inference core could not enter on the question as "
                       "asked; the staircase named a subject by coarsening "
                       "and the core was re-entered there")
        return out
    return {**out, "seeded_from": {"subject": seed, "query": seeded_query}}
