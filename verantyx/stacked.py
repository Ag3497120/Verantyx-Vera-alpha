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
    # A sentence asserts a RELATION between its words — 「時効は光州事件と
    # 制定である」 claims a connection the way a path never does. When the
    # facet order carries no evidence (all counts tied, sequence is a
    # lexicographic accident), composing those facets into a copula is
    # upgrading an artifact into an assertion. The path is still shown; it
    # just is not spoken. Measured before this gate, the sampled sentence
    # stream mixed real relations (地形図は地形断面図ではない) with accidents
    # of the tie-break, and nothing marked which was which.
    if result.get("order_evidence") == "arbitrary":
        return {"verdict": result.get("verdict"), "sentences": [],
                "path": [w for w in text.split() if w],
                "note": "the facets are evidence-tied — an unordered set; "
                        "composing them into a sentence would assert "
                        "relations the corpus never ranked"}
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


def subject_check(store: Any, query: str, seed: str) -> Dict[str, Any]:
    """Does the staircase's seed actually cover what the question asked about?

    The staircase names a subject by coarsening, and coarsening has a failure
    mode that the verdict never showed: it keeps whatever part of the term it
    recognises and drops the rest without a word. Measured on 200 invented
    compounds, 77% came back answered, and in every answered case the seed
    was a substring — ヒュペリオン数人 answered about 数人, ズミルノフ環礁の面積
    about 面積. The reader is told about a different thing than the one they
    asked about, in the shape of a real answer.

    The subject is the LEFTMOST content run of the question. Japanese noun
    phrases are head-final inside a compound but topic-first across の:
    in 殺人罪の刑は the specific thing is 殺人罪 and 刑 is the aspect asked
    about it, so a seed that keeps the leftmost run and drops later ones is
    the measured-good case (subject alone, 113/120), while a seed that keeps
    a LATER run and drops the leftmost is the theft this exists to catch —
    相殺の効果は seeded on 効果 answered with the Meissner effect.

    The seed covers the subject when any of these holds:

      * the seed IS the subject, or contains it as a substring —
        相続の順位は seeded on 相続順位 stays a success
      * the subject is a facet of the seed's own cross — 背任罪とは seeded
        on 利得罪 stays, because crosses[利得罪] holds 背任罪: the arm the
        answer reads actually mentions the asked subject

    When none holds but the subject is itself a held core, the seed is
    REPLACED by the subject rather than refused — the staircase picked a
    worse entry than the question already contained.

    Otherwise the honest verdict is the one that already exists for a term
    the store does not hold: UNKNOWN_NOT_PRESENT, carrying the missing
    subject by name and the nearest held thing the staircase reached. That
    is a refusal with a pointer, not an answer about something else.
    """
    import re as _re

    from .lang import ja_content_runs

    # Japanese questions only. `ja_content_runs` returns Latin words too —
    # 「TypeScriptの型は」 must see TypeScript — so on a PURE-English query
    # it returns ['what', 'is', 'consideration'] and the leftmost-run rule
    # would make 'what' the subject of every English question. The gate is
    # a claim about Japanese topic structure; outside it, it stays silent.
    if not _re.search(r"[぀-ゟ゠-ヺ㐀-䶿一-鿿]", query):
        return {"ok": True, "subject": None}

    runs = ja_content_runs(query)
    if not runs:
        # Contentless — the gate has nothing to anchor on and an anchorless
        # gate would fire on noise. Inert by design.
        return {"ok": True, "subject": None}
    # The subject is the leftmost content PHRASE, not the leftmost run.
    # `ja_content_runs` splits on script boundaries, so ヴォルフガング粒子 is
    # two runs — and taking only the first made the subject equal the seed
    # by construction, which let every katakana-headed invented compound
    # straight through the gate it was built for. Adjacent runs (no gap in
    # the query between them) are one phrase; a particle like の ends it.
    subject = runs[0]
    consumed = 1
    pos = query.find(subject)
    if pos >= 0:
        end = pos + len(subject)
        for nxt in runs[1:]:
            if query.startswith(nxt, end):
                subject += nxt
                end += len(nxt)
                consumed += 1
            else:
                break
    #: The content runs AFTER the subject phrase are the asked ASPECT —
    #: 殺人罪の刑は asks the subject 殺人罪 about the aspect 刑. Entry stays
    #: subject-alone (adding terms to the seed was measured to dilute it,
    #: 113/120 -> 53), but the READ can honour them: see `aspect_read`.
    aspects = [r for r in runs[consumed:] if r and r not in subject]
    #: True when the whole question is one phrase plus grammar — no second
    #: content phrase after a particle. A caller gating the DIRECT path needs
    #: this: a multi-phrase query (「過失 故意」) legitimately lands on a core
    #: that is neither phrase, because intersection is the original
    #: conception working, and gating it would break puzzle inference.
    single = consumed == len(runs)
    if subject == seed or subject in seed:
        return {"ok": True, "subject": subject, "single": single, "aspects": aspects}
    cross = store.crosses.get(seed) or {}
    if subject in cross:
        return {"ok": True, "subject": subject, "single": single, "aspects": aspects,
                "via": "facet_of_seed"}
    if subject in store.crosses:
        return {"ok": True, "subject": subject, "single": single, "aspects": aspects,
                "reseed": subject}
    return {"ok": False, "subject": subject, "single": single, "aspects": aspects}


def aspect_read(store: Any, out: Dict[str, Any],
                aspects: Sequence[str]) -> Dict[str, Any]:
    """Re-read the answered core's faces through the asked aspect.

    Entry and read are different jobs and were conflated: seeding with the
    subject ALONE is measured right for entry (113/120 against 53 with terms
    added), but the read then showed the subject's generic top facets and
    ignored what was asked about it. 殺人罪の刑は entered on 殺人罪 —
    correctly — and showed 一定 人 例, while crosses[殺人罪] held 法定刑加重
    the whole time. Measured across ten core×aspect pairs whose aspect the
    store attests: the aspect reached the shown faces 1 of 10 times.

    So the aspect SELECTS at read time: facets of the answered core that
    contain an asked aspect run come first, ordered by count then
    lexicographically — the same tie rule as everywhere else. Facets the
    core does not hold are never invented (closure is untouched; this only
    reorders within one cross), and when no held facet matches, the read is
    left exactly as it was rather than pretending the aspect was addressed.

    The selection is declared: `order_evidence` becomes "aspect", because
    the sequence is now query-anchored — neither corpus-ranked nor
    arbitrary, and a reader weighing the answer needs to know which of the
    three it is.
    """
    core_key = out.get("core_key") or out.get("core")
    if not (core_key and out.get("text") and aspects):
        return out
    cross = store.crosses.get(str(core_key)) or {}
    matched = sorted(
        (f for f in cross if any(a in f for a in aspects)),
        key=lambda f: (-cross[f], f))
    if not matched:
        return out
    shown = [t for t in str(out["text"]).split() if t]
    head, tail = shown[:1], [t for t in shown[1:] if t not in matched]
    n_facets = max(len(shown) - 1, 1)
    out = dict(out)
    out["text"] = " ".join(head + (matched + tail)[:n_facets])
    out["aspect"] = list(aspects)
    out["aspect_facets"] = matched[:n_facets]
    out["order_evidence"] = "aspect"
    return out


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
    # The same subject gate, on the direct path — but ONLY for single-phrase
    # questions. After the staircase gate landed, the residual 4% of invented
    # compounds still answered were all entering here: ヒュペリオンesa retrieves
    # the core `esa` by Latin substring and never touches the staircase. A
    # multi-phrase query is exempt because intersection landing on a core
    # that is neither phrase (「過失 故意」 → 結果的加重犯) is the original
    # conception working, not a theft.
    if str(direct.get("verdict", "")).startswith("ANSWER"):
        seed0 = str(direct.get("core_key") or direct.get("core") or "")
        cov0 = subject_check(store, query, seed0)
        if cov0.get("ok") and cov0.get("aspects"):
            direct = aspect_read(store, direct, cov0["aspects"])
        if cov0.get("single") and not cov0["ok"]:
            return {
                "verdict": "UNKNOWN_NOT_PRESENT",
                "core": None,
                "text": "",
                "subject": cov0["subject"],
                "nearest_held": direct.get("core"),
                "note": ("retrieval reached only a part of the question's "
                         "subject; answering about the part would answer a "
                         "different question"),
            }
    if direct.get("verdict") != "UNKNOWN_NO_EVIDENCE" or judge is None:
        return direct

    g = judge.ask(query)
    if not str(g.get("verdict", "")).startswith("ANSWER"):
        # The staircase abstained — often CORRECTLY, because two held cores
        # tie: 相続の効果は holds both 相続 and 効果, the rungs cannot pick
        # one, and ties must abstain. But the tie is between subject and
        # ASPECT, and the question itself says which is which: the leftmost
        # phrase is the subject. If that subject is a held core, enter
        # there, exactly as the reseed branch does when the staircase named
        # the wrong thing. Measured: 相続の効果は, 時効の期間は, 契約の解除は
        # and 窃盗罪の刑は all went UNKNOWN_NO_EVIDENCE through this branch
        # while the store held every one of their subjects.
        cov = subject_check(store, query, "")
        if not cov.get("reseed"):
            # No held subject to enter on. The core\'s own refusal stands —
            # a second reader that also found nothing is not a reason to
            # widen further.
            return {**direct, "staircase": g.get("verdict")}
        g = {"item": cov["reseed"], "verdict": g.get("verdict"),
             "via": "subject_entry"}

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
    # The seed must cover the asked subject, or say that it cannot. Without
    # this check, 77% of invented compounds were answered about a substring.
    cov = subject_check(store, query, seed)
    if not cov["ok"]:
        return {
            "verdict": "UNKNOWN_NOT_PRESENT",
            "core": None,
            "text": "",
            "subject": cov["subject"],
            "nearest_held": seed,
            "staircase": g.get("verdict"),
            "note": ("the question's subject is not held; the staircase "
                     "reached only a part of it, and answering about the "
                     "part would answer a different question"),
        }
    if cov.get("reseed"):
        seed = cov["reseed"]
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
        if cov.get("aspects"):
            out = aspect_read(store, out, cov["aspects"])
        return out
    return {**out, "seeded_from": {"subject": seed, "query": seeded_query}}
