"""Forks for the three unbuilt pieces of the original conception:
geometric visibility (E1), memory stacking, and placement granularity (E3).

Same contract as every other *_forks module: each fork states its expected
outcome as a pass criterion and returns evidence. Two disciplines carried
over from the rest of this work:

  - E1 does NOT declare a winner between ring and geometric visibility.
    The conception's "attack a nuance from both poles" reading can justify
    either topology; which one reasons better is the designer's question
    and needs real queries. What a fork CAN pin down is the structural
    difference itself — that the two modes disagree about exactly one
    thing, opposite-pole visibility — so the later A/B measures topology
    and nothing else.

  - The drift fork constructs the telephone-game failure on purpose and
    asserts it surfaces as UNKNOWN_DRIFT, not as a confident wrong answer.
    An extended-inference engine whose failure mode is silent topic drift
    would be worse than no engine.
"""
from __future__ import annotations

from collections import Counter
import tempfile
from pathlib import Path
from typing import Any, Dict, List

from .consensus import (AXES, N_SECTIONS, ConsensusConfig, SearchState,
                        visible_axes)
from .cross import ShellCross
from .cross_store import CrossStore
from .layer_stack import LayeredMemory, layered_ask
from .consensus_store import MAX_ARMS as MAX_ARMS_FORK
from .face_roles import FACET_FACES as _FF

FACES_FORK = len(_FF)


def _opposite(axis: str) -> str:
    return ("-" if axis[0] == "+" else "+") + axis[1]


# ---------------------------------------------------------------------------
# E1 — the structural property, stated exactly.
# ---------------------------------------------------------------------------

def geometric_pole_invisibility_fork() -> Dict[str, Any]:
    """Ring windows contain the opposite pole; geometric ones never do.

    In the current AXES ordering the poles of each axis are adjacent in the
    tuple, so at window=1 EVERY ring section sees its own opposite — worth
    stating as a measured fact, because it is the strongest form of the
    both-poles-visible reading. Geometric mode must make the opposite
    invisible from every section while keeping the visible-arc size equal,
    and widening must erase the difference (escape shows everything in both
    modes).
    """
    ring = ConsensusConfig()
    geo = ConsensusConfig(geometric_visibility=True)
    st = SearchState(shell=ShellCross())

    ring_sees_pole = []
    geo_sees_pole = []
    sizes_equal = True
    for s in range(N_SECTIONS):
        home = AXES[s % N_SECTIONS]
        r = visible_axes(st, s, ring)
        g = visible_axes(st, s, geo)
        ring_sees_pole.append(_opposite(home) in r)
        geo_sees_pole.append(_opposite(home) in g)
        if len(r) != len(g):
            sizes_equal = False

    widened = SearchState(shell=ShellCross(), widened=True)
    widened_equal = all(
        visible_axes(widened, s, ring) == visible_axes(widened, s, geo) == list(AXES)
        for s in range(N_SECTIONS)
    )

    ok = (all(ring_sees_pole) and not any(geo_sees_pole)
          and sizes_equal and widened_equal)
    return {
        "experiment": "cross_geometry",
        "fork": "GEOMETRIC_POLE_INVISIBILITY",
        "pass": bool(ok),
        "result": {
            "ring_sections_seeing_own_pole": sum(ring_sees_pole),
            "geo_sections_seeing_own_pole": sum(geo_sees_pole),
            "visible_arc_sizes_equal": sizes_equal,
            "widened_erases_difference": widened_equal,
        },
    }


def geometric_rotation_reaches_all_fork() -> Dict[str, Any]:
    """Geometric visibility must not orphan any axis: across all rotations,
    every axis becomes visible to some section. The pole is unreachable from
    where you STAND, not unreachable outright — rotation is exactly the move
    that faces it, and if some axis were invisible at every rotation the
    mode would have quietly deleted part of the cross."""
    geo = ConsensusConfig(geometric_visibility=True)
    seen: set = set()
    for rot in range(N_SECTIONS):
        st = SearchState(shell=ShellCross(), rotation=rot)
        for s in range(N_SECTIONS):
            seen.update(visible_axes(st, s, geo))
    ok = seen == set(AXES)
    return {
        "experiment": "cross_geometry",
        "fork": "GEOMETRIC_ROTATION_REACHES_ALL",
        "pass": bool(ok),
        "result": {"axes_reachable": sorted(seen)},
    }


# ---------------------------------------------------------------------------
# Memory stacking — the conception's second matryoshka.
# ---------------------------------------------------------------------------

_SENTENCES = [
    "apple is red fruit",
    "banana is yellow fruit",
    "cherry is small fruit",
    "durian is spiky fruit",
    "elderberry is dark fruit",
]


def layer_stack_growth_fork() -> Dict[str, Any]:
    """Capacity 2, five distinct cores: layers must stack (2,2,1), lower
    layers must FREEZE — later ingests never touch them — and a save/load
    round trip must preserve the whole arrangement. Freezing is the point:
    a frozen layer's internal arrangement is what stays stable while the
    stack above it grows."""
    import tempfile
    from pathlib import Path

    mem = LayeredMemory(capacity=2)
    for s in _SENTENCES:
        mem.ingest_sentence(s)

    frozen_sizes = [lvl.n_cores() for lvl in mem.levels]
    mem.ingest_sentence("fig is soft fruit")  # goes to top, must not touch L0/L1
    after_sizes = [lvl.n_cores() for lvl in mem.levels]
    lower_frozen = frozen_sizes[:-1] == after_sizes[:len(frozen_sizes) - 1]

    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "mem.json"
        mem.save(p)
        back = LayeredMemory.load(p)
        roundtrip = (back.n_levels() == mem.n_levels()
                     and back.total_cores() == mem.total_cores())

    ok = (mem.n_levels() >= 3 and mem.total_cores() == 6
          and lower_frozen and roundtrip)
    return {
        "experiment": "cross_geometry",
        "fork": "LAYER_STACK_GROWTH",
        "pass": bool(ok),
        "result": {"levels_after_five": frozen_sizes,
                   "levels_after_six": after_sizes,
                   "lower_layers_frozen": lower_frozen,
                   "roundtrip": roundtrip},
    }


def layered_ask_stability_fork() -> Dict[str, Any]:
    """Two layers that both know the answer must converge and halt early
    with the stability verdict — two consecutive layers naming the same
    core is the conception's stop condition applied across layers."""
    l0 = CrossStore()
    l1 = CrossStore()
    for s in _SENTENCES[:3]:
        l0.ingest_sentence(s)
        l1.ingest_sentence(s)
    mem = LayeredMemory(levels=[l0, l1])

    out = layered_ask(mem, "what is apple", carry="A")
    ok = (out["verdict"] == "ANSWER" and out.get("core") == "apple"
          and out.get("stable_at_level") == 1)
    return {
        "experiment": "cross_geometry",
        "fork": "LAYERED_ASK_STABILITY",
        "pass": bool(ok),
        "result": {"verdict": out["verdict"], "core": out.get("core"),
                   "stable_at_level": out.get("stable_at_level")},
    }


def layered_carry_drift_fork() -> Dict[str, Any]:
    """The telephone game, constructed on purpose.

    L0 knows apple. L1 does NOT contain apple at all — only a core reachable
    from apple's own answer words. Without the original query riding along
    (carry B), layer 1 can only follow the answer tokens and lands on the
    other core: two layers, two different answers, no consecutive agreement.
    That must surface as UNKNOWN_DRIFT — a type, not a confident wrong
    answer, because a silent topic drift is this engine's worst failure.

    Carry A on the same memory must NOT invent an agreement either: L1
    genuinely does not know apple, so no stability exists to find. What A
    must do is keep the original question in every layer's query — the
    anchor whose absence is what B is testing.
    """
    l0 = CrossStore()
    l0.ingest_sentence("apple is red fruit")
    l1 = CrossStore()
    l1.ingest_sentence("red is bright color")
    mem = LayeredMemory(levels=[l0, l1])

    b = layered_ask(mem, "what is apple", carry="B")
    a = layered_ask(mem, "what is apple", carry="A")

    b_drifted = b["verdict"] == "UNKNOWN_DRIFT" and (b.get("cores_seen") or [])[:1] == ["apple"]
    a_keeps_anchor = all("apple" in t.get("query", "")
                         for t in a.get("trace", []) if not t.get("skipped"))
    b_drops_anchor = any(t.get("level") == 1 and "what" not in t.get("query", "")
                         for t in b.get("trace", []))

    ok = b_drifted and a_keeps_anchor and b_drops_anchor
    return {
        "experiment": "cross_geometry",
        "fork": "LAYERED_CARRY_DRIFT",
        "pass": bool(ok),
        "result": {
            "carry_B": {"verdict": b["verdict"], "cores_seen": b.get("cores_seen")},
            "carry_A": {"verdict": a["verdict"],
                        "anchor_in_every_query": a_keeps_anchor},
        },
    }


# ---------------------------------------------------------------------------
# E3 — placement granularity changes what the structure can do.
# ---------------------------------------------------------------------------

def placement_granularity_fork() -> Dict[str, Any]:
    """The same information, poured two ways, is not the same knowledge.

    Fine: three sentences ingested separately — each contributes its own
    facets to the core. Coarse: the same words as one run-on sentence.
    The conception claims node design and granularity change precision;
    the fork pins the mechanism: fine placement must yield at least as
    many facet links for the core, and the consensus answer text over the
    fine store must contain a facet the coarse store's answer lacks. This
    is the measurable seed of the placement-simulation harness — placement
    policies become data, and this is the first measured pair.
    """
    from .consensus_store import consensus_over_store

    fine = CrossStore()
    for s in ["apple is red", "apple is sweet", "apple grows on trees"]:
        fine.ingest_sentence(s)
    coarse = CrossStore()
    coarse.ingest_sentence("apple is red sweet grows on trees")

    f_links = fine.n_facet_links()
    c_links = coarse.n_facet_links()
    f_out = consensus_over_store(fine, "what is apple")
    c_out = consensus_over_store(coarse, "what is apple")
    f_words = set(str(f_out.get("text", "")).split())
    c_words = set(str(c_out.get("text", "")).split())

    # Parenthesised deliberately: an earlier version wrote `... and X or Y`,
    # which passes on Y alone — a fork that can pass without its own main
    # criterion is the false-pass shape this whole codebase keeps hunting.
    differ = bool(f_words - c_words) or bool(c_words - f_words)
    ok = (f_links >= c_links
          and f_out.get("verdict") == "ANSWER"
          and differ)
    return {
        "experiment": "cross_geometry",
        "fork": "PLACEMENT_GRANULARITY",
        "pass": bool(ok),
        "result": {
            "fine_facet_links": f_links, "coarse_facet_links": c_links,
            "fine": {"verdict": f_out.get("verdict"), "text": f_out.get("text")},
            "coarse": {"verdict": c_out.get("verdict"), "text": c_out.get("text")},
        },
    }


def placement_simulation_fork() -> Dict[str, Any]:
    """Pre-simulation must beat frequency when demand is concentrated,
    and must be REFUSED when it is flat.

    Both halves are the fork. A placement policy that always looks like an
    improvement is one whose gate does not work, and this project's whole
    claim rests on gates that can say no — so the flat-demand case is
    checked as hard as the concentrated one.

    The mechanism: an arm has four facet faces, a contested core has more
    than four facts, and so shipping is a choice of which four. Frequency
    picks the four the corpus repeats. Simulation picks the four the
    anticipated questions ask for. When questions are uniform over facts,
    every policy covers the asked fact with probability 4/N and simulation
    has nothing to learn — which is what the flat half pins down.
    """
    from .placement import accept, compare, derive_split

    store = CrossStore()
    # Twelve facts per core, well over the four faces, so placement is a
    # real choice rather than an identity. The words must be alphabetic:
    # a first version used fact0..fact11, the query synthesiser skips
    # non-alphabetic facets, no queries were generated, and the fork passed
    # on 0.0 <= 0.0. A fork that passes on an empty measurement is worse
    # than no fork, so the emptiness is now checked explicitly below.
    words = ("redness", "sweetness", "crispness", "firmness", "roundness",
             "brightness", "weight", "colour", "texture", "flavour",
             "season", "origin")
    # Forty cores, not three: with three the whole measurement is three
    # questions and a coincidence reads as a result. This is small enough to
    # stay a unit test and large enough that a 12-point gap is not noise.
    # Purely alphabetic names: the query synthesiser also skips non-alpha
    # CORES, so topica0 produced zero test queries the same way fact0 did.
    cores = [f"topic{chr(97 + i // 10)}{chr(97 + i % 10)}" for i in range(40)]
    for core in cores:
        for w in words:
            store.ingest_sentence(f"{core} is {w}")

    out: Dict[str, Any] = {}
    for demand in ("zipf", "uniform"):
        train, test = derive_split(store, len(cores), demand=demand, reps=8)
        cmp_ = compare(store, test, train=train)
        out[demand] = {
            "verdict": accept(cmp_)["verdict"],
            "n_test": cmp_["n_queries"],
            "uncovered_frequency":
                cmp_["summary"]["frequency"]["mean_uncovered_terms"],
            "uncovered_simulated":
                cmp_["summary"]["simulated"]["mean_uncovered_terms"],
            "answer_rate_delta": cmp_["delta"]["answer_rate"],
        }

    measured = all(
        out[d]["n_test"] > 0 and out[d]["uncovered_frequency"] > 0
        for d in ("zipf", "uniform")
    )
    ok = (measured
          and out["zipf"]["uncovered_simulated"] < out["zipf"]["uncovered_frequency"]
          and out["zipf"]["answer_rate_delta"] >= 0
          and out["uniform"]["uncovered_simulated"]
              >= out["uniform"]["uncovered_frequency"])
    return {
        "experiment": "cross_geometry",
        "fork": "PLACEMENT_SIMULATION",
        "pass": bool(ok),
        "result": dict(out, measured=measured),
    }


def placement_cannot_manufacture_confidence_fork() -> Dict[str, Any]:
    """A question that cannot distinguish its candidates must stay AMBIGUOUS,
    whatever the placement policy.

    Six clinics, each carrying dosage and onset among twelve facts. "what has
    dosage onset" matches all six equally — there is no answer, and the
    frequency rule says so. The first demand model credited a question's
    whole demand to its TOP retrieved core, so those two facts were placed on
    clinica alone, its arm out-scored five identical rivals on query overlap,
    and the engine returned a confident ANSWER naming one clinic.

    That is the most dangerous thing this module can do: not a worse answer,
    but a correct refusal converted into a wrong answer. The gate in
    `accept` cannot catch it — the answer rate goes UP, which every other
    check reads as an improvement. So it is pinned here as a shape instead.

    The fix is that all retrieved cores are credited, which is also just
    true: when a descriptive question matches six things, all six are what
    the asker wants to hear about.
    """
    from .placement import (choose_for_core, demand_from_queries,
                            facet_document_frequency)
    from .consensus import run_consensus
    from .consensus_store import _MassView, candidates_for_query
    from .cross import AXES, ShellCross
    from .face_roles import FACET_FACES

    store = CrossStore()
    shared = ["dosage", "contraindication", "onset", "interval",
              "monitoring", "titration"]
    for i in range(6):
        core = f"clinic{chr(97 + i)}"
        for a in shared:
            store.ingest_sentence(f"{core} has {a}")
        for j in range(6):
            store.ingest_sentence(f"{core} has trait{chr(97 + i)}{chr(97 + j)}")

    q = "what has dosage onset"
    cores = candidates_for_query(store, q, k=6)
    df = facet_document_frequency(store)
    asked = demand_from_queries(store, [q])
    masses = _MassView(store)

    out: Dict[str, Any] = {"n_candidates": len(cores)}
    for policy in ("frequency", "simulated"):
        shell = ShellCross()
        for axis, core in zip(AXES, cores):
            shell.faces[axis]["tip"] = core
            shell.reflections[axis] = core
            picks = choose_for_core(
                store, core, policy=policy, df=df, n_cores=store.n_cores(),
                asked=asked if policy == "simulated" else None, weight=0.0)
            for face, facet in zip(FACET_FACES, picks):
                shell.faces[axis][face] = facet
        res = run_consensus(shell, q, masses=masses)
        out[policy] = {"verdict": res.verdict, "core": res.core}

    # Six identical candidates is the precondition; without it the rest is
    # vacuous and the fork would pass on an empty setup.
    ok = (len(cores) == 6
          and out["frequency"]["verdict"] != "ANSWER"
          and out["simulated"]["verdict"] != "ANSWER")
    return {
        "experiment": "cross_geometry",
        "fork": "PLACEMENT_CANNOT_MANUFACTURE_CONFIDENCE",
        "pass": bool(ok),
        "result": out,
    }


def ja_coverage_gate_fork() -> Dict[str, Any]:
    """The Japanese path must refuse a question it did not address.

    `consensus_over_store` ran three gates; `ja_consensus_ask` ran none. So
    on the language this engine was built for, a two-party question came
    back as a confident answer about one party:

        「甲は乙を脅迫した。乙は丙を傷害した。」
        ask 「甲 丙」 -> ANSWER 「甲は主犯、乙、脅迫」

    Every word of that is true and it is still a fabrication, because 丙 was
    asked about and silently dropped. Three cases are pinned: the covered
    question still answers, the uncovered one refuses, and an entirely
    unknown term is REPORTED rather than refused — that one is a vocabulary
    gap, and conflating it with a reading failure would send it to the wrong
    queue.
    """
    from .consensus_store import ja_consensus_ask
    from .document_ingest import Document, ingest_documents

    store = CrossStore()
    ingest_documents(store, [Document(
        source="事件記録",
        text="甲は乙を脅迫した。乙は丙を傷害した。丙は負傷した。甲は主犯である。")])

    covered = ja_consensus_ask(store, "甲 乙")
    dropped = ja_consensus_ask(store, "甲 丙")
    unknown = ja_consensus_ask(store, "甲 未知語")

    ok = (covered.get("verdict") == "ANSWER"
          and dropped.get("verdict") != "ANSWER"
          and "丙" in (dropped.get("uncovered_terms") or [])
          and unknown.get("verdict") == "ANSWER"
          and "未知語" in (unknown.get("uncovered_terms") or []))
    return {
        "experiment": "cross_geometry",
        "fork": "JA_COVERAGE_GATE",
        "pass": bool(ok),
        "result": {
            "covered": covered.get("verdict"),
            "dropped_party": {"verdict": dropped.get("verdict"),
                              "uncovered": dropped.get("uncovered_terms")},
            "unknown_term": {"verdict": unknown.get("verdict"),
                             "uncovered": unknown.get("uncovered_terms")},
        },
    }


def placement_invariance_fork() -> Dict[str, Any]:
    """An answer that depends on an arbitrary tie-break must not stand.

    Placement cannot add information: the store holds the same facts either
    way, and only which four reach the faces changes. So a core that wins
    under one arbitrary ordering of equally-scored facts and loses under
    another won on the ordering. Same argument shape as "layout cannot add
    information" in docs/METAMORPHIC.md, and the same property — no answer
    key, no human, no model, both readings in this process.

    Two halves, and the second is what stops this being a gate that simply
    refuses everything: an answer grounded in counts the store can actually
    tell apart must SURVIVE, because ties are not what carried it.

    The fixture needs RIVALS. A single-arm store was the first attempt and
    the gate could never fire on it: with one candidate the core wins
    whatever occupies the faces, so the two readings always agree on the
    core and only the text differs — which is placement doing its ordinary
    job, not an artifact. Fabrication needs somebody to beat.
    """
    import random

    from .consensus_store import candidates_for_query, consensus_over_store

    # Six rivals, every facet count 1, each carrying eight of twelve aspects.
    # The whole ordering is therefore a tie-break, and a descriptive question
    # that several of them satisfy can only be "won" by the tie-break.
    aspects = [f"asp{chr(97 + i)}" for i in range(12)]
    rng = random.Random(5)
    tied = CrossStore()
    for i in range(6):
        core = f"unit{chr(97 + i)}"
        for a in rng.sample(aspects, 8):
            tied.ingest_sentence(f"{core} has {a}")
    # This exact question is a fabrication under the default tie-break:
    # FOUR of the six units carry both aspd and aspf, and the engine names
    # one of them. The owner count is asserted below so that a fixture drift
    # which made the question legitimately decidable would fail here rather
    # than quietly turn this into a test of nothing.
    q = "what has aspd aspf"
    owners = [c for c in candidates_for_query(tied, q, k=6)
              if "aspd" in tied.crosses[c] and "aspf" in tied.crosses[c]]
    tied_off = consensus_over_store(tied, q)
    tied_on = consensus_over_store(tied, q, placement_invariant=True)

    # Counts differ -> the top four are not tied -> the gate must not fire.
    weighted = CrossStore()
    for _ in range(5):
        weighted.ingest_sentence("gadget is heavy")
    for _ in range(4):
        weighted.ingest_sentence("gadget is fast")
    for _ in range(3):
        weighted.ingest_sentence("gadget is small")
    for _ in range(2):
        weighted.ingest_sentence("gadget is quiet")
    weighted.ingest_sentence("gadget is rare")
    w_on = consensus_over_store(weighted, "what is gadget heavy",
                                placement_invariant=True)

    ok = (len(owners) > 1                       # the question IS undecidable
          and tied_off.get("verdict") == "ANSWER"   # and the engine answered it
          and tied_on.get("verdict") != "ANSWER"    # and the gate refuses it
          and tied_on.get("reason") == "placement_dependent_answer"
          and w_on.get("verdict") == "ANSWER"       # without refusing everything
          and w_on.get("placement_invariant") is True)
    return {
        "experiment": "cross_geometry",
        "fork": "PLACEMENT_INVARIANCE",
        "pass": bool(ok),
        "result": {
            "owners_of_the_question": owners,
            "all_ties_gate_off": tied_off.get("verdict"),
            "all_ties_gate_on": {"verdict": tied_on.get("verdict"),
                                 "reason": tied_on.get("reason")},
            "distinct_counts_gate_on": {"verdict": w_on.get("verdict"),
                                        "invariant": w_on.get("placement_invariant")},
        },
    }


def reified_event_fork() -> Dict[str, Any]:
    """A three-place fact fits a two-place store when the EVENT is the core.

    「甲は乙を脅迫した。乙は丙を傷害した。」 as ordinary sentences gives two
    crosses with no path between them, so a question about 甲 and 丙 reaches
    nothing. Reified, the participants become role-tagged facets of the
    happening, every fact stays unary attribution — the shape the store
    already has — and the chain is walkable.

    Four things are pinned, and the last three are the ones that make it
    usable rather than merely representable:

      * the chain 丙 → 事象2 → 乙 → 事象1 → 甲 is traversable
      * a role claim has THREE outcomes: confirmed, refuted, unknown. Asked
        with the wrong object the store must REFUTE (it holds 対象丙), and
        asked about a role it never recorded it must say so — answering the
        whole event and staying silent about the asked role is the same
        fabrication one level down
      * all four faces carry facts: the citation is provenance, and it was
        occupying one of the four an event needs
      * an unreadable construction is SKIPPED WITH ITS REASON, not guessed.
        「暴行を加えた」 would require deciding that 暴行 rather than 加 is
        the act, and a wrong guess there mislabels who did what.
    """
    from .consensus_store import ja_consensus_ask
    from .events import extract_events, ingest_events, link_cause

    ex = extract_events("甲は乙を脅迫した。乙は丙を傷害した。甲が乙に暴行を加えた。")
    events = ex["events"]
    if len(events) < 2:
        return {"experiment": "cross_geometry", "fork": "REIFIED_EVENT",
                "pass": False, "result": {"extracted": len(events),
                                          "skipped": ex["skipped"]}}
    link_cause(events[1], events[0])
    store = CrossStore()
    ingest_events(store, events, source="事件記録")

    def facts(core):
        return {f for f in (store.crosses.get(core) or {})
                if f not in store.source_labels}

    hit = [c for c, f in store.crosses.items() if "対象丙" in f]
    walked = False
    if hit:
        e2 = facts(hit[0])
        cause = [f[2:] for f in e2 if f.startswith("原因")]
        if cause:
            e1 = facts(cause[0])
            walked = ("主体甲" in e1 and "対象乙" in e1 and "主体乙" in e2)

    confirmed = ja_consensus_ask(store, "事象2 対象丙")
    refuted = ja_consensus_ask(store, "事象2 対象甲")
    unknown_role = ja_consensus_ask(store, "事象2 場所東京")
    faces = [f for f in (store.crosses.get("事象2") or {})
             if f not in store.source_labels]
    light = [s for s in ex["skipped"] if s["reason"].startswith("light_verb")]

    ok = (walked
          and confirmed.get("verdict") == "ANSWER"
          and refuted.get("verdict") == "UNKNOWN_ROLE_MISMATCH"
          and unknown_role.get("verdict") == "UNKNOWN_INSUFFICIENT_EVIDENCE"
          and len(faces) >= 4
          and len(light) == 1)
    return {
        "experiment": "cross_geometry",
        "fork": "REIFIED_EVENT",
        "pass": bool(ok),
        "result": {
            "chain_walked": walked,
            "confirmed": confirmed.get("verdict"),
            "refuted": {"verdict": refuted.get("verdict"),
                        "detail": refuted.get("reason")},
            "role_not_recorded": unknown_role.get("verdict"),
            "facts_on_faces": sorted(faces),
            "skipped_with_reason": ex["skipped"],
        },
    }


def event_extractor_refuses_statute_prose_fork() -> Dict[str, Any]:
    """The event reader must read a fact pattern and REFUSE statute prose.

    Unguarded, this extractor produced 6,800 "events" from 9,087 sentences
    of six Japanese statutes — 74.8%, and inspection showed 又 (a
    conjunction) assigned as the actor and 処 as the act. The rate measured
    how often it produced something, not how often it produced something
    true. With the guard: 2.1% read, everything else refused by name.

    That is not a shortcoming to fix later. Statutes state RULES; a fact
    pattern is not in them. This fork pins the division so a future
    loosening of the guard has to argue with a measurement.
    """
    from .events import extract_events, unreadable

    facts = "甲は乙を脅迫した。乙は丙を傷害した。甲は乙に現金を交付した。"
    got = extract_events(facts)

    # Real sentences from 刑法 and 民法, each carrying a construction the
    # particle reader cannot resolve.
    statute = [
        "前項の規定は、親族でない共犯については、適用しない。",
        "公務員が職権を濫用して、人に義務のないことを行わせ、又は権利の行使を妨害したとき。",
        "権利の行使及び義務の履行は、信義に従い誠実に行わなければならない。",
    ]
    refused = [unreadable(s) for s in statute]

    ok = (len(got["events"]) == 3
          and got["events"][0].roles.get("主体") == "甲"
          and got["events"][0].roles.get("対象") == "乙"
          and got["events"][1].roles.get("主体") == "乙"
          and not got["skipped"]
          and all(r is not None for r in refused))
    return {
        "experiment": "cross_geometry",
        "fork": "EVENT_EXTRACTOR_REFUSES_STATUTE_PROSE",
        "pass": bool(ok),
        "result": {
            "fact_pattern_roles": [e.roles for e in got["events"]],
            "fact_pattern_skipped": got["skipped"],
            "statute_refusals": refused,
        },
    }


def egov_article_is_a_citation_key_fork() -> Dict[str, Any]:
    """An article must be retrievable by the string a reader types.

    Two shapes were measured and rejected on real 刑法 XML. Whole law as one
    document: headings carry forward and 第百八条 came back with captions
    belonging to other articles — a citation tool pointing at the wrong
    provision. One document per article: attribution is right, but the core
    of 「人を殺した者は…」 is 者, which several hundred articles share.

    Reifying the article fixes both, and the last assertion is why it is
    worth having: asked whether 第二百四条 is about 殺人, the engine must
    REFUSE. 204 is 傷害. Returning the article with a confident silence
    about the word actually asked is the failure this whole layer exists
    to prevent.

    The fixture carries 第百九十九条（殺人）as well, and it has to: the gate
    refuses a term the store KNOWS and merely reports one it has never seen,
    because "204 is not about murder" and "I do not know what murder is" are
    different states. Without 殺人 present the fork passed on the wrong
    branch — the real 刑法 refuses because 殺人 is in it.
    """
    import xml.etree.ElementTree as ET
    import tempfile
    from pathlib import Path

    from .consensus_store import ja_consensus_ask
    from .egov import ingest_law

    # A miniature law with both a main and a supplementary 第一条 — the
    # collision that made 民法第一条 return 施行期日 instead of 基本原則.
    xml = """<?xml version="1.0" encoding="UTF-8"?>
<Law><LawBody><LawTitle>試験法</LawTitle>
<MainProvision>
<Article Num="1"><ArticleCaption>（基本原則）</ArticleCaption>
<ArticleTitle>第一条</ArticleTitle><Paragraph><ParagraphSentence>
<Sentence>私権は、公共の福祉に適合しなければならない。</Sentence>
</ParagraphSentence></Paragraph></Article>
<Article Num="199"><ArticleCaption>（殺人）</ArticleCaption>
<ArticleTitle>第百九十九条</ArticleTitle><Paragraph><ParagraphSentence>
<Sentence>人を殺した者は、死刑又は無期拘禁刑に処する。</Sentence>
</ParagraphSentence></Paragraph></Article>
<Article Num="204"><ArticleCaption>（傷害）</ArticleCaption>
<ArticleTitle>第二百四条</ArticleTitle><Paragraph><ParagraphSentence>
<Sentence>人の身体を傷害した者は、十五年以下の拘禁刑に処する。</Sentence>
</ParagraphSentence></Paragraph></Article>
</MainProvision>
<SupplProvision>
<Article Num="1"><ArticleCaption>（施行期日）</ArticleCaption>
<ArticleTitle>第一条</ArticleTitle><Paragraph><ParagraphSentence>
<Sentence>この法律は、公布の日から施行する。</Sentence>
</ParagraphSentence></Paragraph></Article>
</SupplProvision>
</LawBody></Law>"""
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "law.xml"
        p.write_text(xml, encoding="utf-8")
        store = CrossStore()
        placed = ingest_law(store, p)
        main1 = ja_consensus_ask(store, "試験法第一条")
        suppl1 = ja_consensus_ask(store, "試験法附則第一条")
        art204 = ja_consensus_ask(store, "試験法第二百四条 傷害")
        wrong = ja_consensus_ask(store, "試験法第二百四条 殺人")

    ok = (placed.get("articles") == 4
          and main1.get("verdict") == "ANSWER"
          and "基本原則" in str(main1.get("text", ""))
          and suppl1.get("verdict") == "ANSWER"
          and "施行" in str(suppl1.get("text", ""))
          and art204.get("verdict") == "ANSWER"
          and wrong.get("verdict") != "ANSWER")
    return {
        "experiment": "cross_geometry",
        "fork": "EGOV_ARTICLE_IS_A_CITATION_KEY",
        "pass": bool(ok),
        "result": {
            "articles": placed.get("articles"),
            "main_first": main1.get("text"),
            "suppl_first": suppl1.get("text"),
            "correct_topic": art204.get("verdict"),
            "wrong_topic": wrong.get("verdict"),
        },
    }


def sovereign_build_fork() -> Dict[str, Any]:
    """The whole procedure, end to end, on a miniature federation.

    Three things are pinned, and the second is the one that keeps the first
    honest:

      * a question whose term lives in one leaf reaches it, with a path
      * a question whose term spans two DOMAINS refuses to choose, and
        names both — 緊急避難 is in the criminal code and the civil code,
        and picking one would be the fabrication this shape exists against
      * a term in neither comes back NOT PRESENT rather than as the nearest
        thing available

    `gather` lists rather than chooses, so it is allowed to return both
    branches where `descend` must refuse; both are exercised here because
    shipping only the listing path would quietly drop the refusal.
    """
    from .hierarchy import descend, gather
    from .sovereign import assemble

    crim = CrossStore()
    for s in ["刑法第三十七条は緊急避難である。", "刑法第三十七条は危難である。",
              "刑法第百九十九条は殺人である。", "刑法第百九十九条は死刑である。"]:
        _ingest_ja(crim, s)
    civ = CrossStore()
    for s in ["民法第七百二十条は緊急避難である。", "民法第七百二十条は不法行為である。",
              "民法第七百九条は損害賠償である。"]:
        _ingest_ja(civ, s)
    lab = CrossStore()
    for s in ["労働基準法第二十条は解雇である。", "労働基準法第二十条は予告である。"]:
        _ingest_ja(lab, s)

    domains = {"刑事": {"刑法": crim}, "民事": {"民法": civ}, "労働": {"労基": lab}}
    grouping = {"刑事": {"刑法": ["刑法"]}, "民事": {"民法": ["民法"]},
                "労働": {"労基": ["労基"]}}
    root = assemble(domains, grouping, sovereign="主権")

    one = gather(root, "解雇")
    both = gather(root, "緊急避難")
    absent = gather(root, "断水")
    chosen = descend(root, "緊急避難")

    domains_hit = {r["path"][1] for r in both["results"] if len(r["path"]) > 1}

    ok = (one["answered"] == 1
          and "労基" in one["results"][0]["leaf"]
          and both["destinations"] == 2
          and len(domains_hit) == 2
          and chosen["verdict"] != "ANSWER"
          and absent["verdict"] == "UNKNOWN_NOT_PRESENT")
    return {
        "experiment": "cross_geometry",
        "fork": "SOVEREIGN_BUILD",
        "pass": bool(ok),
        "result": {
            "single_leaf": {"answered": one["answered"],
                            "leaf": one["results"][0]["leaf"] if one["results"] else None},
            "spans_two_domains": {"destinations": both["destinations"],
                                  "domains": sorted(domains_hit),
                                  "descend_refused": chosen["verdict"]},
            "absent": absent["verdict"],
        },
    }


def _ingest_ja(store: CrossStore, sentence: str) -> None:
    from .document_ingest import Document, ingest_documents
    ingest_documents(store, [Document(source="fixture", text=sentence)])


def one_root_saturates_at_capacity_fork() -> Dict[str, Any]:
    """A single root routes CAPACITY terms and never more, at any depth.

    Measured on a twelve-field federation (1,098 leaves), asking the router
    alone — no index — about the very questions it was built from:

        anticipated   12    24    48    96   192   384   768
        correct       12    20    23    24    23    23    22

    The ABSOLUTE count saturates at 24 = MAX_ARMS x N_FACES and stops. Depth
    stayed 6 throughout, so layers BELOW the root do not raise it: the root
    has six arms and four faces each, and every question that fails there is
    lost whatever is underneath.

    The consequence for a federated design is the opposite of intuitive. A
    single sovereign is a routing ceiling; querying the field roots in
    parallel multiplies it by the number of fields — 220 of 1,536 against
    the sovereign's 24, on the same tree.

    This fixture is small enough to run as a unit test and shows the same
    shape: correct answers stop growing once the anticipated set passes what
    the root can hold.
    """
    from .hierarchy import descend
    from .sovereign import assemble

    # Twelve fields, each with a leaf carrying terms unique to it, so every
    # question has exactly one correct destination.
    doms: Dict[str, Dict[str, CrossStore]] = {}
    grps: Dict[str, Dict[str, List[str]]] = {}
    by_field: List[List[Any]] = []
    for d in range(12):
        field = f"分野{chr(0x4E00 + d)}"
        st = CrossStore()
        own: List[Any] = []
        for i in range(20):
            term = f"語{chr(0x4E00 + d)}{chr(0x4E00 + i)}"
            _ingest_ja(st, f"{field}項{chr(0x4E00 + i)}は{term}である。")
            own.append((term, field))
        by_field.append(own)
        doms[field] = {field: st}
        grps[field] = {field: [field]}
    # Round-robin, so every prefix of the anticipated set covers all twelve
    # fields. Ordered by field, the first twelve questions all pointed at one
    # branch and the root looked far worse than its capacity.
    key: List[Any] = [row[i] for i in range(20) for row in by_field]

    counts: Dict[int, int] = {}
    for n in (12, 24, 96, 240):
        asked = [q for q, _w in key[:n]]
        root = assemble(doms, grps, sovereign="主権", asked=asked)
        counts[n] = sum(
            1 for q, w in key[:n]
            if w in (descend(root, q, use_probe=False).get("path") or []))

    ceiling = MAX_ARMS_FORK * FACES_FORK
    # Grows while it fits, then stops at the ceiling however many more are
    # anticipated. The second half is the claim; without it this passes on a
    # tree that simply answers everything.
    ok = (counts[12] >= 10
          and counts[24] >= counts[12]
          and counts[96] <= ceiling
          and counts[240] <= ceiling
          and counts[240] <= counts[96] + 1)
    return {
        "experiment": "cross_geometry",
        "fork": "ONE_ROOT_SATURATES_AT_CAPACITY",
        "pass": bool(ok),
        "result": {"correct_by_anticipated": counts, "ceiling": ceiling},
    }


def fusion_is_not_monotonic_fork() -> Dict[str, Any]:
    """A field arriving can DISSOLVE a bridge between two others.

    Fusion is measured on concepts held by two or three fields, because a
    concept every field carries identifies none of them. That band is what
    makes the arrival of a third field able to subtract: a concept bridging
    A and B, once C also has it, spans three and on the next arrival falls
    out of the band entirely.

    Measured on the twelve-field federation:

        知財 arrives   opened 57, closed 37   民事×知財 +42, 刑事×民事 -17
        医療 arrives   opened 41, closed 66   医療×工学 +29, 工学×数学 -43

    So "combine the fields and watch fusion grow" is not what happens.
    Fusion between two fields is a function of everything else present, and
    a federation that only ever reports additions would show a rising number
    while the joins it names quietly changed underneath.

    Pinned here at a scale small enough to read: three fields where A and B
    share a concept, then C arrives carrying it too.
    """
    from .fusion import delta, index

    def mk(pairs: List[Tuple[str, str]]) -> CrossStore:
        st = CrossStore()
        for entity, term in pairs:
            _ingest_ja(st, f"{entity}は{term}である。")
        return st

    # Three fields share 保全 through two entities each — a join, in band.
    a = mk([("甲野第一条", "保全"), ("甲野第二条", "保全"),
            ("甲野第三条", "固有甲")])
    b = mk([("乙野第一条", "保全"), ("乙野第二条", "保全"),
            ("乙野第三条", "固有乙")])
    c = mk([("丙野第一条", "保全"), ("丙野第二条", "保全"),
            ("丙野第三条", "固有丙")])
    # A fourth is what pushes it out: BAND_HIGH is three, so a concept in
    # three fields is still a join. The band is the mechanism, not a filter
    # applied afterwards, and the fixture has to cross it.
    e = mk([("丁野第一条", "保全"), ("丁野第二条", "保全"),
            ("丁野第三条", "固有丁")])

    three = {"甲野": {"甲野": a}, "乙野": {"乙野": b}, "丙野": {"丙野": c}}
    four = dict(three, **{"丁野": {"丁野": e}})

    before = index(three)
    after = index(four)
    d = delta(before, after)

    bridged_before = "保全" in before.get("concepts", {})
    gone_after = "保全" not in after.get("concepts", {})

    ok = bridged_before and gone_after and "保全" in d["closed"]
    return {
        "experiment": "cross_geometry",
        "fork": "FUSION_IS_NOT_MONOTONIC",
        "pass": bool(ok),
        "result": {
            "points_before": before["n_points"],
            "points_after": after["n_points"],
            "bridge_before": bridged_before,
            "bridge_survives_arrival": not gone_after,
            "closed": d["closed"],
        },
    }


def word_form_is_a_fallback_fork() -> Dict[str, Any]:
    """Widen the query only when the printed word found nothing.

    Against an external key (topics ja.wikipedia assigns to specific
    articles), 46.7% of questions could not begin because the word the world
    uses is not the word the statute prints — 傷害罪 against a code that
    prints 傷害.

    Expanding every query fixed that and broke the answer:

        exact only        recall 26.7%   destinations   4
        always expand     recall 86.7%   destinations 117
        staged fallback   recall 73.3%   destinations  41

    不法行為 also matches 行為, which sits in 173 articles across six
    statutes, so an unconditional widening turns one question into most of
    the corpus. Staged keeps the printed term authoritative and reports
    which stage answered, so a reader can see the query was not the one
    they typed.
    """
    from .hierarchy import gather
    from .sovereign import assemble

    st = CrossStore()
    for s in ["刑法第二百四条は傷害である。", "刑法第二百四条は身体である。",
              "刑法第二百五条は傷害致死である。",
              "建築基準法第六条は建築である。", "建築基準法第六条は確認である。",
              "民法第七百九条は不法行為である。", "商法第五百九十条は行為である。",
              "商法第五百九十一条は行為である。"]:
        _ingest_ja(st, s)
    root = assemble({"法": {"法": st}}, {"法": {"法": ["法"]}}, sovereign="主権")

    exact = gather(root, "不法行為", limit=40, morph=True)
    suffix = gather(root, "傷害罪", limit=40, morph=True)
    compound = gather(root, "建築確認", limit=40, morph=True)
    off = gather(root, "傷害罪", limit=40, morph=False)

    ok = (exact["matched_as"] == "exact"            # printed term wins
          and exact["destinations"] == 1            # and is not widened
          and suffix["matched_as"] == "suffix"
          and suffix["destinations"] >= 1
          and compound["matched_as"] == "compound"
          and off["destinations"] == 0)             # off restores the old behaviour
    return {
        "experiment": "cross_geometry",
        "fork": "WORD_FORM_IS_A_FALLBACK",
        "pass": bool(ok),
        "result": {
            "exact": {"stage": exact["matched_as"], "dest": exact["destinations"]},
            "suffix": {"stage": suffix["matched_as"], "dest": suffix["destinations"]},
            "compound": {"stage": compound["matched_as"], "dest": compound["destinations"]},
            "morph_off": off["destinations"],
        },
    }


def linked_is_not_printed_fork() -> Dict[str, Any]:
    """A doctrinal connection must never look like something the source said.

    民法第七百九条 defines 不法行為 and does not contain the word. Measured
    over 271 published topic→article assignments that name articles this
    store holds, indexing the statutes reaches 25.5% of them; the other
    74.5% are connections a third party asserts and no statute prints.
    Raising the per-article index eightfold moved that number not at all.

    So they are kept in their own layer with the source that asserted them,
    and `resolve` returns the two kinds LABELLED. Writing the topic onto the
    article as a facet would make the coverage gate, the contradiction
    detector and the citation shown to a reader all treat a third party's
    doctrine as the legislature's text.
    """
    import tempfile
    from pathlib import Path as _P

    from .links import harvest, resolve

    store = CrossStore()
    _ingest_ja(store, "民法第七百九条は損害賠償である。")
    _ingest_ja(store, "民法第七百九条は侵害である。")
    _ingest_ja(store, "刑事訴訟法第二百十三条は現行犯人である。")
    fields = {"法": {"法": store}}

    with tempfile.TemporaryDirectory() as td:
        d = _P(td)
        (d / "不法行為.txt").write_text(
            "不法行為については民法第709条に定めがある。"
            "現行犯逮捕は刑事訴訟法第213条による。", encoding="utf-8")
        ls = harvest([d / "不法行為.txt"])

    r = resolve(fields, ls, "不法行為")
    linked = {x["article"] for x in r["linked"]}
    printed = {x["article"] for x in r["printed"]}
    # The store never learned the word, so nothing is printed...
    # ...and both articles arrive as links, each naming who said so.
    sources = {s for x in r["linked"] for s in x["asserted_by"]}

    ok = (ls.n_links() == 2
          and "刑事訴訟法第二百十三条" in linked
          and "民法第七百九条" in linked
          and not printed
          and sources == {"不法行為"}
          and all(x["in_store"] for x in r["linked"]))
    return {
        "experiment": "cross_geometry",
        "fork": "LINKED_IS_NOT_PRINTED",
        "pass": bool(ok),
        "result": {"links": ls.n_links(), "printed": sorted(printed),
                   "linked": sorted(linked), "asserted_by": sorted(sources)},
    }


def granularity_composes_fork() -> Dict[str, Any]:
    """Two granularities over one corpus can form what one cannot.

    A single store is closed: measured over 117 real queries the search
    emitted a symbol absent from the initial shell zero times, and it cannot,
    because faces come from the store and the moves permute faces. That is
    why it cannot fabricate and equally why it cannot generalise.

    Two stores at different granularities do not share the limit. The
    word-level store holds 断水 as an atom; the character-level store over
    the same text holds 断 and 水 AND their positions, and those positions
    license strings the word store cannot form.

    Measured on 5,371 legal words against 2.4M characters of held-out
    encyclopedia text that built neither model:

        proposals                       13,969
        appear as a substring            13.2%
        stand alone 3+ times              1.7%   <- words, not fragments
        same characters, paired randomly  0.1%

    自動 死去 公式 人権 主義 実数 定理 — none in the vocabulary that
    generated them. The control is the finding: what the character model
    contributes is not the characters, it is where they go.

    Longer words compose from UNITS, not from characters: 公務員 is 公務+員,
    not three characters chosen by position, which would propose 公再員. At
    three and four characters the yield falls and the advantage RISES —
    2字 x4.9, 3字 x15.5, 4字 x16.0 — because a random four-kanji string is
    almost never Japanese (0.01%), so structure is doing nearly all of it.

    This fork asserts the ADVANTAGE, not the yield. A yield alone could be a
    fact about how many kanji pairs happen to be Japanese.
    """
    from .granularity import (control, control_units, decompose,
                              decompose_units, propose, propose_units, verify)

    # A miniature corpus with a real positional regularity: 災/断/給 begin,
    # 害/水/電 end, and the held-out text contains the crossings.
    words = ["災害", "断水", "給電", "災難", "断熱", "給水"]
    held = ("停電が発生した。断電の記録はない。"
            "災水の被害は確認されていない。給熱の設備を点検する。"
            "停電は各地で発生し、停電の復旧が急がれる。停電。"
            "災水。災水の報告。給熱。給熱の点検。給熱。")

    model = decompose(words)
    got = verify(propose(model, top=6), held, min_standalone=2)
    ctl = verify(control(model, 40, seed=1), held, min_standalone=2)

    # Three characters, composed from units rather than positions.
    words3 = ["行政処分", "行政指導", "懲戒処分", "懲戒解雇"]
    held3 = ("行政解雇の例はない。懲戒指導が行われた。懲戒指導。懲戒指導の記録。"
             "行政解雇。行政解雇の通知。行政解雇。")
    m3 = decompose_units(words3)
    got3 = verify(propose_units(m3, length=4, top=8), held3, min_standalone=2)

    ok = (model.report()["words"] == 6
          and got["words"] >= 2                 # composed real strings
          and got["words"] > ctl["words"]       # and beat the same characters
          and got3["words"] >= 2)               # and units compose too
    return {
        "experiment": "cross_geometry",
        "fork": "GRANULARITY_COMPOSES",
        "pass": bool(ok),
        "result": {"model": model.report(),
                   "proposed_words": got["words"], "proposed_top": got["top"][:5],
                   "control_words": ctl["words"],
                   "unit_composition": got3["top"][:4]},
    }


def resolution_ladder_grades_doubt_fork() -> Dict[str, Any]:
    """A tied rung abstains, and unanimity therefore means something.

    The same corpus read at several grain sizes votes several ways, and how
    many rungs concur grades the answer. Measured on 10,222 statute articles
    asked by their own captions, 600 probes:

        best single rung, answering everything        29.8%
        three or four rungs answer and all agree     100.0%  (77 probes)
        two rungs answer and agree                    31.1%  (103)
        no rung had grounds                             --   (383)

    Three things this fork exists to stop coming back.

    RUNGS MUST BE NESTED. The first version binned terms by length, which is
    a partition: a term reaches exactly one rung, 507 of 600 probes had a
    single rung answer, and agreement was not low but impossible.

    A TIED RUNG MUST ABSTAIN. Breaking ties by insertion order made a rung's
    answer depend on ingest order. Breaking them lexicographically was
    WORSE: every tied rung chose the same smallest item, unanimity rose from
    86 probes to 321 and its accuracy fell from 73.3% to 23.7%, below the
    3-1 majority. Determinism at the tie manufactures agreement.

    REORDERING THE RUNGS MUST CHANGE NOTHING.
    """
    from .resolution import Ladder, ask, grains

    assert grains("損害賠償", 2) == ["損害", "害賠", "賠償"]

    items = {
        "甲条": {"損害賠償", "責任"},
        "乙条": {"損害", "賠償", "損失", "賠責"},
        "丙条": {"契約", "範囲"},
    }
    ladder = Ladder().build(items)

    every_rung_sees = ("損害賠償" in ladder.index["whole"]
                       and "損害" in ladder.index["g2"]
                       and "損" in ladder.index["g1"])

    unanimous = ask(ladder, ["契約"])

    # 損害賠償 is a whole term of 甲条 and is spelled out piecewise by 乙条,
    # so the character rung cannot separate them and says nothing.
    votes = ladder.vote(["損害賠償"])
    abstained = [r for r, v in votes.items() if v is None]
    spoke = ask(ladder, ["損害賠償"])

    flipped = Ladder(rungs=(("g1", 1), ("g2", 2), ("g3", 3), ("whole", 0)))
    flipped.build(items)
    stable = (ask(flipped, ["損害賠償"])["item"] == spoke["item"]
              and ask(flipped, ["契約"])["item"] == unanimous["item"])

    ok = (every_rung_sees
          and unanimous["item"] == "丙条"
          and unanimous["verdict"] == "ANSWER"
          and unanimous["concord"] == 1.0
          and abstained == ["g1"]          # the tied rung, and only it
          and spoke["answered"] == 3       # the others still spoke
          and stable)
    return {
        "experiment": "cross_geometry",
        "fork": "RESOLUTION_LADDER_GRADES_DOUBT",
        "pass": bool(ok),
        "result": {
            "unanimous": {k: unanimous[k] for k in
                          ("item", "verdict", "answered", "majority", "concord")},
            "tie_abstained": abstained,
            "after_abstention": {k: spoke[k] for k in
                                 ("item", "verdict", "answered", "concord")},
            "stable_under_rung_reorder": stable,
        },
    }


def concord_rides_alongside_the_list_fork() -> Dict[str, Any]:
    """Confidence is reported BESIDE the destinations, never instead of them.

    `gather` lists every leaf holding the query terms and chooses nothing —
    that is what makes it unable to fabricate. The resolution ladder does
    choose, and says how much of itself concurred. Wiring the second into
    the first must not let it overwrite the first, or the listing stops
    being a listing.

    Two behaviours are pinned. A multi-term query that the rungs agree on
    reports the leaf AND stays in the list. A single-term query cannot be
    discriminated — every leaf holding it ties — so every rung abstains and
    concord is zero. That is correct rather than broken: the leaves really
    are indistinguishable on one term, and `gather` has already said so by
    listing them all.
    """
    from .hierarchy import gather
    from .sovereign import assemble

    a = CrossStore()
    for s in ["労基第二十条は解雇である。", "労基第二十条は予告である。",
              "労基第二十条は三十日である。"]:
        _ingest_ja(a, s)
    b = CrossStore()
    for s in ["労基第八十九条は解雇である。", "労基第八十九条は就業規則である。"]:
        _ingest_ja(b, s)
    root = assemble({"労働": {"予告": a, "規則": b}},
                    {"労働": {"労働": ["予告", "規則"]}}, sovereign="主権")

    sharp = gather(root, "解雇 予告 三十日", limit=8, concord=True)
    blunt = gather(root, "解雇", limit=8, concord=True)
    plain = gather(root, "解雇 予告 三十日", limit=8)

    ok = (sharp["concord"]["verdict"] == "ANSWER"
          and sharp["concord"]["concord"] == 1.0
          and sharp["concord"]["in_destinations"]
          # the list is unchanged by asking for confidence
          and sharp["destinations"] == plain["destinations"]
          and "concord" not in plain
          # both leaves hold 解雇, so one term cannot separate them
          and blunt["destinations"] == 2
          and blunt["concord"]["answered_rungs"] == 0)
    return {
        "experiment": "cross_geometry",
        "fork": "CONCORD_RIDES_ALONGSIDE_THE_LIST",
        "pass": bool(ok),
        "result": {
            "multi_term": {k: sharp["concord"][k] for k in
                           ("item", "verdict", "agreeing", "answered_rungs",
                            "in_destinations")},
            "single_term": {"destinations": blunt["destinations"],
                            "answered_rungs": blunt["concord"]["answered_rungs"]},
            "list_unchanged": sharp["destinations"] == plain["destinations"],
        },
    }


def long_form_drifts_and_lists_fork() -> Dict[str, Any]:
    """Chained read-out is not prose, and it loses its subject in two steps.

    Asked whether this engine can generate long form for creative use, the
    answer is measured rather than argued. Following a core to one of its
    own facets and reading again, 40 seeds per field:

        field   mean steps   steps still on topic
        数学        7.9            1.7
        工学        7.4            1.9
        経済        6.7            1.7
        医療        6.2            1.3
        刑事/民事    1.0            1.0

    A chain runs six to eight steps and stops being about anything after
    one or two. Observed: 細胞 -> 現象 -> 関係 -> 副作用 -> アメリカ ->
    学術雑誌.

    And every step reads out as 「Xは A、B、C、D」 — the four faces joined by
    a comma. That is a fact list, not a sentence, and no amount of chaining
    turns a list into prose.

    Both properties follow from the closure this design is built on: the
    read-out can only emit stored symbols, and the only ordering it has is
    the face order. Composition into sentences would need something that
    generates function words, which is exactly what a closed system cannot
    do — see `granularity`, where breaking the closure at a granularity
    boundary is the one route that produces anything new, and produces
    single words rather than clauses.

    This fork exists so the limitation is not quietly re-discovered as a
    bug. It asserts the shape, not the numbers.
    """
    from .consensus_store import ja_consensus_ask

    store = CrossStore()
    for s in ["甲は乙である。", "甲は丙である。", "乙は丁である。", "乙は戊である。",
              "丁は己である。", "丁は庚である。", "己は辛である。", "己は壬である。"]:
        _ingest_ja(store, s)

    seed = "甲"
    start = {f for f in store.crosses.get(seed, {})
             if f not in store.source_labels}
    cur, seen, texts, on_topic = seed, [], [], 0
    still = True
    for _ in range(6):
        out = ja_consensus_ask(store, cur)
        if out.get("verdict") != "ANSWER":
            break
        core = out.get("core")
        seen.append(core)
        texts.append(out.get("text", ""))
        facets = {f for f in store.crosses.get(core, {})
                  if f not in store.source_labels}
        if still and (facets & start):
            on_topic += 1
        elif still:
            still = False
        nxt = sorted(f for f in facets if f in store.crosses and f not in seen)
        if not nxt:
            break
        cur = nxt[0]

    listy = all("、" in t or len(t.split("は")[-1].split("、")) >= 1
                for t in texts if t)
    ok = (len(seen) >= 3                 # the chain does run
          and on_topic < len(seen)       # and leaves its subject before the end
          and listy)                     # and every step is a list
    return {
        "experiment": "cross_geometry",
        "fork": "LONG_FORM_DRIFTS_AND_LISTS",
        "pass": bool(ok),
        "result": {"steps": len(seen), "on_topic_steps": on_topic,
                   "path": seen, "sample": texts[:3]},
    }


def trace_is_memory_outside_the_store_fork() -> Dict[str, Any]:
    """The path is a value, kept beside the stores, and it resumes a walk.

    A read-out chain had no memory: each step saw only its current core, so
    the next was chosen without reference to where the walk began or what it
    had passed. I first reported that as the structure forgetting its
    subject. It was not — nothing was remembering. 40 seeds per field:

        rule            steps   subject held   adjacent overlap   with path
        lexicographic     8.7        1.7            0.033           0.269
        anchor on seed   10.9        6.9            0.042           0.385
        anchor on path   11.0        5.0            0.069           0.433

    Two different instruments. Anchoring on the SEED holds the original
    subject four times longer, which is what a question wants. Anchoring on
    the PATH walks furthest and overlaps both its previous step and
    everything before it most, which is what a developing text wants.

    A trace is NOT knowledge — nothing in it was said by a source — so it
    lives outside the stores, where a later reader cannot mistake a footprint
    for a fact. Being a value rather than a local variable is also what makes
    a walk resumable and replayable, which a deterministic engine should be
    and was not.
    """
    from .trace import Trace, replay, walk

    store = CrossStore()
    for s in ["甲は乙である。", "甲は丙である。", "乙は丁である。", "乙は丙である。",
              "丁は戊である。", "丁は丙である。", "戊は己である。", "戊は丙である。"]:
        _ingest_ja(store, s)

    t = walk(store, "甲", mode="path", steps=2)
    first = list(t.seen)

    import tempfile
    from pathlib import Path as _P
    with tempfile.TemporaryDirectory() as td:
        p = _P(td) / "t.json"
        t.save(p)
        back = Trace.load(p)
        round_trip = (back.seen == t.seen and back.horizon == t.horizon)
        resumed = walk(store, "甲", mode="path", steps=2, trace=back)

    rep = replay(store, resumed)
    unchanged = all(not r["changed"] for r in rep)

    # The horizon grows; it is the anchor and it is not the seed's alone.
    grew = len(resumed.horizon) >= len(t.horizon)

    ok = (len(first) >= 2
          and round_trip
          and len(resumed.seen) > len(first)      # resuming continues
          and resumed.seen[:len(first)] == first  # from where it stopped
          and unchanged                           # replay is exact
          and grew)
    return {
        "experiment": "cross_geometry",
        "fork": "TRACE_IS_MEMORY_OUTSIDE_THE_STORE",
        "pass": bool(ok),
        "result": {"first_walk": first, "resumed": resumed.seen,
                   "round_trip": round_trip, "replay_unchanged": unchanged,
                   "horizon": len(resumed.horizon)},
    }


def constellation_beats_one_sovereign_fork() -> Dict[str, Any]:
    """Parallel sovereigns at different settings, with nothing above them.

    One apex is a routing ceiling — six arms, four faces, 24 terms, and no
    depth beneath raises it. Measured: 24 placed against 220 for the same
    tree queried at its field roots in parallel.

    A constellation holds the corpus several times over, each member at its
    own setting, and reports a CENSUS rather than a verdict. At 626MB over
    5,401 leaves, 1,200 probes:

        single sovereign     80.3%
        eight in parallel    84.3%

        agreeing  7   44 probes  100.0%
                  6   92         100.0%
                  5  191         100.0%
                  4  499         100.0%
                  3  119          97.5%
                  2   88          67.0%
                  1  145           7.6%

    Four or more concurring covered 69% of probes and was right every time.

    Three things are pinned. Unanimity reports itself as ANSWER with concord
    1.0. A split is reported as a split and NOT resolved into a winner —
    voting a disagreement into an answer is the failure a census exists to
    avoid. And a member with no single leader ABSTAINS rather than breaking
    the tie, because a tie-break shared across members manufactures
    agreement for a reason unrelated to evidence — measured one level down,
    where it took unanimity from 86 probes to 321 and its accuracy from
    73.3% to 23.7%.
    """
    from .constellation import Constellation, Sovereign

    # Members that genuinely see different things: this is what varying a
    # setting produces, constructed directly so the fork tests the census
    # rather than the indexing.
    coarse = Sovereign("coarse", {"rungs": (("whole", 0),)}).build({
        "甲章": {"損害賠償", "責任"},
        "乙章": {"契約", "解除"},
    })
    fine = Sovereign("fine", {"rungs": (("whole", 0),)}).build({
        "甲章": {"責任"},
        "乙章": {"損害賠償", "契約"},
    })
    third = Sovereign("third", {"rungs": (("whole", 0),)}).build({
        "甲章": {"損害賠償"},
        "乙章": {"契約", "解除"},
    })
    c = Constellation().add(coarse).add(fine).add(third)

    split = c.ask(["損害賠償"])
    unanimous = c.ask(["契約"])

    # One member sees 損害賠償 in 乙章, two in 甲章 — a majority, not a verdict.
    votes = {m.name: m.vote(["損害賠償"]) for m in c.members}
    differ = len({v for v in votes.values() if v}) > 1

    # A member whose top score ties abstains rather than picking.
    tied = Sovereign("tied", {"rungs": (("whole", 0),)}).build({
        "甲章": {"共通"}, "乙章": {"共通"},
    })
    abstains = tied.vote(["共通"]) is None

    ok = (unanimous["verdict"] == "ANSWER"
          and unanimous["concord"] == 1.0
          and unanimous["item"] == "乙章"
          and differ
          and split["verdict"] == "MAJORITY"   # reported as a majority...
          and split["agreeing"] == 2           # ...with the count visible
          and split["spoke"] == 3
          and abstains)
    return {
        "experiment": "cross_geometry",
        "fork": "CONSTELLATION_BEATS_ONE_SOVEREIGN",
        "pass": bool(ok),
        "result": {
            "unanimous": {k: unanimous[k] for k in
                          ("item", "verdict", "spoke", "agreeing", "concord")},
            "split": {k: split[k] for k in
                      ("item", "verdict", "spoke", "agreeing", "concord")},
            "votes": votes,
            "tied_member_abstains": abstains,
        },
    }


def writer_never_reaches_the_answer_path_fork() -> Dict[str, Any]:
    """A generated sentence must not be able to arrive where a citation is.

    The whole closure argument elsewhere is that a reader can trust what
    comes back: every symbol traces to a source, measured at 0 of 117
    outputs containing anything the store did not hold. A composed sentence
    breaks that on purpose — it is a recombination of a form somebody wrote
    and content somebody else wrote, and the combination is nobody's claim.

    So `writer` is imported by nothing on the answer path, and every draft
    carries both sources so the two can be told apart. This fork asserts the
    isolation as a fact about the code, not a convention.
    """
    import inspect

    from . import consensus_store, hierarchy, writer
    from .cross_store import CrossStore
    from .writer import Writer

    # Nothing that produces a verdict may reach the generator.
    leak = [m.__name__ for m in (consensus_store, hierarchy)
            if "writer" in inspect.getsource(m)
            or "compose_ja" in inspect.getsource(m)]

    store = CrossStore()
    for s in ["甲条は届出である。", "甲条は選択である。", "甲条は事情である。"]:
        _ingest_ja(store, s)
    # The prose must attest the SUBJECT too, not just the facets: a term the
    # corpus never uses standing alone is not a word, and `sentence` returns
    # nothing for it. Without 甲条 here the fork passed on an empty draft
    # list and asserted nothing about citations at all.
    # A RECORD-register form on purpose. The corpus here reports; it does
    # not legislate, so a 「〜することができる」 shape would be refused by the
    # modality licence and this fork would be testing that instead — see
    # `form_may_not_assert_more_than_content_licenses_fork`.
    prose = [("fixture",
              "甲条は、いつでも届出を選択した。"
              "事情は、いつでも届出を選択した。"
              "甲条は、いつでも事情を選択した。"
              "甲条は、いつでも届出を選択した。"
              "事情は、いつでも甲条を選択した。")]
    w = Writer.build([store], prose)

    drafts = w.sentence(store, "甲条")
    unknown = w.sentence(store, "存在しない語")

    both_cited = bool(drafts) and all(d.get("content_from") and d.get("form_from")
                                      for d in drafts)

    ok = (not leak
          and w.report()["vocabulary"]["terms"] > 0
          and drafts                 # the fixture must actually write
          and unknown == []          # a non-word writes nothing
          and both_cited)
    return {
        "experiment": "cross_geometry",
        "fork": "WRITER_NEVER_REACHES_THE_ANSWER_PATH",
        "pass": bool(ok),
        "result": {
            "answer_path_imports_generator": leak,
            "built": w.report(),
            "drafts": [d["text"] for d in drafts][:2],
            "non_word_writes_nothing": unknown == [],
        },
    }


def form_may_not_assert_more_than_content_licenses_fork() -> Dict[str, Any]:
    """Closure stops invented symbols. It does not stop invented relations.

    A store holds co-occurrence. A form holds a relation, and supplies it
    whoever fills the holes: 「<0>は、<1>を<2>しなければならない」 states an
    obligation about anything put in hole 0. Measured over 278 generated
    sentences on the 626MB federation, 21.9% asserted a norm — obligation,
    prohibition, or permission — and 42.6% of those were about content no
    statute had ever spoken of. 【女性】は、【従事】を【行為】しなければ
    ならない passes closure on every symbol and fabricates the only thing in
    it that carries meaning.

    So a norm-shaped form needs a norm-registered subject. The restriction
    runs one way: a legal duty stated as a plain fact loses information, a
    plain fact stated as a legal duty invents an obligation.
    """
    from .cross_store import CrossStore
    from .writer import Writer

    store = CrossStore()
    for s in ["甲条は届出である。", "甲条は選択である。", "甲条は事情である。"]:
        _ingest_ja(store, s)

    norm = ("甲条は、いつでも届出を選択しなければならない。" * 3
            + "事情は、いつでも届出を選択しなければならない。"
            + "甲条は、いつでも事情を選択しなければならない。")

    # Same corpus, same store, same subject — only the licence differs.
    unlicensed = Writer.build([store], [("prose", norm)])
    licensed = Writer.build([store], [("statute", norm)],
                            norm_corpora=["statute"])

    a = unlicensed.sentence(store, "甲条")
    b = licensed.sentence(store, "甲条")

    norm_forms = [f for f in licensed.forms.values() if f.register == "norm"]

    ok = (norm_forms                  # the fixture really did offer a norm
          and a == []                 # unlicensed content writes no norm
          and b                       # licensed content still writes
          and unlicensed.licence("甲条") == "record"
          and licensed.licence("甲条") == "norm")
    return {
        "experiment": "cross_geometry",
        "fork": "FORM_MAY_NOT_ASSERT_MORE_THAN_CONTENT_LICENSES",
        "pass": bool(ok),
        "result": {
            "norm_forms_offered": len(norm_forms),
            "licence_without_statute": unlicensed.licence("甲条"),
            "licence_with_statute": licensed.licence("甲条"),
            "unlicensed_wrote": [d["text"] for d in a],
            "licensed_wrote": [d["text"] for d in b],
        },
    }


def concord_is_not_coverage_fork() -> Dict[str, Any]:
    """Agreement about a leaf is not evidence the question was answered.

    The census bands are sharp on questions the corpus can answer: over 300
    in-corpus probes, three or more concurring was right 185 times out of
    185, two 71.4%, one 8.7%. That calibration was built from anticipated
    questions over the corpus, so it says nothing about a question the
    corpus cannot answer — and the two come apart hardest in the middle
    case, where a query mixes a term the federation holds with one it does
    not. Measured over 80 such questions only 3 were refused and 8 reached
    four or more concurring, the band calibrated at 100%, each time by
    answering about the other word.

    Coverage is reported beside the concord and never folded into it. A
    reader who wants the leaf anyway can still have it; a reader who assumed
    a high count meant their term was addressed can now see it was not.
    """
    from .constellation import staircase

    items = {"甲葉": ["登記", "申請", "期間"], "乙葉": ["解雇", "予告", "賃金"]}
    c = staircase(items)

    hit = c.ask(["登記"])
    mixed = c.ask(["登記", "超伝導"])
    absent = c.ask(["超伝導"])

    ok = (hit["coverage"] == 1.0 and hit["missing"] == []
          # the mixed query still answers, and still says what it missed
          and mixed["verdict"] in ("ANSWER", "MAJORITY")
          and mixed["missing"] == ["超伝導"]
          and mixed["coverage"] == 0.5
          # coverage does NOT quietly lower the concord
          and mixed["agreeing"] == hit["agreeing"]
          and absent["verdict"] == "UNKNOWN_NOT_PRESENT"
          and absent["coverage"] == 0.0)
    return {
        "experiment": "cross_geometry",
        "fork": "CONCORD_IS_NOT_COVERAGE",
        "pass": bool(ok),
        "result": {
            "in_corpus": {k: hit[k] for k in
                          ("verdict", "agreeing", "coverage", "missing")},
            "mixed": {k: mixed[k] for k in
                      ("verdict", "agreeing", "coverage", "missing")},
            "absent": {k: absent[k] for k in
                       ("verdict", "agreeing", "coverage", "missing")},
        },
    }


def every_manifest_can_rebuild_its_corpus_fork() -> Dict[str, Any]:
    """A manifest that cannot rebuild its corpus is a receipt, not a backup.

    `corpus_fetch` was written because a corpus was lost in a session temp
    directory. It was lost from one a second time, and the manifests are why
    the two losses ended differently: every e-Gov entry carried a URL and
    came back, while all 202 Wikipedia entries carried an empty one and had
    to be reconstructed from their filenames by a module written after the
    fact. The manifest could prove the corpus was gone and not get it back.

    Three more entries reported no URL for a different reason — index.json
    and urls.tsv were written BY the fetch rather than fetched, so recording
    them as corpus files made two manifests report themselves irreproducible
    over their own bookkeeping.

    Reproducibility is a property of the manifest, checkable without the
    network and without the corpus, so it is checked here every run.
    """
    import json
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent / "corpora"
    seen, broken = [], []
    for path in sorted(root.glob("*.json")):
        m = json.loads(path.read_text(encoding="utf-8"))
        files = m.get("files") or []
        missing = [e["name"] for e in files if not e.get("url")]
        seen.append({"manifest": path.stem, "files": len(files),
                     "without_url": len(missing)})
        if missing or not m.get("reproducible"):
            broken.append({"manifest": path.stem, "examples": missing[:3],
                           "declared": m.get("reproducible")})

    ok = bool(seen) and not broken
    return {
        "experiment": "cross_geometry",
        "fork": "EVERY_MANIFEST_CAN_REBUILD_ITS_CORPUS",
        "pass": bool(ok),
        "result": {"manifests": seen, "not_reproducible": broken},
    }


def a_reloaded_writer_is_the_same_writer_fork() -> Dict[str, Any]:
    """Restoring must bring back the slot tables, not just the vocabulary.

    `SELECTION` and `SELECTION_TAIL` are module globals — `selects` is
    called from inside the fill loop and threading them through every caller
    bought nothing. That makes them the one part of the writer a reload
    cannot recover on its own, and the failure is quiet rather than loud: a
    writer restored without them still has its vocabulary, still has its
    forms, still writes sentences, and answers `selects` with None on every
    fill. Every candidate becomes "unknown but not refused", which is a
    different system that looks like this one.

    So the fork restores into a process whose tables were deliberately
    cleared and requires the same sentence back.
    """
    from .compose_ja import SELECTION, SELECTION_TAIL, learn_selection, selects
    from .cross_store import CrossStore
    from .writer import Writer

    store = CrossStore()
    for s in ["甲条は届出である。", "甲条は選択である。", "甲条は事情である。"]:
        _ingest_ja(store, s)
    prose = [("fixture",
              "甲条は、いつでも届出を選択した。" * 3
              + "事情は、いつでも届出を選択した。"
              + "甲条は、いつでも事情を選択した。")]

    saved_sel = {k: v.copy() for k, v in SELECTION.items()}
    saved_tail = {k: v.copy() for k, v in SELECTION_TAIL.items()}
    try:
        w = Writer.build([store], prose)
        before = w.sentence(store, "甲条")
        attested_before = selects("届出", "を", "選択")

        with tempfile.TemporaryDirectory() as d:
            path = Path(d) / "writer.json"
            w.save(path)
            # Wipe the globals: this is what a fresh process looks like.
            SELECTION.clear()
            SELECTION_TAIL.clear()
            blind = selects("届出", "を", "選択")
            w2 = Writer.load(path)
            after = w2.sentence(store, "甲条")
            attested_after = selects("届出", "を", "選択")

        ok = (before and before == after
              and attested_before is True
              and blind is None              # the wipe really did blind it
              and attested_after is True)
        return {
            "experiment": "cross_geometry",
            "fork": "A_RELOADED_WRITER_IS_THE_SAME_WRITER",
            "pass": bool(ok),
            "result": {
                "before": [d["text"] for d in before],
                "after": [d["text"] for d in after],
                "selects_before": attested_before,
                "selects_while_wiped": blind,
                "selects_after_reload": attested_after,
            },
        }
    finally:
        SELECTION.clear()
        SELECTION.update(saved_sel)
        SELECTION_TAIL.clear()
        SELECTION_TAIL.update(saved_tail)


def coarsening_adds_a_reading_and_never_overturns_one_fork() -> Dict[str, Any]:
    """The verdict is quantized like the index, and typed so it cannot lie.

    刑法第百九十九条 carries the facet 殺人 from its own caption, and the
    doctrinal term is 殺人罪. One character apart, and the whole-grain gate
    reported `query_terms_not_addressed` exactly as it would for a word the
    corpus had never held. Running the judgment at several grains tells the
    two apart.

    Measured over 500 morphological variants that are NOT themselves cores,
    so the whole grain must fail: 468 came back ANSWER_BY_COARSENING and
    98.7% of those reached the ORIGINAL core — 100.0% at two or more settings
    agreeing, 95.0% at one. Against 18 variants of words the corpus never
    held, 17 refused and 1 answered, which is the cost of coarsening and the
    reason a coarse answer is never typed ANSWER.

    This fork pins the type discipline rather than the accuracy: a reading
    only a coarse setting reached must be labelled as one, and a term the
    store does not hold at any grain must still refuse.
    """
    from .cross_store import CrossStore
    from .graded import GradedJudge

    store = CrossStore()
    for s in ["傷害罪は暴行である。", "傷害罪は故意である。", "傷害罪は結果である。",
              "殺人は死刑である。", "殺人は無期である。", "殺人は拘禁刑である。"]:
        _ingest_ja(store, s)
    j = GradedJudge().build(store)

    exact = j.ask("傷害罪とは")          # the store holds this name
    coarse = j.ask("殺人罪とは")         # holds 殺人, asked as 殺人罪
    absent = j.ask("超伝導とは")         # holds nothing like it

    ok = (exact["verdict"] == "ANSWER"
          and exact["item"] == "傷害罪"
          # a coarsened reading is REACHED and is not called ANSWER
          and coarse["verdict"] == "ANSWER_BY_COARSENING"
          and coarse["item"] == "殺人"
          and coarse["readings"]["whole"] is None
          # and coarsening does not manufacture an answer from nothing
          and absent["verdict"] == "UNKNOWN_NOT_PRESENT")
    return {
        "experiment": "cross_geometry",
        "fork": "COARSENING_ADDS_A_READING_AND_NEVER_OVERTURNS_ONE",
        "pass": bool(ok),
        "result": {
            "exact": {k: exact[k] for k in ("verdict", "item", "agreeing")},
            "coarsened": {k: coarse[k] for k in ("verdict", "item", "agreeing")},
            "coarsened_strict_reading": coarse["readings"]["whole"],
            "absent": {k: absent[k] for k in ("verdict", "item", "agreeing")},
        },
    }


def a_citation_is_listed_not_chosen_fork() -> Dict[str, Any]:
    """Three readings from one ask, kept apart because merging destroys them.

    A sovereign was only ever a ladder: placement, hierarchy, granularity
    and links were built, measured, and never reached by one. Wiring them in
    produced two results worth pinning.

    Merging citations into the core ladder scored 0 of 387 gold links. The
    first wiring added the cited articles to the TOPIC's terms, so asking
    わいせつ returned the core わいせつ — what it already returned. Keying
    the ARTICLE by the topic instead answers, and then ties: わいせつ is
    provided for by 刑法第百七十四条 through 第百七十七条, four articles at
    one point each, and a ladder asked which ONE abstains. Choosing answered
    54 of 130 topics and was right every time; listing reaches 164 of 164.

    Listing is not choosing, so it cannot fabricate — and the citation stays
    a fact about the citing document, never a claim the statute made.
    """
    from .cross_store import CrossStore
    from .full_sovereign import FullSovereign, DEFAULT_SETTINGS

    store = CrossStore()
    for s in ["刑法第百七十四条はわいせつである。", "刑法第百七十五条はわいせつである。",
              "わいせつは公然である。", "わいせつは頒布である。"]:
        _ingest_ja(store, s)

    sv = FullSovereign(name="cites", setting=dict(DEFAULT_SETTINGS)["cites"])
    sv.build(store, None, shared={}, link_paths=[],
             with_tree=False, with_placement=False)
    # Stand in for a harvest: one topic, two articles — the tie case.
    sv.links = {"わいせつ": ["刑法第百七十四条", "刑法第百七十五条"]}

    listed = sv.cited(["わいせつ"])
    none = sv.cited(["超伝導"])

    ok = (listed["verdict"] == "ANSWER"
          # BOTH articles, not one picked
          and len(listed["articles"]) == 2
          and "刑法第百七十四条" in listed["articles"]
          and "刑法第百七十五条" in listed["articles"]
          # the topic that cited them travels with them
          and listed["by_topic"]["わいせつ"]
          # and a topic nobody cited gets a typed refusal, not an empty answer
          and none["verdict"] == "UNKNOWN_NO_CITATION"
          and none["articles"] == [])
    return {
        "experiment": "cross_geometry",
        "fork": "A_CITATION_IS_LISTED_NOT_CHOSEN",
        "pass": bool(ok),
        "result": {"listed": listed["articles"],
                   "by_topic": listed["by_topic"],
                   "uncited": none["verdict"]},
    }


def a_template_cut_inside_a_word_is_caught_at_the_seam_fork() -> Dict[str, Any]:
    """The break shows at the fill, not at the harvest, so test it there.

    `_CUT_STEM` lists eleven inflection fragments by hand — a rule written
    rather than learned, and measured to catch NONE of the 659 forms
    harvested at 626MB. 「う」 is not on the list, so 「<0>うものとする」
    survives and fills to 法律うものとする.

    The fragment is not the defect. 「う」 is a fine ending for 行う. The
    JOIN is: 律+う never occurs in 32,259,912 characters of this corpus and
    律+は occurs 3,863 times. Both sides of the seam are only known once a
    term has been chosen, so the test belongs at fill time.

    Measured over 465 generated sentences: 18% had an unattested seam
    before, 0% after, at the same 465 sentences — free, because a rejected
    form falls through to the next one. The threshold is one occurrence;
    anything higher starts refusing rare words for being rare (解雇+は
    occurs once in a 8.7M-character sample).
    """
    from .compose_ja import (JOIN, compose, harvest, joins, learn_joins,
                             learn_selection)
    from .vocabulary import attest

    body = ("法律は、届出を行うものとする。" * 4
            + "行為は、届出を行うものとする。" * 4
            + "法律は、届出である。" * 4)
    corpora = [("fixture", body)]
    saved = Counter(JOIN)
    try:
        JOIN.clear()
        learn_joins(corpora)
        # 律+う is written here (法律は…行う never puts them adjacent), so the
        # fixture must show the pair genuinely absent.
        seam_bad = joins("法律", "う")
        seam_good = joins("法律", "は")

        forms = harvest(corpora)
        learn_selection(corpora)
        vocab = attest(["法律", "行為", "届出"], corpora)
        drafts = compose(forms, "法律", ["届出", "行為"], limit=3,
                         content_from=["法律"], vocab=vocab)
        texts = [d.text for d in drafts]
        broken = [x for x in texts if "法律う" in x or "行為う" in x]

        ok = (not seam_bad and seam_good and not broken)
        return {
            "experiment": "cross_geometry",
            "fork": "A_TEMPLATE_CUT_INSIDE_A_WORD_IS_CAUGHT_AT_THE_SEAM",
            "pass": bool(ok),
            "result": {"joins_法律_う": seam_bad, "joins_法律_は": seam_good,
                       "drafts": texts[:3], "broken": broken},
        }
    finally:
        JOIN.clear()
        JOIN.update(saved)


def presence_in_the_corpus_is_not_support_fork() -> Dict[str, Any]:
    """A verification layer must check the SUBJECT's link, not the vocabulary.

    Put an LLM in the generation layer and Vera underneath as verification
    and citation, and everything rests on the check actually catching an
    unsupported sentence. A first version asked whether the corpus held each
    term anywhere. Measured against a local 4B model over 14 subjects it
    ranked FREE generation ABOVE grounded — 95.7% term presence to 85.5% —
    because a fluent answer about Japanese law is built from 法律, 制定,
    原則, 国民 and a federation of 54,244 legal cores holds all of them. In
    a large corpus presence is nearly free and a checker built on it passes
    everything, which is worse than no checker: it staples a citation to a
    fluent invention.

    The subject's own cross is not free. Asked about 第37条 the model wrote
    「国家の権限を保障し…」 — fluent, plausible, sharing nothing with what
    the store records there. Same 14 subjects, scored that way: grounded
    64.1%, free 6.4%; at a 30% threshold 0 of 14 grounded flagged and 14 of
    14 free flagged.
    """
    from .attest_llm import check_all
    from .cross_store import CrossStore

    store = CrossStore()
    for s in ["第三十七条は補償である。", "第三十七条は費用である。",
              "第三十七条は請求である。", "法律は制定である。",
              "法律は国家である。", "国家は権限である。"]:
        _ingest_ja(store, s)

    # Both sentences use only words this store holds. Only one of them says
    # what the store says about the subject.
    grounded = check_all(store, "第三十七条", "第三十七条は補償の費用を請求する。")
    invented = check_all(store, "第三十七条", "第三十七条は国家の権限を制定する。")

    ok = (grounded["verdict"] == "ANSWER"
          and invented["verdict"] == "UNSUPPORTED_BY_CORPUS"
          and grounded["support"] > invented["support"]
          # every word of the rejected sentence IS in the corpus
          and all(w in store.crosses or any(w in (c or ()) for c in store.crosses.values())
                  for w in ("国家", "権限", "制定")))
    return {
        "experiment": "cross_geometry",
        "fork": "PRESENCE_IN_THE_CORPUS_IS_NOT_SUPPORT",
        "pass": bool(ok),
        "result": {
            "grounded": {k: grounded[k] for k in ("verdict", "support")},
            "invented": {k: invented[k] for k in ("verdict", "support")},
            "invented_unlinked": invented["reports"][0]["unlinked"],
        },
    }


def a_wholesale_replacement_is_not_no_change_fork() -> Dict[str, Any]:
    """Drift has to name what moved, because a count cannot see the worst case.

    A store that is written to keeps moving, and "it grew" is rarely the
    interesting event. The one that matters is a core recorded one way and
    now recorded another with nobody deciding — and if a baseline reports
    only totals, the case where every facet was swapped and the SIZE stayed
    the same reads as no change at all.

    So `compare` lists added, removed and changed rather than summing them,
    and flags `replaced` explicitly. DRIFTED is not a failure and STABLE is
    not a pass: a gain is usually a lesson, a loss is usually a correction,
    and only a reader knows which the design intended.
    """
    from .cross_store import CrossStore
    from .drift import compare, snapshot

    store = CrossStore()
    for s in ["甲条は届出である。", "甲条は選択である。", "乙条は事情である。"]:
        _ingest_ja(store, s)
    base = snapshot(store, label="design", keep=["甲条"])

    same = compare(base, store)

    # Swap every facet, keep the count identical.
    before = sorted(base["kept"]["甲条"])
    store.crosses["甲条"] = {f + "改": 1 for f in before}
    after = compare(base, store)
    det = next((d for d in after["detail"] if d["core"] == "甲条"), None)

    ok = (same["verdict"] == "STABLE"
          and after["verdict"] == "DRIFTED"
          and after["changed"] == 1
          # the totals alone would show a core count that never moved
          and after["cores_then"] == after["cores_now"]
          and det is not None and det["replaced"] is True
          and det["lost"] and det["gained"])
    return {
        "experiment": "cross_geometry",
        "fork": "A_WHOLESALE_REPLACEMENT_IS_NOT_NO_CHANGE",
        "pass": bool(ok),
        "result": {
            "unchanged": same["verdict"],
            "after": {k: after[k] for k in
                      ("verdict", "added", "removed", "changed",
                       "cores_then", "cores_now")},
            "detail": det,
        },
    }


def not_knowing_is_not_disagreeing_fork() -> Dict[str, Any]:
    """A verifier must refuse a subject it never learned, not fail it.

    The first version of `attest_llm` returned the SAME verdict for a true
    sentence and a false one about a subject the store does not hold:

        超伝導は電気抵抗がゼロになる現象である。   true, absent
        超伝導は江戸時代の農地制度である。         false, absent

    Both scored 0.00 and came back UNSUPPORTED_BY_CORPUS, because an empty
    cross links nothing. Shipped as an MCP tool that would have read as a
    judgment on a correct sentence — the exact failure this package refuses
    everywhere else, where "no evidence" and "evidence against" are
    different typed verdicts.

    So the subject is checked first: `UNKNOWN_SUBJECT_NOT_HELD` is a refusal
    to judge, and a core too thin to judge on gets its own verdict too. The
    median core in the 626MB federation holds 11 facets and 3.5% hold fewer
    than three, which is where the floor sits — below it one facet decides.
    """
    from .attest_llm import check_all
    from .cross_store import CrossStore

    store = CrossStore()
    for s in ["第三十七条は補償である。", "第三十七条は費用である。",
              "第三十七条は請求である。", "乙条は単独である。"]:
        _ingest_ja(store, s)

    good = check_all(store, "第三十七条", "第三十七条は補償の費用を請求する。")
    bad = check_all(store, "第三十七条", "第三十七条は国家の権限を制定する。")
    absent_true = check_all(store, "超伝導", "超伝導は電気抵抗がゼロになる現象である。")
    absent_false = check_all(store, "超伝導", "超伝導は江戸時代の農地制度である。")
    thin = check_all(store, "乙条", "乙条は単独の規定である。")

    ok = (good["verdict"] == "ANSWER"
          and bad["verdict"] == "UNSUPPORTED_BY_CORPUS"
          # the two absent cases agree with each other and differ from `bad`
          and absent_true["verdict"] == "UNKNOWN_SUBJECT_NOT_HELD"
          and absent_false["verdict"] == "UNKNOWN_SUBJECT_NOT_HELD"
          and absent_true["verdict"] != bad["verdict"]
          # a refusal carries no score at all — there is nothing to score
          and "support" not in absent_true
          and thin["verdict"] == "UNKNOWN_SUBJECT_TOO_THIN")
    return {
        "experiment": "cross_geometry",
        "fork": "NOT_KNOWING_IS_NOT_DISAGREEING",
        "pass": bool(ok),
        "result": {
            "supported": good["verdict"],
            "contradicted": bad["verdict"],
            "absent_true": absent_true["verdict"],
            "absent_false": absent_false["verdict"],
            "thin": thin["verdict"],
            "refusal_carries_no_score": "support" not in absent_true,
        },
    }


def placement_is_backward_compatible_fork() -> Dict[str, Any]:
    """A store with no baked placement must answer exactly as it always did.

    Every store built before `placement` existed lacks the field, and the
    shell builder falls back to top_facets for any core it does not cover.
    This pins that the fallback is byte-identical rather than merely similar,
    because a silent shift in every historical store's answers is the kind
    of regression that only shows up as "the numbers in the README are wrong
    now" months later.
    """
    from .consensus_store import consensus_over_store

    store = CrossStore()
    for s in ["apple is red", "apple is sweet", "apple is crisp",
              "apple is bright", "apple is round", "apple is firm"]:
        store.ingest_sentence(s)

    before = consensus_over_store(store, "what is apple")
    store.placement = {}          # explicitly empty, as a loaded old store is
    empty = consensus_over_store(store, "what is apple")
    # And a placement that names a core must actually be used.
    store.placement = {"apple": ["firm", "round", "bright", "crisp"]}
    steered = consensus_over_store(store, "what is apple")

    ok = (before.get("text") == empty.get("text")
          and before.get("verdict") == "ANSWER"
          and steered.get("verdict") == "ANSWER"
          and steered.get("text") != before.get("text")
          and "firm" in str(steered.get("text", "")))
    return {
        "experiment": "cross_geometry",
        "fork": "PLACEMENT_BACKWARD_COMPATIBLE",
        "pass": bool(ok),
        "result": {
            "no_placement": before.get("text"),
            "empty_placement": empty.get("text"),
            "baked_placement": steered.get("text"),
        },
    }


def all_cross_geometry_forks() -> List[Dict[str, Any]]:
    return [
        geometric_pole_invisibility_fork(),
        geometric_rotation_reaches_all_fork(),
        layer_stack_growth_fork(),
        layered_ask_stability_fork(),
        layered_carry_drift_fork(),
        placement_granularity_fork(),
        placement_simulation_fork(),
        placement_cannot_manufacture_confidence_fork(),
        placement_invariance_fork(),
        ja_coverage_gate_fork(),
        reified_event_fork(),
        event_extractor_refuses_statute_prose_fork(),
        egov_article_is_a_citation_key_fork(),
        sovereign_build_fork(),
        one_root_saturates_at_capacity_fork(),
        fusion_is_not_monotonic_fork(),
        word_form_is_a_fallback_fork(),
        linked_is_not_printed_fork(),
        granularity_composes_fork(),
        resolution_ladder_grades_doubt_fork(),
        concord_rides_alongside_the_list_fork(),
        long_form_drifts_and_lists_fork(),
        trace_is_memory_outside_the_store_fork(),
        constellation_beats_one_sovereign_fork(),
        writer_never_reaches_the_answer_path_fork(),
        form_may_not_assert_more_than_content_licenses_fork(),
        concord_is_not_coverage_fork(),
        every_manifest_can_rebuild_its_corpus_fork(),
        a_reloaded_writer_is_the_same_writer_fork(),
        coarsening_adds_a_reading_and_never_overturns_one_fork(),
        a_citation_is_listed_not_chosen_fork(),
        a_template_cut_inside_a_word_is_caught_at_the_seam_fork(),
        presence_in_the_corpus_is_not_support_fork(),
        a_wholesale_replacement_is_not_no_change_fork(),
        not_knowing_is_not_disagreeing_fork(),
        placement_is_backward_compatible_fork(),
        promotion_pyramid_fork(),
        conversation_overflow_is_typed_fork(),
        conversation_speaker_attribution_fork(),
    ]


# ---------------------------------------------------------------------------
# Promotion — the multiresolution pyramid.
# ---------------------------------------------------------------------------

def promotion_pyramid_fork() -> Dict[str, Any]:
    """A freezing layer distils its heaviest cores into the new top layer.

    Three properties, and the third is the one that makes promotion safe:
    the summary must NOT replace the evidence. The frozen fine-grained
    original stays below, so a coarse claim can later be checked against
    the detail it came from — which is what turns a promoted node into a
    second opinion rather than a lossy overwrite.
    """
    # Five sentences, four distinct cores: apple appears twice so it carries
    # the highest mass and must be the one promoted. The fifth ingest is what
    # trips capacity — an earlier draft of this fixture used four sentences
    # and never stacked at all, because apple's two sentences are ONE core.
    _corpus = ["apple is red sweet", "apple is round",
               "banana is yellow", "cherry is small", "durian is spiky"]
    mem = LayeredMemory(capacity=3, promote_k=2)
    for s in _corpus:
        mem.ingest_sentence(s)

    promoted_top = set(mem.top.crosses.keys())
    stacked = mem.n_levels() >= 2
    # apple has the highest mass (two sentences) -> promoted.
    apple_promoted = "apple" in promoted_top
    # The original is still in the frozen layer, with MORE facets than the
    # promoted summary: evidence outlives its summary.
    fine_facets = len(mem.levels[0].top_facets("apple", k=9))
    coarse_facets = len(mem.top.top_facets("apple", k=9))
    evidence_survives = mem.levels[0].has("apple") and fine_facets >= coarse_facets

    off = LayeredMemory(capacity=3, promote_k=0)
    for s in _corpus:
        off.ingest_sentence(s)
    # With promotion off the new top layer holds ONLY what arrived after the
    # freeze — no distilled ancestors. This is the control that proves the
    # promoted nodes above came from promotion and not from ordinary ingest.
    off_is_clean = set(off.top.crosses.keys()) == {"durian"}

    ok = stacked and apple_promoted and evidence_survives and off_is_clean
    return {
        "experiment": "cross_geometry",
        "fork": "PROMOTION_PYRAMID",
        "pass": bool(ok),
        "result": {"levels": [l.n_cores() for l in mem.levels],
                   "promoted_into_top": sorted(promoted_top),
                   "fine_facets": fine_facets, "coarse_facets": coarse_facets,
                   "evidence_survives": evidence_survives,
                   "promotion_off_is_clean": off_is_clean},
    }


# ---------------------------------------------------------------------------
# Conversation context as space.
# ---------------------------------------------------------------------------

def conversation_overflow_is_typed_fork() -> Dict[str, Any]:
    """Context overflow must be FROZEN, never silence.

    An LLM whose window rolls past a turn cannot distinguish "never said"
    from "said and forgotten". Here the two are different verdicts, and an
    overflowed topic keeps its speaker and turn index. That difference IS
    the claim being tested — so the fork requires an overflowed topic to be
    FROZEN (not ABSENT), a never-mentioned one to be ABSENT, and the
    overflowed one to remain answerable from its frozen layer.
    """
    from .conversation import Conversation

    conv = Conversation(memory=LayeredMemory(capacity=2, promote_k=0))
    conv.add_turn("user", "the vault requires a hardware token")
    conv.add_turn("assistant", "the cluster runs three nodes")
    conv.add_turn("user", "the pipeline uses blue green rollout")

    vault = conv.locate("vault")
    never = conv.locate("kangaroo")
    overflowed_kept = (vault["status"] == "FROZEN"
                       and vault["mentions"] == [{"turn": 0, "speaker": "user"}])
    never_said_absent = never["status"] == "ABSENT" and never["mentions"] == []
    # Still answerable after overflow — frozen is not lost.
    ans = conv.recall("what is vault")
    still_answerable = ans.get("verdict") == "ANSWER" and ans.get("core") == "vault"

    ok = overflowed_kept and never_said_absent and still_answerable
    return {
        "experiment": "cross_geometry",
        "fork": "CONVERSATION_OVERFLOW_TYPED",
        "pass": bool(ok),
        "result": {"vault": vault, "kangaroo_status": never["status"],
                   "recall_after_overflow": {"verdict": ans.get("verdict"),
                                             "core": ans.get("core")},
                   "stats": conv.stats()},
    }


def conversation_speaker_attribution_fork() -> Dict[str, Any]:
    """Who said it, and when, survives ingestion. Without attribution a
    conversation store is a pile of assertions with no way to tell a user's
    claim from the assistant's own earlier guess — the shape of failure that
    makes a system quote itself back as evidence."""
    from .conversation import Conversation

    conv = Conversation(memory=LayeredMemory(capacity=64))
    conv.add_turn("user", "the ledger is append only")
    conv.add_turn("assistant", "the ledger is signed hourly")
    loc = conv.locate("ledger")
    speakers = [m["speaker"] for m in loc["mentions"]]
    ok = (loc["status"] == "ACTIVE" and speakers == ["user", "assistant"]
          and [m["turn"] for m in loc["mentions"]] == [0, 1])
    return {
        "experiment": "cross_geometry",
        "fork": "CONVERSATION_SPEAKER_ATTRIBUTION",
        "pass": bool(ok),
        "result": {"mentions": loc["mentions"], "status": loc["status"]},
    }
