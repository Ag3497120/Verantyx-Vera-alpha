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
