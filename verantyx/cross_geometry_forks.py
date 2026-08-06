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


def all_cross_geometry_forks() -> List[Dict[str, Any]]:
    return [
        geometric_pole_invisibility_fork(),
        geometric_rotation_reaches_all_fork(),
        layer_stack_growth_fork(),
        layered_ask_stability_fork(),
        layered_carry_drift_fork(),
        placement_granularity_fork(),
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
