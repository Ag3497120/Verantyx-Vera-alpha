"""Nodes that contain nodes — the tree the geometry forces you to build.

A stereo cross has six arms and four fact faces per arm. That is not a
styling choice; it is a hard ceiling on how much one node can tell apart.
Measured on six Japanese statutes ingested as separate domains:

    terms placed on the faces        20 of 24 routed to the right domain
    terms NOT placed on the faces     0 of 60

A node routes on what fits its faces and on nothing else. Pushing more in
does not help — it hurts, because the four that get placed are then chosen
by an arbitrary tie-break:

    facets per arm      4     8    16    32    60
    routing correct   4/8   3/8   1/8   0/8   0/8

So a node's routing vocabulary is exactly

    CAPACITY = MAX_ARMS x N_FACES = 6 x 4 = 24

and a domain with more distinctions than that cannot be served by one node,
whatever you do to it. The only remedy is another layer. Depth is therefore
not a design preference:

    depth ~= log6(V / N_FACES)      V = terms that must be routable

which also gives the exponential fan-out — layer n holds 6^n nodes. For the
six-statute store (2,208 cores): 552 arms -> 92 nodes -> 16 -> 3 -> 1, so
depth 4. Ten layers would address 6^10 leaves.

## Why the leaves must stay separate

The same six statutes measured flat: 184 terms appear in three or more of
them, and 行為 in 173 articles across all six. 民法's 法律行為 is not 刑法's
行為. A single flat store merges them silently. Keeping domains in their own
stores and connecting them only through routers is what makes 正当防衛
retrievable as BOTH 刑法第三十六条 and 民法第七百二十条 — two answers and a
refusal to choose, rather than one confident wrong one.

## What descent may never do

`route` returns a typed refusal when no branch is reachable, and `descend`
stops there. A router that guesses a branch is worse than no router: the
question arrives at a domain that cannot answer it, that domain answers
about something else, and the wrong answer now carries a routing path that
looks like justification.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from .consensus_store import MAX_ARMS
from .cross_store import CrossStore
from .face_roles import FACET_FACES

N_FACES = len(FACET_FACES)

#: Terms one node can route on. Measured, not chosen: see the module
#: docstring. Exceeding it does not degrade gracefully — it goes to zero.
CAPACITY = MAX_ARMS * N_FACES

#: Terms shared by more than this many sibling domains are not evidence of
#: any one of them. 行為 sits in all six statutes; routing on it would send
#: every question to whichever arm sorted first.
MAX_SIBLING_SHARE = 2


@dataclass
class Node:
    """A domain node: either a leaf holding claims, or a router over children."""

    name: str
    store: Optional[CrossStore] = None
    children: Dict[str, "Node"] = field(default_factory=dict)
    #: Arms = child names, facets = terms that distinguish that child.
    router: Optional[CrossStore] = None

    @property
    def is_leaf(self) -> bool:
        return not self.children

    def depth(self) -> int:
        return 1 if self.is_leaf else 1 + max(c.depth() for c in self.children.values())

    def leaves(self) -> List["Node"]:
        if self.is_leaf:
            return [self]
        return [lf for c in self.children.values() for lf in c.leaves()]

    def counts_by_layer(self) -> List[int]:
        """Nodes per layer, root first — the fan-out, measured on this tree."""
        out: List[int] = []
        layer = [self]
        while layer:
            out.append(len(layer))
            layer = [c for n in layer for c in n.children.values()]
        return out


def distinctive_terms(
    own: CrossStore, siblings: List[CrossStore], k: int = N_FACES,
    *, exclude: Optional[set] = None,
) -> List[str]:
    """The terms that identify ``own`` against its siblings, best first.

    Ranked by how concentrated a term is in this store rather than by how
    often it occurs: a domain's most FREQUENT word is usually one every
    sibling also uses. Capped at ``k`` because a longer list is not stored
    anywhere — the faces are the storage.

    ENTITY NAMES ARE EXCLUDED, and that is the difference between a router
    that works and one that does not. The first version ranked purely by
    concentration, and the 刑事 branch came back routing on 刑事訴訟法,
    刑法, and 刑事訴訟法第三百五十条 — a law's own name and its most-cited
    article. Those are perfectly distinctive and perfectly useless: with
    only four faces they crowded out 検察官, 被告人, 勾留, so every
    conceptual question refused at the root.

    A citation is already directly retrievable at the leaf, where the
    article IS a core. What a router needs is what the branch is ABOUT.
    """
    df: Counter = Counter()
    for st in siblings:
        df.update({t for c in st.crosses.values() for t in c})
    mine: Counter = Counter()
    for c in own.crosses.values():
        mine.update(c)
    skip = set(exclude or ())
    # Anything that is a core somewhere is an entity — an article, a law, a
    # reified event — not a description of the branch.
    for st in [own] + list(siblings):
        skip |= set(st.crosses)
        # ...and so is the name an entity is built from. 民法 is not a core
        # (民法第一条 is), so a core-set check alone left every law's own
        # name eligible, and 民法/刑法/労働基準法 took three of the four
        # faces at the root. A citation prefix is a label, not a description.
        skip |= {c.split("第", 1)[0] for c in st.crosses if "第" in c}
    ranked = sorted(
        mine,
        key=lambda t: (-mine[t] / max(1, df[t]), -mine[t], t),
    )
    # Strictly absent from every sibling. MAX_SIBLING_SHARE was written for
    # a wide fan-out; at three children "in two of them" is most of the tree,
    # which is how 法律 and 規定 became routing evidence for two branches at
    # once. Tolerance scales with how many siblings there are.
    tolerance = min(MAX_SIBLING_SHARE, max(0, (len(siblings) - 1) // 2))
    return [t for t in ranked
            if len(t) >= 2 and df[t] <= tolerance and t not in skip][:k]


def build_router(children: Dict[str, "Node"]) -> CrossStore:
    """A router store whose cores are child names and facets identify them."""
    from .document_ingest import Document, ingest_documents

    stores = {name: (n.store if n.store is not None else _merged(n))
              for name, n in children.items()}
    # The children's own names are never routing evidence: 「刑事」 reaching
    # the 刑事 branch teaches nothing a lookup would not, and it costs a face.
    names = set(children)
    lines: List[str] = []
    for name, st in stores.items():
        others = [s for n2, s in stores.items() if n2 != name]
        for t in distinctive_terms(st, others, exclude=names):
            lines.append(f"{name}は{t}である。")
    router = CrossStore()
    if lines:
        ingest_documents(router, [Document(source="router", text="".join(lines))])
    return router


def _merged(node: "Node") -> CrossStore:
    """A view of everything under a node, for computing what identifies it.

    Used only to rank router terms. It is NOT what a question is answered
    from — that would put every domain back in one flat store and undo the
    separation the tree exists for.
    """
    if node.store is not None:
        return node.store
    out = CrossStore()
    for lf in node.leaves():
        if lf.store is None:
            continue
        for core, cross in lf.store.crosses.items():
            out.crosses.setdefault(core, {}).update(cross)
            out.core_count[core] = out.core_count.get(core, 0) + 1
    return out


def build(name: str, domains: Dict[str, CrossStore]) -> Node:
    """One router over leaf domains. The unit the tree is grown from."""
    children = {n: Node(name=n, store=st) for n, st in domains.items()}
    node = Node(name=name, children=children)
    node.router = build_router(children)
    return node


def federate(name: str, nodes: Dict[str, Node]) -> Node:
    """Bind several domain trees under one node — the sovereign layer.

    Called again on its own output to add a layer, which is how the design
    grows: connect to the top node until it is full, then build above it.
    "Full" is not a feeling; it is CAPACITY, and `over_capacity` says so.
    """
    node = Node(name=name, children=dict(nodes))
    node.router = build_router(node.children)
    return node


def over_capacity(node: Node) -> Dict[str, Any]:
    """Does this node need a layer inserted beneath it?

    Two ways to exceed the ceiling, and they are different failures. Too
    many children means arms are dropped outright. Too many distinguishing
    terms means the arms exist but route on an arbitrary four.
    """
    routable = MAX_ARMS * N_FACES
    n_children = len(node.children)
    needed = 0
    for child in node.children.values():
        st = child.store if child.store is not None else _merged(child)
        needed += len({t for c in st.crosses.values() for t in c})
    return {
        "node": node.name,
        "children": n_children,
        "arms_available": MAX_ARMS,
        "routable_terms": routable,
        "terms_beneath": needed,
        "over": n_children > MAX_ARMS or needed > routable,
        "reason": ("children exceed arms" if n_children > MAX_ARMS
                   else "terms beneath exceed routable capacity"
                   if needed > routable else "within capacity"),
        "suggested_extra_layers": max(0, _layers_for(needed) - 1),
    }


def _layers_for(v: int) -> int:
    """How many layers a vocabulary of ``v`` terms needs."""
    import math

    if v <= 0:
        return 1
    return max(1, math.ceil(math.log(max(1.0, v / N_FACES), MAX_ARMS)))


def route(node: Node, query: str) -> Dict[str, Any]:
    """Which child should answer, or a typed refusal.

    A refusal here is the correct outcome for a question this branch cannot
    serve, and it must never be replaced by a guess: a wrong branch produces
    an answer about something else that arrives carrying a routing path
    which reads like justification.
    """
    from .consensus_store import ja_consensus_ask

    if node.is_leaf:
        return {"verdict": "LEAF", "node": node.name}
    if node.router is None:
        return {"verdict": "UNKNOWN_NO_ROUTER", "node": node.name}
    out = ja_consensus_ask(node.router, query)
    if out.get("verdict") == "ANSWER" and out.get("core") in node.children:
        return {"verdict": "ANSWER", "child": out["core"], "node": node.name}
    return {
        "verdict": out.get("verdict", "UNKNOWN_NO_EVIDENCE"),
        "node": node.name,
        "candidates": [c for c in (out.get("retrieved") or [])
                       if c in node.children],
        "reason": "no_branch_reachable",
    }


def descend(root: Node, query: str, *, max_depth: int = 16) -> Dict[str, Any]:
    """Walk from the root to a leaf, then answer there. Stops at a refusal."""
    from .consensus_store import ja_consensus_ask

    path: List[str] = [root.name]
    node = root
    for _ in range(max_depth):
        if node.is_leaf:
            break
        step = route(node, query)
        if step["verdict"] != "ANSWER":
            return {"verdict": step["verdict"], "path": path,
                    "stopped_at": node.name,
                    "candidates": step.get("candidates", []),
                    "reason": step.get("reason", "")}
        node = node.children[step["child"]]
        path.append(node.name)
    if node.store is None:
        return {"verdict": "UNKNOWN_EMPTY_LEAF", "path": path}
    out = ja_consensus_ask(node.store, query)
    out["path"] = path
    return out


def shape(root: Node) -> Dict[str, Any]:
    """The tree's measured shape — what a simulator or a report needs."""
    return {
        "root": root.name,
        "depth": root.depth(),
        "nodes_per_layer": root.counts_by_layer(),
        "leaves": len(root.leaves()),
        "capacity_per_node": CAPACITY,
        "arms": MAX_ARMS,
        "faces": N_FACES,
    }
