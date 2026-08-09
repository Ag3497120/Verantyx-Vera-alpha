"""Retrieval → consensus bridge: CrossStore の候補十字を殻に載せて合意探索.

クエリ分解 → 候補 core を k 個引く → 各 core の腕 (tip + 上位 facets) を
殻に載せる → run_consensus / matryoshka_consensus。未ヒットは型付き
UNKNOWN のまま (捏造しない)。

候補ランキング (決定論):
  1. クエリ内容語と core の一致 (head 一致を最優先)
  2. facet がクエリ語に重なる core
  3. 同点は mass (出現回数) 降順 → アルファベット順
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Set, Tuple

from .consensus import (
    ConsensusConfig,
    matryoshka_consensus,
    query_content,
    run_consensus,
)
from .cross import AXES, ShellCross
from .cross_store import CrossStore
from .en_decompose import decompose
from .face_roles import FACET_FACES
from .lex_filters import PROPER_SUFFIX, display_sym

MAX_ARMS = len(AXES)

# "contest:2026h1" (structured key:value core) — the generic tokenizer
# strips ":" before this ever reaches decompose(), which is why a hand-
# structured record couldn't be found by a natural-language question even
# though recall(exact_name) worked. Scanned against the RAW query text,
# before tokenization, so the colon survives.
_NAMESPACED_ID = re.compile(r"[a-z0-9_]+(?::[a-z0-9_]+)+", re.I)


class _MassView:
    """masses adapter for consensus.axis_energy (log-damped counts)."""

    def __init__(self, store: CrossStore):
        self._store = store

    def get(self, sym: str) -> float:
        import math

        return 1.0 + math.log1p(self._store.mass(sym))


def candidates_for_query(
    store: CrossStore,
    query: str,
    *,
    k: int = MAX_ARMS,
) -> List[str]:
    qset, head = query_content(query)
    rec = decompose(query)
    ordered = list(rec.content_tokens)
    # who/where → 固有名チャネル優先; what/define → common 優先
    proper_first = rec.pattern in ("who_is", "where_is")

    def variants(tok: str) -> List[str]:
        both = [tok + PROPER_SUFFIX, tok] if proper_first else [tok, tok + PROPER_SUFFIX]
        return [v for v in both if store.has(v)]

    scored: List[Tuple[int, float, str]] = []
    seen: set = set()
    # 名前空間付きID ("contest:2026h1") をクエリの生テキストから直接拾う —
    # 最優先 (rank -1): これは曖昧さのない exact レコード参照なので、
    # 語のオーバーラップよりも先に候補にする
    for m in _NAMESPACED_ID.finditer(query or ""):
        key = m.group(0).casefold()
        for v in (key, key + PROPER_SUFFIX):
            if store.has(v) and v not in seen:
                seen.add(v)
                scored.append((-1, -store.mass(v), v))
    # 隣接内容語の複合 ("sun tzu" → sun_tzu#p) を最優先で引く
    for i in range(len(ordered) - 1):
        joined = f"{ordered[i]}_{ordered[i + 1]}"
        for v in (joined + PROPER_SUFFIX, joined):
            if store.has(v) and v not in seen:
                seen.add(v)
                scored.append((0, -store.mass(v), v))
    # head が store に居るなら、非 head の単独語は core 候補にしない —
    # それらは語義選択の指定語 ("sun newspaper" の newspaper)。巨大質量の
    # 汎用 core が head を押し出す事故を防ぐ。
    head_in_store = bool(head and variants(head))
    for tok in ordered:
        if head_in_store and tok != head:
            continue
        for rank, v in enumerate(variants(tok)):
            if v in seen:
                continue
            seen.add(v)
            pri = 1 if tok == head else 2
            scored.append((pri + rank, -store.mass(v), v))
    # facet 重なりで引く — 直接ヒットが1つも無い場合のみの補完。
    # (直接 core があるのに高頻度 core を facet 経由で足すと、質量で
    #  本命を押し出す — tokyo vs film の教訓)
    if not seen and qset:
        for core, cross in store.crosses.items():
            if core in seen:
                continue
            overlap = len(qset & set(cross))
            if overlap > 0:
                scored.append((9, -(overlap * 1000 + store.mass(core)), core))
    scored.sort()
    out = [c for _, _, c in scored]

    # 包摂: クエリの隣接2語を覆う複合語があれば、その部分語候補を落とす
    #   ("sun_tzu#p" が居るとき "sun" / "sun#p" / "tzu" は退場)
    from .lex_filters import norm_words

    compounds = [c for c in out if len(norm_words(c)) >= 2 and norm_words(c) <= qset]
    covered: set = set()
    for c in compounds:
        covered |= norm_words(c)
    if compounds:
        out = [
            c
            for c in out
            if len(norm_words(c)) >= 2 or not (norm_words(c) <= covered)
        ]

    # 同名双子の選別: 同じ単語に common と proper 両チャネルがあるとき、
    # what/define は common、who/where は proper を残す (片方しか無ければ維持)
    bases: Dict[str, set] = {}
    for c in out:
        if len(norm_words(c)) == 1:
            base = next(iter(norm_words(c)))
            bases.setdefault(base, set()).add(c)
    drop: set = set()
    for base, group in bases.items():
        common = base in group
        proper = (base + PROPER_SUFFIX) in group
        if common and proper:
            drop.add(base if proper_first else base + PROPER_SUFFIX)
    out = [c for c in out if c not in drop]
    return out[:k]


def build_shell_from_store(
    store: CrossStore,
    cores: List[str],
    *,
    facets_per_arm: int = len(FACET_FACES),
    tie: str = "asc",
) -> ShellCross:
    """Place each core's facts on its arm's faces.

    ``tie`` decides only how EQUALLY-SCORED facts are ordered. "asc" is the
    historical rule (lexicographic ascending, what `top_facets` does) and is
    the default, so nothing changes unless a caller asks. "desc" reverses
    each run of ties and leaves genuinely different scores where they are.

    That distinction is the whole point. Among facts the store cannot tell
    apart, the order is arbitrary — so an answer that depends on which
    arbitrary order was chosen is an artifact of the placement rather than
    of the evidence. `placement_invariant` in `consensus_over_store` uses
    this to refuse such answers. See docs/PLACEMENT.md.
    """
    shell = ShellCross()
    placement = getattr(store, "placement", None) or {}
    for axis, core in zip(AXES, cores[:MAX_ARMS]):
        shell.faces[axis]["tip"] = core
        shell.reflections[axis] = core
        # A baked placement, if this store has one for this core; otherwise
        # the historical frequency rule. Which four facets occupy the faces
        # was measured to change the answer text on 120 of 120 real queries,
        # so this is the choice, not a detail — see verantyx/placement.py.
        picks = placement.get(core)
        if picks is None:
            picks = _ranked_facets(store, core, tie=tie)
        elif tie == "desc":
            # A baked placement is already ordered; the only arbitrary part
            # left is which of the equally-scored tail entries made the cut,
            # so reverse the whole list rather than pretending to know which
            # of them tied. Coarser than the unbaked path, and honest about it.
            picks = list(reversed(picks))
        for face, facet in zip(FACET_FACES, picks[:facets_per_arm]):
            shell.faces[axis][face] = facet
    return shell


def _ranked_facets(store: CrossStore, core: str, *, tie: str = "asc") -> List[str]:
    """A core's facets, by count desc, with ties broken as ``tie`` says."""
    cross = store.crosses.get(str(core).casefold().strip()) or {}
    if not cross:
        return []
    # The citation is provenance, not a fact about the subject, and an arm
    # has four faces. A reified event with 主体/対象/行為/原因 needs all four
    # and got three because "(reported by X)" had taken one. It stays in the
    # store; it just does not occupy a face.
    labels = {str(s).casefold().strip() for s in getattr(store, "source_labels", ()) or ()}
    if labels:
        cross = {f: c for f, c in cross.items() if f not in labels}
        if not cross:
            return []
    if tie == "asc":
        return [f for f, _c in sorted(cross.items(), key=lambda kv: (-kv[1], kv[0]))]
    if tie != "desc":
        raise ValueError(f"tie must be 'asc' or 'desc', got {tie!r}")
    items = sorted(cross.items(), key=lambda kv: (-kv[1], kv[0]))
    out: List[str] = []
    i = 0
    while i < len(items):
        j = i
        while j < len(items) and items[j][1] == items[i][1]:
            j += 1
        out += [f for f, _c in sorted(items[i:j], key=lambda kv: kv[0], reverse=True)]
        i = j
    return out


def consensus_over_store(
    store: CrossStore,
    query: str,
    *,
    k: int = MAX_ARMS,
    cfg: Optional[ConsensusConfig] = None,
    matryoshka: bool = False,
    carry: str = "A",
    n_layers: int = 3,
    placement_invariant: bool = False,
) -> Dict[str, Any]:
    """End-to-end: retrieve → shell → consensus (typed verdicts).

    ``placement_invariant`` re-asks with the arbitrary part of the placement
    reversed and downgrades any ANSWER the two readings disagree about. Off
    by default: it trades recall for calibration, and that is the caller's
    decision. See `_apply_placement_invariance`.
    """
    cores = candidates_for_query(store, query, k=k)
    if not cores:
        return {
            "verdict": "UNKNOWN_NO_EVIDENCE",
            "core": None,
            "text": "",
            "retrieved": [],
            "reason": "no_candidate_cross",
        }
    shell = build_shell_from_store(store, cores)
    masses = _MassView(store)
    if matryoshka:
        out = matryoshka_consensus(
            shell, query, carry=carry, n_layers=n_layers, masses=masses
        )
    else:
        out = run_consensus(shell, query, cfg=cfg, masses=masses).as_dict()
    out["retrieved"] = cores
    out["core_key"] = out.get("core")
    if out.get("core"):
        out["core"] = display_sym(out["core"])
    if out.get("text"):
        out["text"] = " ".join(
            display_sym(t) for t in out["text"].split()
        )
    _apply_sense_selection(store, out, query)
    _apply_coverage_gate(out, query)
    # Polarity gate — inert by construction on stores that never ingested
    # polar keyed facets (every store before polarity.ingest_polar existed),
    # so wiring it unconditionally preserves historical behaviour while any
    # store that DOES carry poles gets contradiction honesty for free.
    from .polarity import apply_polarity_gate
    apply_polarity_gate(store, out, query)
    if placement_invariant:
        _apply_placement_invariance(store, out, query, k=k, cfg=cfg)
    return out


def _apply_placement_invariance(
    store: CrossStore,
    out: Dict[str, Any],
    query: str,
    *,
    k: int,
    cfg: Optional[ConsensusConfig],
    ja: bool = False,
) -> None:
    """Refuse an ANSWER that a different arbitrary placement would not give.

    **Placement cannot add information.** The store holds the same facts
    either way; all that changes is which four occupy the faces and in what
    order. So a core that wins under one arbitrary tie-break and loses under
    another won on the tie-break, not on the evidence.

    This is the same argument shape as "layout cannot add information" in
    docs/METAMORPHIC.md, and it has the same property that makes it worth
    having: it needs no answer key, no human and no model. Both readings run
    in this process against this store.

    Measured on a planted-answer store, 68 held-out descriptive questions:

        frequency placement     manufacture rate 30.9% -> 0.0%  (justified 1 -> 0)
        simulated placement     manufacture rate 13.2% -> 7.4%  (justified 3 -> 1)

    The asymmetry is the useful part. Every facet in that store had count 1,
    so the frequency rule's whole ordering WAS a tie-break and none of its
    answers survived — correctly. Demand-scored facets are not tied, so they
    survive, and simulated placement keeps some of its answers.

    It costs recall, and the cost is real: justified answers fall too. This
    is a dial, and it is off by default, because a caller who has not decided
    that refusal beats a wrong answer should not have it decided for them.
    """
    if out.get("verdict") != "ANSWER":
        return
    cores = out.get("retrieved") or []
    if not cores:
        return
    shell = build_shell_from_store(store, list(cores), tie="desc")
    if ja:
        from .lang import ja_content_runs
        alt = run_consensus(shell, query, cfg=cfg, masses=_MassView(store),
                            qset_override=set(ja_content_runs(query))).as_dict()
        alt_core = alt.get("core")
    else:
        alt = run_consensus(shell, query, cfg=cfg, masses=_MassView(store)).as_dict()
        alt_core = display_sym(alt["core"]) if alt.get("core") else None
    if alt.get("verdict") == "ANSWER" and alt_core == out.get("core"):
        out["placement_invariant"] = True
        return
    out["placement_invariant"] = False
    out["verdict"] = "AMBIGUOUS"
    out["reason"] = "placement_dependent_answer"
    out["placement_alternative"] = {"verdict": alt.get("verdict"), "core": alt_core}
    out["core"] = None
    out["text"] = ""


def _apply_sense_selection(
    store: CrossStore, out: Dict[str, Any], query: str
) -> None:
    """指定語 (core 以外の内容語) があれば語義クラスタで読み出しを差し替え."""
    if out.get("verdict") != "ANSWER" or not out.get("core_key"):
        return
    from .sense_split import sense_facets

    qset, _head = query_content(query)
    core_key = out["core_key"]
    from .lex_filters import norm_words

    specifiers = qset - norm_words(core_key)
    if not specifiers:
        return
    sel = sense_facets(store, core_key, specifiers)
    if sel["selected"] is None:
        return
    facets = sel["facets"]
    out["sense_cluster"] = {
        "selected": sel["selected"],
        "n_clusters": len(sel["clusters"]),
        "members": sel["clusters"][sel["selected"]],
    }
    out["text"] = " ".join(
        [display_sym(core_key)] + [display_sym(f) for f in facets]
    )


def _apply_coverage_gate(out: Dict[str, Any], query: str) -> None:
    """部分接地の検査: 勝ち腕が覆わないクエリ内容語を報告し、
    未被覆語が core 語と原文で隣接する場合（複合の意図: "quantum
    chromodynamics"）は、その複合を知らないので ANSWER を降格する."""
    if out.get("verdict") != "ANSWER" or not out.get("core"):
        return
    from .en_decompose import tokenize
    from .lex_filters import norm_words

    qset, _head = query_content(query)
    # namespaced core ("contest:2026h1") の構成語をここでも展開する —
    # そうしないと自分自身の core 名の一部が「未被覆」と誤報告される
    covered: Set[str] = set()
    for tok in out["core"].split():
        covered |= norm_words(tok)
    for tok in out.get("text", "").split():
        covered |= norm_words(tok)
    uncovered = sorted(qset - covered)
    out["uncovered_terms"] = uncovered
    if not uncovered:
        return
    raw_toks = tokenize(query)
    core_words: Set[str] = set()
    for tok in out["core"].split():
        core_words |= norm_words(tok)
    for i, t in enumerate(raw_toks):
        if t in core_words:
            neighbors = {
                raw_toks[j] for j in (i - 1, i + 1) if 0 <= j < len(raw_toks)
            }
            if neighbors & set(uncovered):
                out["verdict"] = "UNKNOWN_INSUFFICIENT_EVIDENCE"
                out["reason"] = f"compound_not_known:{'_'.join(sorted(neighbors & set(uncovered)))}"
                out["core"] = None
                out["text"] = ""
                return


def ja_consensus_ask(
    store: CrossStore,
    query: str,
    *,
    k: int = MAX_ARMS,
    cfg: Optional[ConsensusConfig] = None,
    placement_invariant: bool = False,
) -> Dict[str, Any]:
    """日本語の合意分解経路: 文字種 run → qset → 多断面合意 (ゲート共通).

    英語と同じ三段ゲート (合意・根拠・接地) が run ベースの qset で動く。
    候補ゼロは型付き拒否。文書は選ばれた腕の core+facets を「は/、」で連結。
    """
    from .lang import ja_content_runs

    runs = ja_content_runs(query)
    if not runs:
        return {"verdict": "UNKNOWN_UNPARSED", "lang": "ja"}
    qset = set(runs)
    cores: List[str] = []
    for r in runs:
        for v in (r, r + PROPER_SUFFIX):
            if store.has(v) and v not in cores:
                cores.append(v)
    if not cores:
        return {
            "verdict": "UNKNOWN_NO_EVIDENCE",
            "lang": "ja",
            "queried": runs,
            "retrieved": [],
        }
    shell = build_shell_from_store(store, cores[:k])
    out = run_consensus(
        shell, query, cfg=cfg, masses=_MassView(store), qset_override=qset
    ).as_dict()
    out["lang"] = "ja"
    out["retrieved"] = cores[:k]
    if out.get("verdict") == "ANSWER" and out.get("core"):
        core = out["core"]
        facets = [t for t in out.get("tokens", []) if t != core]
        out["text"] = core + ("は" + "、".join(facets) if facets else "")
    _apply_ja_coverage_gate(store, out, runs)
    if placement_invariant:
        _apply_placement_invariance(store, out, query, k=k, cfg=cfg, ja=True)
    return out


def _apply_ja_coverage_gate(
    store: CrossStore, out: Dict[str, Any], runs: List[str],
) -> None:
    """The Japanese path must not answer a question it did not address.

    This gate did not exist. `consensus_over_store` ran three — sense
    selection, coverage, polarity — and `ja_consensus_ask` ran none, so on
    the language this engine was built for, a question naming two parties
    came back as a confident ANSWER about one of them:

        「甲は乙を脅迫した。乙は丙を傷害した。」
        ask 「甲 丙」  ->  ANSWER  「甲は主犯、乙、脅迫」

    Nothing in that answer is false, and it is still a fabrication: 丙 was
    asked about and silently dropped. For a system whose entire claim is
    that it refuses rather than guesses, an open channel like this on the
    primary language is the defect that matters most.

    The downgrade fires only for a term the store KNOWS — as a core or as
    anyone's facet. A term it has never seen is a vocabulary gap, which is
    reported and left to the approval queue rather than treated as a reading
    failure; that division is the same one the rest of the package makes.
    """
    if out.get("verdict") != "ANSWER":
        return
    covered = {out.get("core") or ""}
    covered |= {t for t in (out.get("tokens") or [])}
    for tok in str(out.get("text", "")).replace("は", "、").split("、"):
        if tok:
            covered.add(tok)
    uncovered = [r for r in runs if r not in covered]
    if not uncovered:
        out["uncovered_terms"] = []
        return
    out["uncovered_terms"] = uncovered

    # A role-tagged token is a composite, and for composites "never seen"
    # means the OPPOSITE of what it means for a plain word. Asked 「事象二
    # 対象甲」 against a store holding 対象丙, the token 対象甲 is unseen —
    # and treating that as a vocabulary gap answers a question the store
    # positively refutes. Checked before the unknown-term branch for exactly
    # that reason. See verantyx/events.py.
    from .events import role_refutation

    from .events import split_role

    core_key = out.get("core_key") or out.get("core")
    refutations = [r for r in (role_refutation(store, core_key, t)
                               for t in uncovered) if r] if core_key else []

    # A role claim has three outcomes, never two: the store confirms it, the
    # store fills that role differently (refutation, below), or the role is
    # absent and the store does not know. The third was answering ANSWER —
    # asked 「事象2 場所東京」 against an event with no 場所, it returned the
    # whole event and said nothing about where. That is the same shape as the
    # 甲/丙 fabrication this gate exists to stop, one level down.
    unfilled = [t for t in uncovered if split_role(t)[0] is not None
                and not any(r["asked"] == split_role(t)[1]
                            and r["role"] == split_role(t)[0] for r in refutations)]
    if unfilled and not refutations:
        out["verdict"] = "UNKNOWN_INSUFFICIENT_EVIDENCE"
        out["reason"] = "role_not_recorded:" + ",".join(unfilled)
        out["unfilled_roles"] = unfilled
        out["core"] = None
        out["text"] = ""
        return

    if refutations:
        out["verdict"] = "UNKNOWN_ROLE_MISMATCH"
        out["reason"] = "role_filled_differently:" + ",".join(
            f"{r['role']}={'/'.join(r['held'])}≠{r['asked']}" for r in refutations)
        out["role_refutations"] = refutations
        out["core"] = None
        out["text"] = ""
        return

    known = [r for r in uncovered
             if store.has(r) or any(r in c for c in store.crosses.values())]
    if not known:
        out["unknown_terms"] = uncovered
        return
    out["verdict"] = "UNKNOWN_INSUFFICIENT_EVIDENCE"
    out["reason"] = "query_terms_not_addressed:" + "_".join(known)
    out["core"] = None
    out["text"] = ""


def probe_coverage(
    store: CrossStore,
    queries: List[str],
    *,
    k: int = MAX_ARMS,
) -> Dict[str, Any]:
    """クエリ集合に対する verdict 内訳 (coverage 測定)."""
    breakdown: Dict[str, int] = {}
    answers: List[Dict[str, Any]] = []
    for q in queries:
        out = consensus_over_store(store, q, k=k)
        v = out["verdict"]
        breakdown[v] = breakdown.get(v, 0) + 1
        answers.append(
            {"query": q, "verdict": v, "core": out.get("core"), "text": out.get("text", "")}
        )
    n = max(1, len(queries))
    return {
        "n_queries": len(queries),
        "breakdown": breakdown,
        "answer_rate": round(breakdown.get("ANSWER", 0) / n, 4),
        "samples": answers,
    }
