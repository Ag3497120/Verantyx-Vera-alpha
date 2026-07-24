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
) -> ShellCross:
    shell = ShellCross()
    for axis, core in zip(AXES, cores[:MAX_ARMS]):
        shell.faces[axis]["tip"] = core
        shell.reflections[axis] = core
        for face, (facet, _cnt) in zip(
            FACET_FACES, store.top_facets(core, k=facets_per_arm)
        ):
            shell.faces[axis][face] = facet
    return shell


def consensus_over_store(
    store: CrossStore,
    query: str,
    *,
    k: int = MAX_ARMS,
    cfg: Optional[ConsensusConfig] = None,
    matryoshka: bool = False,
    carry: str = "A",
    n_layers: int = 3,
) -> Dict[str, Any]:
    """End-to-end: retrieve → shell → consensus (typed verdicts)."""
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
    return out


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
    return out


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
