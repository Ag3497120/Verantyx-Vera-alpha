"""Facet co-occurrence sense clustering — 「軸ずらしでニュアンス選択」の実体.

混合した core の十字（例: sun = 天文 + 企業 + 新聞）を、facet 同士の
類縁で語義クラスタに分割し、クエリの指定語（astronomy / sky / software…）
でクラスタを選択して読み出す。

類縁のオラクルはストア自身: facet f の「世界」= f 自身の十字の上位語。
  cross_words(f) = {f} ∪ top_facets(f)
  f ~ g  ⟺  g ∈ cross_words(f) ∨ f ∈ cross_words(g)
             ∨ |cross_words(f) ∩ cross_words(g)| ≥ min_shared
連結成分 = 語義クラスタ（決定論: count 降順 → アルファベット順）。

新規データを投入しても追加の格納は不要 — クラスタはクエリ時にストアから
再計算されるため、pour パイプラインはそのまま恩恵を受ける。
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Set, Tuple

from .cross_store import CrossStore
from .lex_filters import norm_words


def cross_words(
    store: CrossStore, sym: str, *, top: int = 12, max_mass: float = 800.0
) -> Set[str]:
    """f 自身の十字の上位語 + f の構成語 (十字が無ければ {f} のみ).

    汎用高頻度語 (mass > max_mass: "new", "open", "full"…) は世界から除外 —
    それ経由の類縁は語義ではなくコーパス頻度の産物なので (過連結の原因)。
    """
    out: Set[str] = set(norm_words(sym))
    for key in (sym, sym + "#p"):
        for f, _cnt in store.top_facets(key, top):
            for w in norm_words(f):
                if store.mass(w) + store.mass(w + "#p") <= max_mass:
                    out.add(w)
    return out


def facet_clusters(
    store: CrossStore,
    core: str,
    *,
    top: int = 20,
    min_shared: int = 3,
    max_mass: float = 800.0,
) -> List[List[str]]:
    """core の上位 facets を語義クラスタへ分割 (連結成分・決定論)."""
    facets = [f for f, _ in store.top_facets(core, top)]
    if not facets:
        return []
    words = {f: cross_words(store, f, max_mass=max_mass) for f in facets}

    def specific(sym: str) -> Set[str]:
        # containment 判定にも汎用語遮断を適用
        return {
            w
            for w in norm_words(sym)
            if store.mass(w) + store.mass(w + "#p") <= max_mass
        }

    parent: Dict[str, str] = {f: f for f in facets}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            # deterministic root: lexicographically smaller wins
            if rb < ra:
                ra, rb = rb, ra
            parent[rb] = ra

    for i, f in enumerate(facets):
        for g in facets[i + 1:]:
            linked = (
                bool(specific(g) & words[f])
                or bool(specific(f) & words[g])
                or len(words[f] & words[g]) >= min_shared
            )
            if linked:
                union(f, g)

    groups: Dict[str, List[str]] = {}
    for f in facets:
        groups.setdefault(find(f), []).append(f)

    counts = dict(store.top_facets(core, top))

    def cluster_weight(members: List[str]) -> Tuple[int, str]:
        return (-sum(counts.get(m, 0) for m in members), members[0])

    clusters = sorted(groups.values(), key=cluster_weight)
    for c in clusters:
        c.sort(key=lambda m: (-counts.get(m, 0), m))
    return clusters


def select_cluster(
    store: CrossStore,
    clusters: List[List[str]],
    specifiers: Set[str],
) -> Optional[int]:
    """指定語で語義クラスタを選ぶ。

    スコア = メンバー直接一致×3 + メンバーの世界 (cross_words) 一致×1。
    正のスコアが無ければ None (指定が効かない → 既定順)。
    """
    if not specifiers or not clusters:
        return None
    best_i: Optional[int] = None
    best_score = 0
    for i, members in enumerate(clusters):
        score = 0
        for m in members:
            if norm_words(m) & specifiers:
                score += 3
            if cross_words(store, m) & specifiers:
                score += 1
        if score > best_score:
            best_score = score
            best_i = i
    return best_i


def sense_facets(
    store: CrossStore,
    core: str,
    specifiers: Set[str],
    *,
    k: int = 4,
    top: int = 20,
    min_shared: int = 2,
) -> Dict[str, Any]:
    """指定語つき読み出し: 選ばれたクラスタの facets を count 順で返す."""
    clusters = facet_clusters(store, core, top=top, min_shared=min_shared)
    idx = select_cluster(store, clusters, specifiers)
    if idx is None:
        return {
            "selected": None,
            "clusters": clusters,
            "facets": [f for f, _ in store.top_facets(core, k)],
        }
    return {
        "selected": idx,
        "clusters": clusters,
        "facets": clusters[idx][:k],
    }
