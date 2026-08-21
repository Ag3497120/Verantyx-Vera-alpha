# -*- coding: utf-8 -*-
"""二つの顔の自己検査 — 番人(G1〜G4)と単体の装置(S1〜S4)。

事前登録: experiments/guard/PREREG5_FREEZE.md / PREREG6_STANDALONE.md

他人のマシンに入れた直後に叩くもの。**「実装されている」ではなく
「今このマシンで動いた」を見る**ので、検査は毎回その場で店と台帳を
作って実際に走らせる。利用者の店にも台帳にも触らない。

壊れた実装を注入できるようにしてあるのは、**壊れているときに
BROKEN と言えることを測るため**。通るだけの自己検査は自己申告と同じで、
嘘をつく自己検査は無いより悪い。
"""
from __future__ import annotations

import itertools
from typing import Any, Callable, Dict, List

#: 治具は小さく閉じたものにする(導入直後に叩くので速さが要る)。
#: 企業文書2件と技術方針1件 — 条番号を持つ文を入れてあるのは、
#: 答えが取り込んだ文の語だけでできていることを見るため。
_FIXTURE = [
    ("社内規程", "第3条 出張費は事前承認が必要である。"
                 "第4条 交際費は上限を月5万円とする。"),
    ("就業規則", "第7条 在宅勤務は週3日まで認める。"),
    ("技術方針", "実装言語はTypeScriptを用いる。テストはpytestで書く。"),
]
#: 在庫内の問いと、期待する核。在庫外の問いは何を返しても ANSWER で
#: あってはならない(近傍を返す装置との差はここに出る)。
_IN_STOCK = [("出張費", "出張費"), ("交際費", "交際費"),
             ("在宅勤務", "在宅勤務")]
_OUT_OF_STOCK = ["ゾルタクスゼイアン", "深海探査船の定員", "quantum flux capacitor"]


def _build_store(order: List[int], ingest: Callable, store_cls: Any):
    from .document_ingest import Document

    st = store_cls()
    ingest(st, [Document(source=_FIXTURE[i][0], text=_FIXTURE[i][1])
                for i in order])
    return st


def store_self_check(ask: Callable = None, ingest: Callable = None,
                     store_cls: Any = None) -> Dict[str, Any]:
    """単体の装置としての保証 S1〜S4 を、その場で実演して確かめる。"""
    from .cross_store import CrossStore
    from .document_ingest import ingest_documents

    ask = ask or _default_ask
    ingest = ingest or ingest_documents
    store_cls = store_cls or CrossStore

    probes: List[Dict[str, Any]] = []

    def probe(name, ok, detail):
        probes.append({"probe": name, "pass": bool(ok), "detail": detail})

    try:
        st = _build_store([0, 1, 2], ingest, store_cls)
    except Exception as e:                     # noqa: BLE001
        for n in ("S1_in_stock_answers", "S2_absence_is_refused",
                  "S3_ingest_order_invariant", "S4_answer_uses_only_stored_words"):
            probe(n, False, {"error": repr(e)})
        return _verdict(probes)

    # S1 在庫にあることは答える
    try:
        got = [(q, ask(st, q)) for q, _core in _IN_STOCK]
        ok = all(o.get("verdict") == "ANSWER" and o.get("core") == core
                 for (q, o), (_q, core) in zip(got, _IN_STOCK))
        probe("S1_in_stock_answers", ok,
              {q: [o.get("verdict"), o.get("core")] for q, o in got})
    except Exception as e:                     # noqa: BLE001
        probe("S1_in_stock_answers", False, {"error": repr(e)})

    # S2 無いことは型つきで断る(近傍を返さない)
    try:
        got = [(q, ask(st, q)) for q in _OUT_OF_STOCK]
        # 「不在」は UNKNOWN_* で、理由が付いていること。ANSWER は不可。
        ok = all(str(o.get("verdict", "")).startswith("UNKNOWN")
                 and not o.get("text") for _q, o in got)
        probe("S2_absence_is_refused", ok,
              {q: [o.get("verdict"), o.get("reason")] for q, o in got})
    except Exception as e:                     # noqa: BLE001
        probe("S2_absence_is_refused", False, {"error": repr(e)})

    # S3 取り込み順に依らない
    try:
        seen: Dict[str, set] = {q: set() for q, _c in _IN_STOCK}
        for order in itertools.permutations(range(len(_FIXTURE))):
            s2 = _build_store(list(order), ingest, store_cls)
            for q, _core in _IN_STOCK:
                o = ask(s2, q)
                seen[q].add((o.get("verdict"), o.get("core")))
        ok = all(len(v) == 1 for v in seen.values())
        probe("S3_ingest_order_invariant", ok,
              {"permutations": 6,
               "distinct_outcomes": {q: len(v) for q, v in seen.items()}})
    except Exception as e:                     # noqa: BLE001
        probe("S3_ingest_order_invariant", False, {"error": repr(e)})

    # S4 答えは店にある語だけでできている(店の外の語を持ち込まない)
    try:
        known = set(st.crosses)
        for _core, facets in st.crosses.items():
            known.update(facets)
        strangers: Dict[str, List[str]] = {}
        for q, _core in _IN_STOCK:
            o = ask(st, q)
            out = [t for t in (o.get("tokens") or str(o.get("text", "")).split())
                   if t not in known]
            if out:
                strangers[q] = out
        probe("S4_answer_uses_only_stored_words", not strangers,
              {"strangers": strangers or "none",
               "store_vocabulary": len(known)})
    except Exception as e:                     # noqa: BLE001
        probe("S4_answer_uses_only_stored_words", False, {"error": repr(e)})

    return _verdict(probes)


def _default_ask(store: Any, question: str) -> Dict[str, Any]:
    from .consensus_store import consensus_over_store

    return consensus_over_store(store, question)


def _verdict(probes: List[Dict[str, Any]]) -> Dict[str, Any]:
    failed = [p["probe"] for p in probes if not p["pass"]]
    return {"verdict": "BROKEN" if failed else "OK",
            "guarantees": probes, "failed": failed}


def full_doctor() -> Dict[str, Any]:
    """二つの顔を1回で。**片方が壊れていれば全体は BROKEN**。

    番人だけ緑で単体が壊れている(あるいはその逆)を「概ね健全」と
    まとめない — 要約が証拠を隠さない、はこの装置の線。
    """
    from .covenant import self_check

    guard = self_check()
    standalone = store_self_check()
    failed = list(guard["failed"]) + list(standalone["failed"])
    return {
        "verdict": "BROKEN" if failed else "OK",
        "guard": guard,
        "standalone": standalone,
        "failed": failed,
        "note": "guaranteed properties are re-run here and now; what is "
                "NOT guaranteed is named in experiments/guard/"
                "PREREG5_FREEZE.md (N1-N7) and PREREG6_STANDALONE.md",
    }
