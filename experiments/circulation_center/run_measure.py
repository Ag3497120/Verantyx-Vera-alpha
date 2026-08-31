# -*- coding: utf-8 -*-
"""巡回の到達と中心の占有 — PREREG.md の M1/M2。

治具は測るものと同じ経路で作る: ingest_sentence / consensus_over_store /
ja_consensus_ask。M1 は配線前の死線確認、M2 は配線後の到達・無害・近道。
どちらのモードで走ったかはコードの現状が決める(種が届く経路が在るか)
ので、スクリプトは同一で、観測される事実だけを記録する。
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from verantyx.consensus_store import (build_shell_from_store,
                                      consensus_over_store, ja_consensus_ask)
from verantyx.cross_store import CrossStore


def ranked_store() -> CrossStore:
    """counts が割れる店 — locks が立ち得る形(axis_lock_fork と同系)."""
    st = CrossStore()
    for _ in range(5):
        st.ingest_sentence("gadget is heavy")
    for _ in range(4):
        st.ingest_sentence("gadget is fast")
    for _ in range(3):
        st.ingest_sentence("gadget is small")
    for _ in range(2):
        st.ingest_sentence("gadget is quiet")
    for _ in range(5):
        st.ingest_sentence("widget is bright")
    for _ in range(4):
        st.ingest_sentence("widget is loud")
    for _ in range(3):
        st.ingest_sentence("widget is cheap")
    return st


def tied_store() -> CrossStore:
    """全 facet 同点の店 — locks は立たず、種も書かれてはならない."""
    st = CrossStore()
    for a in ("aspa", "aspb", "aspc", "aspd"):
        st.ingest_sentence(f"gizmo has {a}")
    return st


def ja_store() -> CrossStore:
    st = CrossStore()
    for _ in range(5):
        st.ingest_sentence("過失 has 注意義務")
    for _ in range(4):
        st.ingest_sentence("過失 has 損害賠償")
    for _ in range(3):
        st.ingest_sentence("過失 has 予見可能性")
    for _ in range(2):
        st.ingest_sentence("過失 has 結果回避")
    return st


EN_PROBES = ["what is gadget heavy", "what is gadget fast",
             "what is gadget small", "what is widget bright",
             "what is widget loud", "what is widget cheap"]
JA_PROBES = ["過失とは", "過失 注意義務", "過失 損害賠償"]


def m1() -> dict:
    st = ranked_store()
    circ: dict = {}
    r1 = consensus_over_store(st, "what is gadget heavy",
                              placement_invariant=True, circulation=circ)
    shell = build_shell_from_store(
        st, ["gadget", "widget"])
    seed_key_used_by_reader = shell.center      # 読み出しの鍵
    written_keys = sorted(circ)                 # 書き込みの鍵
    reader_finds = {k: (circ.get(seed_key_used_by_reader) is not None)
                    for k in (written_keys or ["<empty>"])}
    r2 = consensus_over_store(st, "what is gadget fast",
                              placement_invariant=True, circulation=circ)
    return {
        "first_ask": {"verdict": r1.get("verdict"), "locks": r1.get("locks")},
        "circulation_written_keys": written_keys,
        "shell_center_at_read": seed_key_used_by_reader,
        "seed_reachable": any(reader_finds.values()),
        "second_ask": {"verdict": r2.get("verdict"),
                       "seeded_from": r2.get("seeded_from"),
                       "moves_used": r2.get("moves_used")},
    }


def m2() -> dict:
    out: dict = {"en": [], "ja": [], "tied": None}

    # ② 無害 + ③ 近道 (EN)
    for probes, mk in ((EN_PROBES, ranked_store),):
        circ: dict = {}
        # 1周目: 巡回に配置を書かせる
        for q in probes:
            consensus_over_store(mk(), q, placement_invariant=True,
                                 circulation=circ)
        for q in probes:
            st = mk()
            plain = consensus_over_store(st, q, placement_invariant=True)
            seeded = consensus_over_store(st, q, placement_invariant=True,
                                          circulation=circ)
            out["en"].append({
                "q": q,
                "identical": (plain.get("verdict") == seeded.get("verdict")
                              and plain.get("core") == seeded.get("core")
                              and plain.get("text") == seeded.get("text")),
                "verdict": seeded.get("verdict"),
                "seeded_from": seeded.get("seeded_from"),
                "moves_plain": plain.get("moves_used"),
                "moves_seeded": seeded.get("moves_used"),
            })

    # ⑤ ja 経路
    circ_ja: dict = {}
    for q in JA_PROBES:
        ja_consensus_ask(ja_store(), q, placement_invariant=True,
                         circulation=circ_ja)
    for q in JA_PROBES:
        st = ja_store()
        plain = ja_consensus_ask(st, q, placement_invariant=True)
        seeded = ja_consensus_ask(st, q, placement_invariant=True,
                                  circulation=circ_ja)
        out["ja"].append({
            "q": q,
            "identical": (plain.get("verdict") == seeded.get("verdict")
                          and plain.get("core") == seeded.get("core")
                          and plain.get("text") == seeded.get("text")),
            "verdict": seeded.get("verdict"),
            "seeded_from": seeded.get("seeded_from"),
            "moves_plain": plain.get("moves_used"),
            "moves_seeded": seeded.get("moves_used"),
        })

    # ③ 近道 — 移動が実際に受理される店(探索が回転/退避を使う形)で、
    # 会話扉(mcp)と同じ書き方で終端配置を巡回に書き、再訪の moves を測る
    def moving_store() -> CrossStore:
        st = CrossStore()
        words = ["alpha", "bravo", "carla", "delta", "echof", "foxtr"]
        for i, c in enumerate(words):
            for w in words:
                if w != c:
                    st.ingest_sentence(f"{c} has {c}{w}")
            for _n in range(6 - i):
                st.ingest_sentence(f"{c} has shared")
        return st

    q = "what has shared"
    st = moving_store()
    first = consensus_over_store(st, q)
    circ_m: dict = {}
    if first.get("core_key") and first.get("carry_state"):
        # mcp_server の書き方と同形(鍵は core_key、locks は合流)
        circ_m[str(first["core_key"])] = dict(first["carry_state"])
    revisit_plain = consensus_over_store(moving_store(), q)
    revisit_seed = consensus_over_store(moving_store(), q, circulation=circ_m)
    out["shortcut"] = {
        "q": q,
        "first": {"verdict": first.get("verdict"),
                  "moves": first.get("moves_used"),
                  "escape": first.get("escape_used"),
                  "carry": first.get("carry_state")},
        "revisit_plain": {"verdict": revisit_plain.get("verdict"),
                          "core": revisit_plain.get("core"),
                          "moves": revisit_plain.get("moves_used"),
                          "escape": revisit_plain.get("escape_used")},
        "revisit_seeded": {"verdict": revisit_seed.get("verdict"),
                           "core": revisit_seed.get("core"),
                           "moves": revisit_seed.get("moves_used"),
                           "escape": revisit_seed.get("escape_used"),
                           "seeded_from": revisit_seed.get("seeded_from")},
        "identical": (revisit_plain.get("verdict") == revisit_seed.get("verdict")
                      and revisit_plain.get("core") == revisit_seed.get("core")
                      and revisit_plain.get("text") == revisit_seed.get("text")),
    }

    # ④ 同点は棄権 — locks も種も書かれない
    circ_t: dict = {}
    rt = consensus_over_store(tied_store(), "what is gizmo aspa",
                              placement_invariant=True, circulation=circ_t)
    locks_written = any((v or {}).get("locks") for v in circ_t.values()
                        if isinstance(v, dict))
    rt2 = consensus_over_store(tied_store(), "what is gizmo aspb",
                               placement_invariant=True, circulation=circ_t)
    out["tied"] = {"first_verdict": rt.get("verdict"),
                   "locks_written": locks_written,
                   "second_seeded_from": rt2.get("seeded_from")}
    return out


def main() -> None:
    mode = (sys.argv[1] if len(sys.argv) > 1 else "both")
    result: dict = {}
    if mode in ("m1", "both"):
        result["M1"] = m1()
    if mode in ("m2", "both"):
        result["M2"] = m2()
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
