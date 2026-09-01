# -*- coding: utf-8 -*-
"""PREREG2 の M3 — CLI 扉の巡回継続性を、実 CLI サブプロセスで測る。

治具は測るものと同じ経路: `python -m verantyx.cli ask` を実際に叩く。
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from verantyx.consensus_store import consensus_over_store
from verantyx.cross_store import CrossStore


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


def cli_ask(store: Path, query: str) -> str:
    r = subprocess.run(
        [sys.executable, "-m", "verantyx.cli", "--store", str(store),
         "ask", query],
        capture_output=True, text=True, cwd=str(ROOT), timeout=300)
    return r.stdout


def main() -> None:
    q = "what has shared"
    out: dict = {}
    with tempfile.TemporaryDirectory() as td:
        sp = Path(td) / "m3_store.json"
        moving_store().save(sp)
        side = sp.with_name(sp.stem + ".circulation.json")

        # 測定1 — 側車なし: 2回のバイト同一・側車が作られない
        a = cli_ask(sp, q)
        b = cli_ask(sp, q)
        no_side = {
            "deterministic": a == b,
            "sidecar_created": side.is_file(),
            "seeded_from_absent": '"seeded_from"' not in a,
        }
        ja = json.loads(a)

        # 測定2 — 側車あり(会話扉が書く形): 到達と無害
        first = consensus_over_store(moving_store(), q)
        side.write_text(json.dumps(
            {str(first["core_key"]): dict(first["carry_state"])},
            ensure_ascii=False), "utf-8")
        c = cli_ask(sp, q)
        jc = json.loads(c)
        with_side = {
            "seeded_from": jc.get("seeded_from"),
            "identical": (ja.get("verdict") == jc.get("verdict")
                          and ja.get("core") == jc.get("core")
                          and ja.get("text") == jc.get("text")),
            "moves_plain": ja.get("moves_used"),
            "moves_seeded": jc.get("moves_used"),
            "escape_plain": ja.get("escape_used"),
            "escape_seeded": jc.get("escape_used"),
        }

        # 測定3 — 書き戻し
        after = json.loads(side.read_text("utf-8"))
        key = str(first["core_key"])
        with_side["written_back"] = (key in after
                                     and isinstance(after[key], dict)
                                     and "widened" in after[key])

        out = {"no_sidecar": no_side, "with_sidecar": with_side}
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
