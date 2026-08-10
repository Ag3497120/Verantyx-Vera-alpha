"""The federation as one SQLite file — the form it can actually be published in.

`build_ja` writes `federation.pkl`, and pickle is the wrong thing to hand to
anyone. `pickle.loads` executes whatever the file says to execute, so a
published .pkl is a published code-execution primitive; that is why Hugging
Face scans for them and warns. It is also opaque — a reader who wants to
check a number in the model card cannot look inside it without running the
project's own code, which is the same circularity this project keeps trying
to remove from its measurements.

SQLite fixes both. The file loads without executing anything, and anyone can
audit the claims directly:

    sqlite3 vera.db "SELECT count(*) FROM cores"
    sqlite3 vera.db "SELECT facet, count FROM facets
                     WHERE core='正当防衛' ORDER BY count DESC LIMIT 5"

## One file, not 6,037

`CrossStore.save()` already speaks SQLite, but a federation is a dict of
domain -> leaf -> store, and one file per leaf is not a distribution. The
leaf becomes a column instead, which is also what lets a reader ask which
statute division a facet came from — a question the per-store format cannot
answer at all.

## Both languages in one file, still two sovereigns

`lang` is a column, and `load` hands back one store per language exactly as
the pickle path did. Storing them together is not pooling them: nothing ever
queries across the column, because a cross-language census counts a tokenizer
collision as corroboration (see `polyglot`).

The distributed file must answer what the built one answers; `--verify`
checks that on a question bank rather than trusting the round trip.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sqlite3
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SCHEMA = """
PRAGMA journal_mode=OFF;
CREATE TABLE IF NOT EXISTS meta (k TEXT PRIMARY KEY, v TEXT);
CREATE TABLE IF NOT EXISTS leaves (
  id INTEGER PRIMARY KEY, lang TEXT NOT NULL,
  domain TEXT NOT NULL, name TEXT NOT NULL,
  merged INTEGER NOT NULL DEFAULT 0
);
CREATE TABLE IF NOT EXISTS cores (
  leaf INTEGER NOT NULL, core TEXT NOT NULL, count INTEGER NOT NULL,
  PRIMARY KEY (leaf, core)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS facets (
  leaf INTEGER NOT NULL, core TEXT NOT NULL,
  facet TEXT NOT NULL, count INTEGER NOT NULL,
  PRIMARY KEY (leaf, core, facet)
) WITHOUT ROWID;
CREATE TABLE IF NOT EXISTS labels (leaf INTEGER NOT NULL, label TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS caps (
  leaf INTEGER NOT NULL, word TEXT NOT NULL,
  cap INTEGER NOT NULL, low INTEGER NOT NULL,
  PRIMARY KEY (leaf, word)
) WITHOUT ROWID;
"""

#: Built after the bulk insert, never before — building an index per row is
#: what made the first version take minutes instead of seconds.
INDEXES = """
CREATE INDEX IF NOT EXISTS idx_cores_core ON cores(core);
CREATE INDEX IF NOT EXISTS idx_facets_core ON facets(core);
CREATE INDEX IF NOT EXISTS idx_facets_facet ON facets(facet);
CREATE INDEX IF NOT EXISTS idx_labels_leaf ON labels(leaf);
"""


def _stores(root: Path) -> List[Tuple[str, str, str, Any]]:
    """(lang, domain, leaf name, store) in the order `vera.load` merges them.

    Insertion order, NOT sorted. The merge below overwrites rather than adds,
    so the order decides the answer, and sorting the leaves here silently
    changed two of twelve bank questions the first time this was written.
    """
    out: List[Tuple[str, str, str, Any]] = []
    fed = root / "build" / "federation.pkl"
    if fed.exists():
        doms = pickle.loads(fed.read_bytes())
        for d in doms:
            for name in doms[d]:
                out.append(("ja", d, name, doms[d][name]))
    eng = root / "build" / "english.pkl"
    if eng.exists():
        obj = pickle.loads(eng.read_bytes())
        if isinstance(obj, dict):
            for name in obj:
                out.append(("en", "en", name, obj[name]))
        else:
            out.append(("en", "en", "english", obj))
    return out


def export(root: Path, out: Path) -> Dict[str, Any]:
    t0 = time.time()
    out = Path(out)
    if out.exists():
        out.unlink()
    con = sqlite3.connect(str(out))
    con.executescript(SCHEMA)

    n_cores = n_facets = 0
    by_lang: Dict[str, int] = {}
    with con:
        for i, (lang, dom, name, st) in enumerate(_stores(root)):
            # `merged` records that this leaf was one of many folded into a
            # single sovereign, which is what decides whether its core_count
            # survives the load. See `load`.
            con.execute("INSERT INTO leaves VALUES (?,?,?,?,?)",
                        (i, lang, dom, name, int(lang == "ja")))
            labels = getattr(st, "source_labels", set()) or set()
            con.executemany("INSERT INTO labels VALUES (?,?)",
                            ((i, l) for l in sorted(labels)))
            cc = getattr(st, "core_count", {}) or {}
            con.executemany(
                "INSERT OR REPLACE INTO cores VALUES (?,?,?)",
                ((i, c, int(cc.get(c, 0))) for c in st.crosses))
            rows = [(i, c, f, int(n))
                    for c, cross in st.crosses.items()
                    for f, n in cross.items()]
            con.executemany("INSERT OR REPLACE INTO facets VALUES (?,?,?,?)",
                            rows)
            caps = getattr(st, "cap_stats", {}) or {}
            con.executemany(
                "INSERT OR REPLACE INTO caps VALUES (?,?,?,?)",
                ((i, w, int(v[0]), int(v[1])) for w, v in caps.items()
                 if isinstance(v, (list, tuple)) and len(v) >= 2))
            n_cores += len(st.crosses)
            n_facets += len(rows)
            by_lang[lang] = by_lang.get(lang, 0) + 1
    con.executescript(INDEXES)
    with con:
        con.executemany("INSERT OR REPLACE INTO meta VALUES (?,?)", [
            ("format", "vera-federation-1"),
            ("built", time.strftime("%Y-%m-%dT%H:%M:%S")),
            ("leaves", str(sum(by_lang.values()))),
            ("cores", str(n_cores)),
            ("facets", str(n_facets)),
            ("languages", json.dumps(by_lang, sort_keys=True)),
        ])
    con.execute("VACUUM")
    con.close()
    return {"path": str(out), "mb": round(out.stat().st_size / 1048576, 1),
            "leaves": sum(by_lang.values()), "cores": n_cores,
            "facets": n_facets, "by_language": by_lang,
            "seconds": round(time.time() - t0, 1)}


def load(path: Path) -> Dict[str, Any]:
    """Reconstruct one CrossStore per language. No code is executed.

    Reproduces `vera.load` exactly, including the parts of it that are
    arguably wrong, because this function changes the CONTAINER and must not
    change the answers. Two behaviours are inherited on purpose:

    * merging is `dict.update`, so a facet held by two leaves keeps the LAST
      leaf's count rather than the sum. Duplicate attestation is discarded.
    * `core_count` is not merged across a federation at all, so the Japanese
      sovereign runs with an empty one.

    Both are worth revisiting; neither may be revisited here, or the file
    published as "the model" would answer differently from the model whose
    numbers the card reports.
    """
    from .cross_store import CrossStore

    con = sqlite3.connect(f"file:{Path(path)}?mode=ro", uri=True)
    leaves = list(con.execute(
        "SELECT id, lang, merged FROM leaves ORDER BY id"))
    lang_of = {i: l for i, l, _m in leaves}
    merged = {l: bool(m) for _i, l, m in leaves}
    stores: Dict[str, Any] = {l: CrossStore() for l in set(lang_of.values())}

    for leaf, label in con.execute("SELECT leaf, label FROM labels"):
        stores[lang_of[leaf]].source_labels.add(label)

    # One leaf at a time, in id order, so `update` overwrites in the same
    # sequence the pickle merge did.
    for leaf, lang, _m in leaves:
        dst = stores[lang].crosses
        # Cores FIRST, and every one of them, including the ones holding no
        # facets. Rebuilding `crosses` from the facets table alone dropped
        # 1,828 of the English sovereign's 15,268 cores — and a facet-less
        # core is not a nothing: it is the difference between
        # UNKNOWN_NOT_PRESENT (the term is held, nothing supports an answer)
        # and UNKNOWN_NO_EVIDENCE (the census found no such term). Losing it
        # silently downgrades the more precise refusal into the vaguer one.
        for (core,) in con.execute(
                "SELECT core FROM cores WHERE leaf=?", (leaf,)):
            dst.setdefault(core, {})
        cross: Dict[str, Dict[str, int]] = {}
        for core, facet, n in con.execute(
                "SELECT core, facet, count FROM facets WHERE leaf=?", (leaf,)):
            cross.setdefault(core, {})[facet] = n
        for c, cr in cross.items():
            dst.setdefault(c, {}).update(cr)

    for leaf, lang, m in leaves:
        if m:                       # merged federations carry neither
            continue
        st = stores[lang]
        for core, n in con.execute(
                "SELECT core, count FROM cores WHERE leaf=?", (leaf,)):
            st.core_count[core] = n
        for w, cap, low in con.execute(
                "SELECT word, cap, low FROM caps WHERE leaf=?", (leaf,)):
            st.cap_stats[w] = [cap, low]
    con.close()
    return stores


def vera(path: Path) -> Any:
    """A `Vera` built from the published file rather than from pickles."""
    from .vera import Vera
    from .writer import Writer

    v = Vera()
    for lang, st in sorted(load(path).items()):
        v.add(lang, st)
    w = Path(path).parent / "writer.json"
    if w.exists():
        v.writer = Writer.load(w)
    return v


#: Deliberately spans the answering verdicts AND the refusing ones. A round
#: trip that keeps the answers and loses the refusals would look green on a
#: bank of questions the store can answer.
BANK = ("正当防衛とは", "殺人罪の刑は", "契約の成立要件は", "時効とは",
        "過失相殺とは", "超伝導とは", "今日の天気は", "こんにちは",
        "フロベニウス双対とは", "negligence", "consideration", "jurisdiction")


def _shape(v: Any) -> Dict[str, Any]:
    return {lang: {"cores": len(s.crosses),
                   "facets": sum(len(c) for c in s.crosses.values()),
                   "labels": len(getattr(s, "source_labels", ()) or ())}
            for lang, s in sorted(v.stores.items())}


def verify(root: Path, path: Path) -> Dict[str, Any]:
    """Does the published file hold what the built one holds, and answer the same?

    Both halves are needed. The bank alone passed 12/12 while the round trip
    was silently dropping every facet-less core, because a core with nothing
    on it cannot answer a question either way — the shape check is what
    catches a loss the answers cannot show.
    """
    from .vera import load as pickle_load

    a, b = pickle_load(root), vera(path)
    shape_a, shape_b = _shape(a), _shape(b)
    rows, same = [], 0
    for q in BANK:
        ra, rb = a.ask(q), b.ask(q)
        ok = (ra.get("verdict") == rb.get("verdict")
              and ra.get("core") == rb.get("core")
              and ra.get("text") == rb.get("text"))
        same += ok
        rows.append({"q": q, "same": ok, "pickle": ra.get("verdict"),
                     "sqlite": rb.get("verdict"),
                     "core": rb.get("core"), "text": rb.get("text")})
    ok = same == len(BANK) and shape_a == shape_b
    return {"verdict": "ANSWER" if ok else "DRIFTED",
            "identical": f"{same}/{len(BANK)}",
            "shape_matches": shape_a == shape_b,
            "shape": {"pickle": shape_a, "sqlite": shape_b},
            "rows": rows}


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=str(Path.home() / "Projects" / "vera-corpus"))
    ap.add_argument("--out")
    ap.add_argument("--verify", action="store_true")
    a = ap.parse_args(argv)
    root = Path(a.root)
    out = Path(a.out) if a.out else root / "build" / "vera.db"

    rep: Dict[str, Any] = {"verdict": "ANSWER"}
    if not a.verify or not out.exists():
        rep["export"] = export(root, out)
    if a.verify:
        t = time.time()
        rep["load_seconds"] = None
        v = verify(root, out)
        rep["verify"] = v
        rep["verdict"] = v["verdict"]
        rep["load_seconds"] = round(time.time() - t, 1)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    return 0 if rep["verdict"] == "ANSWER" else 1


if __name__ == "__main__":
    sys.exit(main())
