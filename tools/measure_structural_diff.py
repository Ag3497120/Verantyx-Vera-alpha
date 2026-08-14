"""Structural diff: machine-oracle containment + preregistered 30-pair bank.

Protocol (SPEC_2026-08-14_meaning_layers 納品2). The bank file
tools/diff_bank_2026-08-14.json is written BEFORE the first diff call.
A pair whose every comparable layer abstains is an abstention, not a
failure. Machine oracle: remain/held split of (subject, predicate)
pairs, seed 20260814; a claimed only_a predicate must sit on A's side
and not B's side in that split.

## Measured — jawiki leads, heuristic profiles, seed 20260814

    subjects                         1,419,406
    lattice                          527,175 words, 787,333 slots
    shelf cores                      309,864
    aliases                          941,604
    fork STRUCTURAL_DIFF_DEFENSE     pass

    machine oracle (200 subject pairs, remain profiles)
        only_a predicate claims      680
        contained                    668
        leaked onto B                12
        containment rate             0.9824

    preregistered bank (30 pairs, written before first diff)
        hits                         11 / 30
        misses                       19
        abstentions                  0
        (INSUFFICIENT_PROFILE is per-layer; a pair still
         emits a DIFF when any layer survives, so thin
         short-title cores count as misses, not silence)

The oracle is tight: a remain-based only_a claim is almost always
absent from B's remain, and B's held-out leaked 12 of 680. The bank
is not. Heuristic profiles are thin (typical total ~4) and short
shelf cores (水, 馬, 町) are collision dumps, so the expected axis
often sits below markup-free but off-topic high-mass facets. リンゴ/
電気 hits on 果実のことである; 馬/自転車 returns 麻雀. Both numbers
stay visible.
"""
from __future__ import annotations

import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from verantyx.cross_store import CrossStore
from verantyx.lattice import build
from verantyx.predicate_profile import OUT, load
from verantyx.structural_diff import LAYER_PROFILE, diff, regression

SEED = 20260814
HOLD = 0.20
MIN_PREDS = 3
ORACLE_PAIRS = 200
BANK = Path(__file__).resolve().parent / "diff_bank_2026-08-14.json"
ALIASES = (Path.home() / "Projects" / "vera-corpus" / "build"
           / "jawiki_aliases.json")
SHELF = (Path.home() / "Projects" / "vera-corpus" / "build"
         / "jawiki_shallow.json")

_DIFF_CALLS = 0


def _checked_diff(*args, **kwargs):
    global _DIFF_CALLS
    if not BANK.is_file():
        raise SystemExit("bank file missing before first diff: %s" % BANK)
    _DIFF_CALLS += 1
    return diff(*args, **kwargs)


def _json_obj_at(buf: bytes, start: int) -> bytes:
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(buf)):
        c = buf[i]
        if in_str:
            if esc:
                esc = False
            elif c == 0x5C:
                esc = True
            elif c == 0x22:
                in_str = False
            continue
        if c == 0x22:
            in_str = True
        elif c == 0x7B:
            depth += 1
        elif c == 0x7D:
            depth -= 1
            if depth == 0:
                return buf[start:i + 1]
    return b""


def load_shelf(subjects) -> CrossStore:
    """CrossStore over the shallow shelf, local cores only for provenance."""
    raw = SHELF.read_bytes()
    start = raw.find(b'"crosses":')
    obj = raw.find(b"{", start)
    end = raw.find(b', "core_count":')
    if end < 0:
        end = raw.find(b',"core_count":')
    crosses = json.loads(raw[obj:end])
    labels = []
    sl = raw.find(b'"source_labels":')
    if sl >= 0:
        arr = raw.find(b"[", sl)
        labels, _ = json.JSONDecoder().raw_decode(raw[arr:].decode("utf-8"))
    st = CrossStore(crosses=crosses, source_labels=set(labels))
    p0 = raw.find(b'"provenance":')
    p1 = raw.find(b'"source_labels":')
    section = raw[p0:p1] if p0 >= 0 and p1 > p0 else b""
    for s in subjects:
        if not s:
            continue
        needle = ('"%s": {' % s).encode()
        i = section.find(needle)
        if i < 0:
            continue
        j = section.find(b"{", i + len(s.encode()) )
        blob = _json_obj_at(section, j)
        if not blob:
            continue
        try:
            st.provenance[s] = json.loads(blob)
        except json.JSONDecodeError:
            continue
    st.track_provenance = bool(st.provenance)
    return st


def _as_profiles(remain):
    return {s: {"predicates": dict(p), "total": int(sum(p.values()))}
            for s, p in remain.items()}


def _top_blob(items, n=3):
    return "".join(it.get("token", "") for it in items[:n])


def score_pair(pair, result):
    if result.get("verdict") == "INSUFFICIENT_PROFILE":
        return "abstain"
    if pair.get("expect_bucket") == "shared":
        blob = _top_blob(result.get("shared") or [])
        kws = pair.get("axis_shared") or []
        return "hit" if any(kw in blob for kw in kws) else "miss"
    blob_a = _top_blob(result.get("only_a") or [])
    blob_b = _top_blob(result.get("only_b") or [])
    ha = any(kw in blob_a for kw in (pair.get("axis_a") or []))
    hb = any(kw in blob_b for kw in (pair.get("axis_b") or []))
    return "hit" if (ha or hb) else "miss"


def main() -> None:
    t0 = time.time()
    if not BANK.is_file():
        raise SystemExit("preregistered bank missing: %s" % BANK)
    bank_stat = BANK.stat()
    bank = json.loads(BANK.read_text(encoding="utf-8"))
    pairs = bank["pairs"]
    print("bank", BANK, "n", len(pairs), "mtime", int(bank_stat.st_mtime),
          "diff_calls_so_far", _DIFF_CALLS, flush=True)

    print("regression…", flush=True)
    fork = regression()
    # regression() calls diff directly; that is the first diff, and the
    # bank file already existed (checked above). Recount via wrapper next.
    print(json.dumps({"fork": fork["fork"], "pass": fork["pass"],
                      "result": fork["result"]}, ensure_ascii=False),
          flush=True)
    if not fork["pass"]:
        raise SystemExit("STRUCTURAL_DIFF_DEFENSE failed")

    print("loading profiles…", flush=True)
    extractor, profiles = load(OUT)
    print("profiles", len(profiles), "extractor", extractor,
          "%.1fs" % (time.time() - t0), flush=True)

    print("loading aliases…", flush=True)
    aliases = json.loads(ALIASES.read_text(encoding="utf-8"))
    print("aliases", len(aliases), flush=True)

    bank_subjects = []
    for p in pairs:
        bank_subjects.append(p["a"])
        bank_subjects.append(p["b"])
        bank_subjects.append(aliases.get(p["a"], p["a"]))
        bank_subjects.append(aliases.get(p["b"], p["b"]))
    bank_subjects = sorted(set(bank_subjects))

    print("loading shelf…", flush=True)
    shelf = load_shelf(bank_subjects)
    print("shelf cores", len(shelf.crosses), "prov", len(shelf.provenance),
          "%.1fs" % (time.time() - t0), flush=True)

    print("building lattice…", flush=True)
    lat = build(profiles)
    print("lattice", json.dumps(lat.report()),
          "%.1fs" % (time.time() - t0), flush=True)

    # --- machine oracle -------------------------------------------------
    ge3 = {s: r for s, r in profiles.items()
           if len(r["predicates"]) >= MIN_PREDS}
    all_pairs = [(s, p) for s, r in ge3.items() for p in r["predicates"]]
    all_pairs.sort()
    rng = random.Random(SEED)
    rng.shuffle(all_pairs)
    n_hold = int(len(all_pairs) * HOLD)
    held = all_pairs[:n_hold]
    held_set = set(held)
    remain = {}
    for s, r in profiles.items():
        keep = {p: n for p, n in r["predicates"].items()
                if (s, p) not in held_set}
        if keep:
            remain[s] = keep
    held_by = {}
    for s, p in held:
        held_by.setdefault(s, set()).add(p)
    remain_profiles = _as_profiles(remain)

    elig = sorted(s for s, preds in remain.items() if len(preds) >= MIN_PREDS)
    n_sample = min(ORACLE_PAIRS, len(elig) // 2)
    oracle_pairs = [(elig[i], elig[-(i + 1)]) for i in range(n_sample)]

    claims = 0
    contained = 0
    leaked = 0
    print("oracle pairs", len(oracle_pairs), flush=True)
    for ia, (sa, sb) in enumerate(oracle_pairs):
        if ia % 50 == 0:
            print("oracle", ia, flush=True)
        result = _checked_diff(
            sa, sb, profiles=remain_profiles, aliases=aliases,
            lattice=lat, shelf=shelf, k=8, min_profile=MIN_PREDS,
        )
        ca = result["canonical"]["a"]
        cb = result["canonical"]["b"]
        a_side = set(remain.get(ca, {})) | held_by.get(ca, set())
        b_side = set(remain.get(cb, {})) | held_by.get(cb, set())
        for it in result.get("only_a") or []:
            if it.get("layer") != LAYER_PROFILE:
                continue
            tok = it.get("token")
            claims += 1
            if tok in a_side and tok not in b_side:
                contained += 1
            if tok in b_side:
                leaked += 1

    containment = round(contained / claims, 4) if claims else None

    # --- preregistered bank --------------------------------------------
    bank_rows = []
    hits = misses = abstentions = 0
    for pair in pairs:
        result = _checked_diff(
            pair["a"], pair["b"], profiles=profiles, aliases=aliases,
            lattice=lat, shelf=shelf, k=8, min_profile=MIN_PREDS,
        )
        mark = score_pair(pair, result)
        if mark == "hit":
            hits += 1
        elif mark == "abstain":
            abstentions += 1
        else:
            misses += 1
        bank_rows.append({
            "a": pair["a"], "b": pair["b"],
            "expected_axis": pair["expected_axis"],
            "mark": mark,
            "verdict": result["verdict"],
            "coverage": result["coverage"],
            "abstain": result["abstain"],
            "top_only_a": [it["token"] for it in result["only_a"][:3]],
            "top_only_b": [it["token"] for it in result["only_b"][:3]],
            "top_shared": [it["token"] for it in result["shared"][:3]],
        })

    example = _checked_diff(
        "リンゴ", "電気", profiles=profiles, aliases=aliases,
        lattice=lat, shelf=shelf, k=8, min_profile=MIN_PREDS,
    )

    out = {
        "extractor": extractor,
        "subjects": len(profiles),
        "seed": SEED,
        "bank_path": str(BANK),
        "bank_existed_before_first_wrapper_diff": True,
        "bank_mtime": int(bank_stat.st_mtime),
        "diff_calls": _DIFF_CALLS,
        "fork": {"name": fork["fork"], "pass": fork["pass"]},
        "oracle": {
            "pairs": len(oracle_pairs),
            "only_a_predicate_claims": claims,
            "contained": contained,
            "leaked_to_b": leaked,
            "containment_rate": containment,
        },
        "bank": {
            "n": len(pairs),
            "hits": hits,
            "misses": misses,
            "abstentions": abstentions,
            "score": "%d/%d" % (hits, len(pairs)),
            "rows": bank_rows,
        },
        "example_apple_electricity": example,
        "seconds": round(time.time() - t0, 1),
    }
    print(json.dumps(out, ensure_ascii=False, indent=1), flush=True)


if __name__ == "__main__":
    main()
