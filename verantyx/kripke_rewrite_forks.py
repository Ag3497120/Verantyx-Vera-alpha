"""Forks for Kripke model checking + term rewriting kernel."""
from __future__ import annotations

from typing import Any, Dict, List

from .cross_store import CrossStore
from .kripke import KripkeModel, check
from .rewrite_kernel import (
    RuleStore,
    default_algebra_rules,
    default_logic_rules,
    simplify,
)


def _model() -> KripkeModel:
    m = KripkeModel()
    m.add_world("w1", ["p"])
    m.add_world("w2", ["p", "q"])
    m.add_world("w3", ["q"])
    m.add_edge("w1", "w2")
    m.add_edge("w1", "w3")
    m.add_edge("w2", "w3")
    return m


def kripke_model_check_fork() -> Dict[str, Any]:
    """□=全後続一致 / ◇=1本到達 / 空虚な□ / 決定論."""
    m = _model()
    box_q = check(m, "box q", "w1")
    box_p = check(m, "box p", "w1")
    dia_pq = check(m, "dia (p and q)", "w1")
    vac = check(m, "box p", "w3")
    ok = (
        box_q["verdict"] == "ANSWER" and box_q["value"] is True
        and box_p["value"] is False
        and dia_pq["value"] is True
        and vac["value"] is True  # 後続なし → 空虚に真
        and check(m, "box q", "w1") == box_q
    )
    return {
        "experiment": "kripke",
        "fork": "KRIPKE_MODEL_CHECK",
        "pass": bool(ok),
        "result": {"box_q": box_q["value"], "box_p": box_p["value"],
                   "dia_pq": dia_pq["value"], "vacuous_box": vac["value"]},
    }


def kripke_validity_fork() -> Dict[str, Any]:
    """モデル内妥当性 = 全世界一致ゲート."""
    m = _model()
    valid = check(m, "p or q")
    invalid = check(m, "p and q")
    ok = (
        valid["value"] is True
        and invalid["value"] is False
        and invalid["per_world"]["w1"] is False
    )
    return {
        "experiment": "kripke",
        "fork": "KRIPKE_VALIDITY",
        "pass": bool(ok),
        "result": {"valid": valid["value"], "invalid_per_world": invalid["per_world"]},
    }


def kripke_unknown_refuses_fork() -> Dict[str, Any]:
    """未知世界・モデルに無い命題・非論理式は型付き拒否."""
    m = _model()
    prop = check(m, "box r", "w1")
    world = check(m, "p", "w9")
    junk = check(m, "p ++ q", "w1")
    ok = (
        prop["verdict"] == "UNKNOWN_NO_EVIDENCE"
        and world["verdict"] == "UNKNOWN_NO_EVIDENCE"
        and junk["verdict"] == "UNKNOWN_UNPARSED"
    )
    return {
        "experiment": "kripke",
        "fork": "KRIPKE_UNKNOWN_REFUSES",
        "pass": bool(ok),
        "result": {"prop": prop["verdict"], "world": world["verdict"],
                   "junk": junk["verdict"]},
    }





def completion_derives_laws_fork() -> Dict[str, Any]:
    """完備化は、公理に無い等式を導く — そしてそれは真である。

    群の公理3本(結合・左単位元・**左**逆元)から臨界対を計算して
    完備化すると、公理に無い `x * i(x) → e`(**右**逆元)と
    `i(i(x)) * z → x * z`(対合)が導かれる。実測 2026-08-19:
    12本導出、Lean が整数の加法として解釈して **12/12 VERIFIED**、偽0。

    これは帰納法ではない。**格納された規則の帰結を明示化する**操作で、
    根拠は全て格納済み — 閉包の内側。ただし導けるのは与えた公理の
    帰結だけで、`n + m = m + n` のように再帰的定義から帰納法でしか
    出ない等式は出ない。experiments/completion。

    2点: (a) 公理に無い等式が導かれる (b) 導かれた等式が真である
    (整数の加法として Lean が検査)。
    """
    import sys as _sys
    from pathlib import Path as _P

    _sys.path.insert(0, str(_P(__file__).resolve().parents[1]
                            / "experiments" / "completion"))
    try:
        from run_completion import complete, to_int_expr
    except Exception as exc:      # 実験ファイルが無い配布形では報告して抜ける
        return {"experiment": "rewrite", "fork": "COMPLETION_DERIVES_LAWS",
                "skipped": "run_completion not importable: %s" % type(exc).__name__}
    from .lean_witness import lean_binary, verify
    from .rewrite_kernel import RuleStore

    if lean_binary() is None:
        return {"experiment": "rewrite", "fork": "COMPLETION_DERIVES_LAWS",
                "skipped": "no lean toolchain on this machine"}

    rs = RuleStore()
    rs.add("assoc", "(?x * ?y) * ?z", "?x * (?y * ?z)")
    rs.add("unit", "e * ?x", "?x")
    rs.add("inv", "i(?x) * ?x", "e")
    added, log, _done = complete(rs, max_rules=8, rounds=8)

    derived = added > 0
    ver = 0
    for e in log:
        lhs = to_int_expr(e["lhs"]).replace("?", "")
        rhs = to_int_expr(e["rhs"]).replace("?", "")
        v = verify("theorem t (x y z : Int) : %s = %s := by omega"
                   % (lhs, rhs))
        if str(v.get("verdict")) == "VERIFIED":
            ver += 1
    sound = (ver == added)
    ok = bool(derived and sound)
    return {"experiment": "rewrite", "fork": "COMPLETION_DERIVES_LAWS",
            "pass": ok,
            "result": {"derived": added, "lean_verified": ver,
                       "laws": [e["lhs"] + " -> " + e["rhs"] for e in log[:4]]}}


def ordered_rewrite_fork() -> Dict[str, Any]:
    """対称規則は、向きを付ければ止まり、置換を同一視する。

    可換律 `?a + ?b → ?b + ?a` を無向きで足すと a+b→b+a→a+b と往復し、
    正規形到達が 60/60 → 19/60 に崩れる(実測 2026-08-19、健全性は保持)。
    結果が項順序で真に小さくなる時だけ適用すると 60/60 に戻り、さらに
    `a+b+c` の12通り(順列×括弧付け)が**単一の正規形**に落ちる。

    探索ではない — 適用可否を決定論的な述語で決めるだけで、後戻りも
    分岐の試行もしない。格納された規則を格納された順序で適用する。

    3点: (a) 無向きは予算切れ (b) 向き付きは正規形に至る
    (c) 置換が同一視される(正準化)。
    """
    import itertools

    from .rewrite_kernel import default_algebra_rules, simplify

    # 正準化には可換だけでなく**結合の並べ替え**も要る。可換律だけだと
    # (a+b)+c と a+(b+c) は別の項のままで、12通りは3形に留まる(実測)。
    # 実測(experiments/ordered_rewrite)と同じ規則集合で固定する。
    SYM = [("add_comm", "?a + ?b", "?b + ?a"),
           ("add_assoc_l", "(?a + ?b) + ?c", "?a + (?b + ?c)")]
    rs = default_algebra_rules()
    for n, l, r in SYM:
        rs.add(n, l, r)
    names = [n for n, _l, _r in SYM]

    loops = simplify("(b + a)", rs)
    stops = simplify("(b + a)", rs, oriented=names)

    forms = set()
    for p_ in itertools.permutations(["a", "b", "c"]):
        for shape in ("(%s + (%s + %s))", "((%s + %s) + %s)"):
            r = simplify(shape % p_, rs, oriented=names)
            forms.add(r.get("term") if str(r.get("verdict")) == "ANSWER"
                      else "BUDGET")

    ok = (str(loops.get("verdict")) == "UNKNOWN_BUDGET"
          and str(stops.get("verdict")) == "ANSWER"
          and len(forms) == 1 and "BUDGET" not in forms)
    return {"experiment": "rewrite", "fork": "ORDERED_REWRITE_CANONICAL",
            "pass": bool(ok),
            "result": {"unoriented": loops.get("verdict"),
                       "oriented": stops.get("term"),
                       "permutation_forms": sorted(forms)}}


def rewrite_roundtrip_fork() -> Dict[str, Any]:
    """印字した項は、読み直すと同じ項である。

    `term_to_str` は右側の複合項に、演算子が - か * のときだけ括弧を
    付けていた。そのため `a + (b + c)` が `a + b + c` と印字され、
    読み直すと `(a + b) + c` — **別の項**になる(2026-08-19実測、往復
    2/8 で破綻)。Int では両方とも真の等式になるので実害は出ていなかったが、
    印字は項の表現であって「たまたま同値な別の項」ではない。非結合的な
    演算子を載せた瞬間に嘘になる。

    そしてこの欠陥は実際に**測定を誤らせた**: 帰納法の壁の実験で、
    印字文字列を比べたために加法の結合律が「導出された」ように見えた
    (カーネルは内部では正しく区別していた)。

    往復性は表現の最低条件なので、ここで固定する。
    """
    from .rewrite_kernel import parse_term, term_to_str

    cases = ["(a + (b + c))", "((a + b) + c)", "(a - (b - c))",
             "((a - b) - c)", "(a * (b + c))", "((a * b) + c)",
             "(a - (b + c))", "(a + (b - c))", "(a * (b * c))",
             "((a + b) * c)", "(1 + (2 * 3))", "((a + 0) - (b - c))"]
    broken = []
    for src in cases:
        t = parse_term(src)
        if t is None:
            broken.append((src, "unparsed"))
            continue
        again = parse_term(term_to_str(t))
        if again != t:
            broken.append((src, term_to_str(t)))
    ok = not broken
    return {"experiment": "rewrite", "fork": "REWRITE_PRINT_ROUNDTRIP",
            "pass": ok,
            "result": {"cases": len(cases), "broken": len(broken),
                       "examples": broken[:3]}}


def rewrite_algebra_fork() -> Dict[str, Any]:
    """代数規則セット + wire 定数畳み込み (トレース付き正規形)."""
    a = simplify("x + 0")
    b = simplify("(2 + 3) * y")
    c = simplify("(1 * x) + (y * 0)")
    d = simplify("x - x")
    ok = (
        a["term"] == "x"
        and b["term"] == "5 * y"
        and any(s["rule"] == "wire_add" for s in b["steps"])
        and c["term"] == "x"
        and d["term"] == "0"
        and all(r["verdict"] == "ANSWER" for r in (a, b, c, d))
    )
    return {
        "experiment": "rewrite",
        "fork": "REWRITE_ALGEBRA",
        "pass": bool(ok),
        "result": {"a": a["term"], "b": b["term"], "c": c["term"],
                   "b_rules": [s["rule"] for s in b["steps"]]},
    }


def rewrite_logic_fork() -> Dict[str, Any]:
    """論理は同カーネルの規則セット違い."""
    r = simplify("lnot(lnot(land(true, p)))", default_logic_rules())
    r2 = simplify("lor(false, lor(q, true))", default_logic_rules())
    ok = r["term"] == "p" and r2["term"] == "true"
    return {
        "experiment": "rewrite",
        "fork": "REWRITE_LOGIC_SET",
        "pass": bool(ok),
        "result": {"double_neg": r["term"], "or_true": r2["term"]},
    }


def rewrite_rules_are_data_fork() -> Dict[str, Any]:
    """規則は十字に pour → 復元しても同じ挙動。実行時の規則追加で挙動が変わる."""
    cs = CrossStore()
    default_algebra_rules().pour_into(cs)
    poured = RuleStore.from_cross_store(cs)
    same = simplify("x + 0", poured)["term"] == "x"

    ext = default_algebra_rules()
    before = simplify("y * 2", ext)["term"]
    ext.add("mul_two", "?a * 2", "?a + ?a")
    after = simplify("y * 2", ext)["term"]
    ok = same and before == "y * 2" and after == "y + y"
    return {
        "experiment": "rewrite",
        "fork": "REWRITE_RULES_ARE_DATA",
        "pass": bool(ok),
        "result": {"poured_rules": len(poured.rules), "before": before,
                   "after": after},
    }


def rewrite_permission_fork() -> Dict[str, Any]:
    """H と同型の許可: 許可されない規則は発火しない (正規形が変わる)."""
    blocked = simplify("x + 0", allowed=["mul_one_r"])
    allowed = simplify("x + 0", allowed=["add_zero_r"])
    ok = blocked["term"] == "x + 0" and allowed["term"] == "x"
    return {
        "experiment": "rewrite",
        "fork": "REWRITE_PERMISSION_H",
        "pass": bool(ok),
        "result": {"blocked": blocked["term"], "allowed": allowed["term"]},
    }


def all_kripke_rewrite_forks() -> List[Dict[str, Any]]:
    return [
        kripke_model_check_fork(),
        kripke_validity_fork(),
        kripke_unknown_refuses_fork(),
        rewrite_algebra_fork(),
        rewrite_roundtrip_fork(),
        ordered_rewrite_fork(),
        completion_derives_laws_fork(),
        rewrite_logic_fork(),
        rewrite_rules_are_data_fork(),
        rewrite_permission_fork(),
    ]
