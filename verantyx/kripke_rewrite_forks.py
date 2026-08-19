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
        rewrite_logic_fork(),
        rewrite_rules_are_data_fork(),
        rewrite_permission_fork(),
    ]
