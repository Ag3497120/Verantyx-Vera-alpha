# -*- coding: utf-8 -*-
"""立体十字エネルギー証明器 — 候補補題を腕に置き、エネルギーが経路を決める。

構想(これまで.pdf)の写像:
  「一番端に足されるクエリによる各ノードが持つエネルギー比率が変わる
   ことによって中心に向かっての経路が変わることによって推論する」
  → 詰まった段の等式がクエリ。候補補題が腕。記号被覆×特定性×接地検査
    がエネルギー。厳密首位の候補から証明を試み、証明できた補題が規則に
    昇格して段を解錠する。

新しい器官(このコードベースに無かったもの):
  1. 有限接地モデル検査 — 候補等式を小さい接地項で全数評価し、両辺の
     正規形が食い違えば REFUTED(反例つき)。偽の候補は主張の前に死ぬ。
     「証明できない」と「偽」を初めて分ける(不在と否定を混ぜない、の数学版)。
  2. 部分項の抽象化 + 右辺列挙 — 詰まった等式の部分項から補題の左辺を
     作り(異質部分項→変数)、右辺は同じ変数上の小さい項を列挙して
     接地検査で淘汰する。梯子の段の「発明」はここで起きる。
  3. 十字のアジェンダ — 生き残った候補を ShellCross の腕に置き、
     エネルギー厳密首位から試す。試行台帳(failed_before)が減点され、
     二度目は探索でなく参照になる。

健全性は構造で保証: 候補がどれだけ間違っていても、書き換え核が閉じ
なければ規則に昇格しない。昇格した規則は全て自前の帰納証明を持ち、
最終的に Lean が独立検査する。エネルギーは探索の順序だけを決め、
真偽には一切触れない — 票を持たない。
"""
from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from verantyx.cross import AXES, ShellCross
from verantyx.face_roles import FACET_FACES
from verantyx.rewrite_kernel import RuleStore, parse_term, simplify, term_to_str

# ---------------------------------------------------------------------------
# 署名(型つき): N = 自然数, L = リスト
# ---------------------------------------------------------------------------
SIG: Dict[str, Tuple[Tuple[str, ...], str]] = {
    "0":    ((), "N"),
    "s":    (("N",), "N"),
    "add":  (("N", "N"), "N"),
    "mul":  (("N", "N"), "N"),
    "nil":  ((), "L"),
    "cons": (("N", "L"), "L"),
    "app":  (("L", "L"), "L"),
    "rev":  (("L",), "L"),
    "len":  (("L",), "N"),
    # PREREG12: 不等式は等式に畳む — le は B 値の再帰関数、
    # 命題 a≤b は等式 le(a,b)=true。monus は切り捨て減算。
    "true":  ((), "B"),
    "false": ((), "B"),
    "le":    (("N", "N"), "B"),
    "monus": (("N", "N"), "N"),
}
CONSTRUCTORS = {"0", "s", "nil", "cons", "true", "false"}
DEFINED = {f for f in SIG if f not in CONSTRUCTORS}
AC_OPS = ("add", "mul")

DEFS = [
    ("add_0", "add(?x, 0)", "?x"),
    ("add_s", "add(?x, s(?y))", "s(add(?x, ?y))"),
    ("mul_0", "mul(?x, 0)", "0"),
    ("mul_s", "mul(?x, s(?y))", "add(mul(?x, ?y), ?x)"),
    ("app_nil", "app(nil, ?y)", "?y"),
    ("app_cons", "app(cons(?h, ?t), ?y)", "cons(?h, app(?t, ?y))"),
    ("rev_nil", "rev(nil)", "nil"),
    ("rev_cons", "rev(cons(?h, ?t))", "app(rev(?t), cons(?h, nil))"),
    ("len_nil", "len(nil)", "0"),
    ("len_cons", "len(cons(?h, ?t))", "s(len(?t))"),
    ("le_0", "le(0, ?y)", "true"),
    ("le_s0", "le(s(?x), 0)", "false"),
    ("le_ss", "le(s(?x), s(?y))", "le(?x, ?y)"),
    ("monus_0", "monus(?x, 0)", "?x"),
    ("monus_0l", "monus(0, ?y)", "0"),
    ("monus_ss", "monus(s(?x), s(?y))", "monus(?x, ?y)"),
]

#: 条件付き規則(PREREG12 v0)。条件は放電義務 — マッチ代入の下で
#: 先に証明できた時だけ発火する。仮定される条件は存在しない。
COND_RULES = [
    ("R_le0", "monus(?x, ?y)", "0", ("le(?x, ?y)", "true")),
]

#: 変数はこの表に載った名前だけ(閉じた表 — 開いた賢さは持ち込まない)
VAR_TYPES: Dict[str, str] = {
    "x": "L", "y": "L", "z": "L", "l": "L", "m": "L",
    "n": "N", "a": "N", "b": "N", "c": "N",
}
#: 帰納で導入する新鮮な定数(変数ではない — IH の一般化から除外される)。
#: 型ごとに名前を分ける: kn=N上の帰納、kl=L上の帰納、h0=consの頭。
FRESH = {"kn": "N", "kl": "L", "h0": "N"}


# ---------------------------------------------------------------------------
# 項の操作(tuple 表現、rewrite_kernel.parse_term と同じ形)
# ---------------------------------------------------------------------------
def t_vars(t: Any) -> List[str]:
    if not isinstance(t, tuple):
        return [t] if isinstance(t, str) and t in VAR_TYPES else []
    out: List[str] = []
    for a in t[1:]:
        for v in t_vars(a):
            if v not in out:
                out.append(v)
    return out


def t_subst(t: Any, env: Dict[str, Any]) -> Any:
    if not isinstance(t, tuple):
        return env.get(t, t) if isinstance(t, str) else t
    return (t[0],) + tuple(t_subst(a, env) for a in t[1:])


def t_size(t: Any) -> int:
    if not isinstance(t, tuple):
        return 1
    return 1 + sum(t_size(a) for a in t[1:])


def t_symbols(t: Any) -> Set[str]:
    if not isinstance(t, tuple):
        return ({t} if isinstance(t, str) and t not in VAR_TYPES
                and t not in FRESH else set())
    out = {t[0]}
    for a in t[1:]:
        out |= t_symbols(a)
    return out


def subterms(t: Any) -> List[Any]:
    out = [t]
    if isinstance(t, tuple):
        for a in t[1:]:
            out += subterms(a)
    return out


def infer_type(t: Any, tenv: Dict[str, str]) -> Optional[str]:
    """項の型。矛盾したら None(型の合わない候補は列挙から落ちる)。"""
    if isinstance(t, int):
        return "N"          # parse_term は数字を int 葉にする
    if isinstance(t, str):
        if t in tenv:
            return tenv[t]
        if t in SIG:
            return SIG[t][1]
        if t in FRESH:
            return FRESH[t]
        return None
    if t[0] not in SIG:
        return None
    args, ret = SIG[t[0]]
    if len(args) != len(t) - 1:
        return None
    for want, a in zip(args, t[1:]):
        got = infer_type(a, tenv)
        if got is not None and got != want:
            return None
        if got is None and isinstance(a, str) and a in VAR_TYPES:
            return None
    return ret


def well_typed(t: Any) -> bool:
    tenv = {v: VAR_TYPES[v] for v in t_vars(t)}
    return infer_type(t, tenv) is not None


# ---------------------------------------------------------------------------
# AC 正規化つき正規形(run_ac.py の実測済みの形)
# ---------------------------------------------------------------------------
def _flat(t: Any, op: str) -> List[Any]:
    if isinstance(t, tuple) and t[0] == op and len(t) == 3:
        return _flat(t[1], op) + _flat(t[2], op)
    return [ac_norm(t)]


def ac_norm(t: Any) -> Any:
    if isinstance(t, tuple) and t[0] in AC_OPS and len(t) == 3:
        parts = sorted(_flat(t, t[0]),
                       key=lambda x: x if isinstance(x, str) else term_to_str(x))
        out = parts[-1]
        for p in reversed(parts[:-1]):
            out = (t[0], p, out)
        return out
    if isinstance(t, tuple):
        return (t[0],) + tuple(ac_norm(x) for x in t[1:])
    return t


def make_store(rules: List[Tuple[str, str, str]]) -> RuleStore:
    rs = RuleStore()
    for n, l, r in rules:
        rs.add(n, l, r)
    return rs


MAX_TERM = 80   # 項サイズの門。超えたら None(予算切れと同じ「主張しない」)


def nf(term: Any, rules: List[Tuple[str, str, str]],
       oriented: Optional[List[str]] = None, budget: int = 600,
       fired: Optional[List[str]] = None) -> Optional[Any]:
    """正規形(AC正規化込み)。予算切れ・サイズ超過は None — 一致ではない。

    ``fired`` にリストを渡すと、発火した規則名が追記される(引用の記録)。
    """
    if not isinstance(term, str) and t_size(term) > MAX_TERM:
        return None
    e = term if isinstance(term, str) else term_to_str(term)
    try:
        r = simplify(e, make_store(rules), oriented=oriented or None,
                     budget=budget)
    except RecursionError:
        # 増大規則が深さ方向に爆発した — 予算切れと同じ「主張しない」
        return None
    if str(r.get("verdict")) != "ANSWER":
        return None
    if fired is not None:
        fired.extend(s["rule"] for s in r.get("steps", []))
    out = parse_term(r.get("term"))
    if t_size(out) > MAX_TERM:
        return None
    return ac_norm(out)


# ---------------------------------------------------------------------------
# 器官1: 有限接地モデル検査(反例つき REFUTED)
# ---------------------------------------------------------------------------
def _ground_terms(ty: str) -> List[Any]:
    """小さい接地項。N: 0..3 / L: 長さ0..2 の {0,s(0)} リスト。"""
    if ty == "N":
        out: List[Any] = ["0"]
        for _ in range(3):
            out.append(("s", out[-1]))
        return out
    nats = ["0", ("s", "0")]
    lists: List[Any] = ["nil"]
    for h in nats:
        lists.append(("cons", h, "nil"))
    for h in nats:
        for g in nats:
            lists.append(("cons", h, ("cons", g, "nil")))
    return lists


_GROUND = {"N": _ground_terms("N"), "L": _ground_terms("L")}


def ground_check(lhs: Any, rhs: Any, max_cases: int = 64) -> Dict[str, Any]:
    """全変数に接地項を代入し、DEFS だけで両辺を評価して比較する。

    定義は構成子上で完全なので、接地項の正規形は構成子項に落ちる —
    そこで食い違えば等式は偽(REFUTED、反例を名指す)。全て一致なら
    PASSED(検査数つき)。予算切れが混じれば UNDECIDED(主張しない)。
    """
    vs = t_vars(lhs) + [v for v in t_vars(rhs) if v not in t_vars(lhs)]
    domains = [_GROUND[VAR_TYPES[v]] for v in vs]
    cases = list(itertools.product(*domains))[:max_cases] if vs else [()]
    passed = 0
    for combo in cases:
        env = dict(zip(vs, combo))
        a = nf(t_subst(lhs, env), DEFS)
        b = nf(t_subst(rhs, env), DEFS)
        if a is None or b is None:
            return {"verdict": "UNDECIDED", "passed": passed,
                    "reason": "budget"}
        if a != b:
            return {"verdict": "REFUTED", "passed": passed,
                    "witness": {v: term_to_str(t) if not isinstance(t, str)
                                else t for v, t in env.items()},
                    "lhs_value": term_to_str(a) if not isinstance(a, str) else a,
                    "rhs_value": term_to_str(b) if not isinstance(b, str) else b}
        passed += 1
    return {"verdict": "PASSED", "passed": passed}


# ---------------------------------------------------------------------------
# 器官2: 候補の発明 — 部分項の抽象化(左辺) + 小項の列挙(右辺)
# ---------------------------------------------------------------------------
def _abstract(t: Any, alien_ok) -> List[Tuple[Any, Dict[str, Any]]]:
    """異質部分項(alien_ok が真を返すもの)を新変数に置いた抽象の列挙。

    決定論: 各異質部分項について「置く/置かない」の全組合せを、
    出現順に生成する(上限つき)。"""
    aliens: List[Any] = []
    for st in subterms(t):
        if not isinstance(st, str) and alien_ok(st) and st not in aliens:
            aliens.append(st)
    # PREREG3 変更1: マスク列挙(先頭3個)に、極小異質項の一括変数化と
    # 極大異質項の一括変数化を**追加**する。一般結合律のような3箇所同時の
    # 抽象は、マスクの走査順では構造的に生まれなかった(確認2の実測)。
    def _contains(big: Any, small: Any) -> bool:
        return big != small and small in subterms(big)

    minimal = [a for a in aliens
               if not any(_contains(a, b) for b in aliens)]
    maximal = [a for a in aliens
               if not any(_contains(b, a) for b in aliens)]
    extra_sets = []
    if minimal:
        extra_sets.append(minimal[:4])
    if maximal and maximal != minimal:
        extra_sets.append(maximal[:4])

    aliens = aliens[:3]
    # PREREG4: 空集合(改名のみ)も候補にする — 詰まりに異質項が無い
    # とき(B7 の s(add(0,kn)))、部分項そのものが補題の左辺になれる。
    chosen_sets = [
        [a for i, a in enumerate(aliens) if mask >> i & 1]
        for mask in range(1, 1 << len(aliens))
    ] + extra_sets + [[]]
    out: List[Tuple[Any, Dict[str, Any]]] = []
    for chosen in chosen_sets:
        env: Dict[str, Any] = {}
        # 既に t に居る変数名は新変数に使わない(衝突すると別物が同一視
        # される)。極小/極大の一括変数化では3個以上要るので池も広げる。
        _used = set(t_vars(t))
        fresh_l = iter([n for n in ("x", "y", "z", "l", "m")
                        if n not in _used])
        fresh_n = iter([n for n in ("a", "b", "c", "n")
                        if n not in _used])
        def repl(u: Any) -> Any:
            for c in chosen:
                if u == c:
                    key = term_to_str(c) if not isinstance(c, str) else c
                    if key not in env:
                        ty = infer_type(c, {v: VAR_TYPES[v] for v in t_vars(c)})
                        pool = fresh_l if ty == "L" else fresh_n
                        try:
                            env[key] = next(pool)
                        except StopIteration:
                            return u
                    return env[key]
            if not isinstance(u, tuple):
                return u
            return (u[0],) + tuple(repl(a) for a in u[1:])
        out.append((repl(t), env))
    return out


_ENUM_MEMO: Dict[Tuple[Tuple[str, ...], str], List[Any]] = {}


def _enumerate_rhs(vs: List[str], want_ty: str, max_size: int = 7) -> List[Any]:
    """変数集合上の小さい項を列挙(型つき・サイズ上限・決定論順)。"""
    memo_key = (tuple(sorted(vs)), want_ty)
    if memo_key in _ENUM_MEMO:
        return _ENUM_MEMO[memo_key]
    by_ty: Dict[str, List[Any]] = {"N": [], "L": [], "B": []}
    for v in vs:
        by_ty[VAR_TYPES[v]].append(v)
    by_ty["N"].append("0")
    by_ty["L"].append("nil")
    by_ty["B"].extend(["true", "false"])
    grown = dict(by_ty)
    for _round in range(3):
        new: Dict[str, List[Any]] = {"N": [], "L": [], "B": []}
        for f, (args, ret) in sorted(SIG.items()):
            if not args:
                continue
            pools = [grown[a] for a in args]
            for combo in itertools.product(*pools):
                t = (f,) + tuple(combo)
                if t_size(t) <= max_size and t not in grown[ret] + new[ret]:
                    new[ret].append(t)
        for ty in ("N", "L", "B"):
            grown[ty] = grown[ty] + new[ty]
            grown[ty] = grown[ty][:400]
    cands = [t for t in grown[want_ty] if set(t_vars(t)) <= set(vs)]
    out = sorted(cands, key=lambda t: (t_size(t), term_to_str(t)
                                       if not isinstance(t, str) else t))
    _ENUM_MEMO[memo_key] = out
    return out


def invent_candidates(stuck_l: Any, stuck_r: Any,
                      rules: List[Tuple[str, str, str]],
                      max_cands: int = 24) -> List[Dict[str, Any]]:
    """詰まった段の等式から補題候補を発明し、接地検査で淘汰する。"""
    def alien(st: Any) -> bool:
        syms = t_symbols(st)
        if bool(syms & DEFINED) and (
                any(s in FRESH for s in _strs(st)) or t_size(st) >= 2):
            return True
        # PREREG2 の単一変更: 新鮮定数入りの構成子項(cons(h0, nil) 等)も
        # 抽象できる — これが無いと一般結合律 app(app(x,y),z) の z が
        # 生まれず、C1 の梯子が構造的に作れなかった(確認1で実測)。
        return (not (syms & DEFINED) and t_size(st) >= 2
                and any(s in FRESH for s in _strs(st)))

    out: List[Dict[str, Any]] = []
    seen: Set[Tuple[str, str]] = set()
    for side in (stuck_l, stuck_r):
        for st in subterms(side):
            if isinstance(st, str) or not (t_symbols(st) & DEFINED):
                continue
            for lhs, _env in _abstract(st, alien):
                lhs = _rename_fresh(lhs)
                if not well_typed(lhs) or not (t_symbols(lhs) & DEFINED):
                    continue
                vs = t_vars(lhs)
                if not vs:
                    continue
                ty = infer_type(lhs, {v: VAR_TYPES[v] for v in vs})
                if ty is None:
                    continue
                for rhs in _enumerate_rhs(vs, ty):
                    if rhs == lhs:
                        continue
                    # 向き: 規則に昇格したとき左→右でサイズ非増大である
                    # こと(completion の orient と同じ線)。増大規則は
                    # 自分の右辺に再発火して深さ爆発する実測がある。
                    if t_size(rhs) > t_size(lhs):
                        continue
                    # 右辺の変数は左辺の変数の部分集合(規則の健全条件)
                    if not set(t_vars(rhs)) <= set(t_vars(lhs)):
                        continue
                    key = (term_to_str(lhs), term_to_str(rhs)
                           if not isinstance(rhs, str) else rhs)
                    if key in seen:
                        continue
                    seen.add(key)
                    # 既に規則で同じ正規形なら新情報ではない
                    a, b = nf(lhs, rules), nf(rhs, rules)
                    if a is not None and a == b:
                        continue
                    g = ground_check(lhs, rhs)
                    if g["verdict"] != "PASSED":
                        continue
                    out.append({"lhs": lhs, "rhs": rhs,
                                "origin": term_to_str(st),
                                "ground_passed": g["passed"]})
                    if len(out) >= max_cands:
                        return out
    return out


def _rename_fresh(t: Any) -> Any:
    """帰納定数 kn/kl/h0 が残った抽象は、変数に付け替える(型を保って)。"""
    env: Dict[str, Any] = {}
    ks = sorted({s for s in _strs(t) if s in FRESH})
    for s in ks:
        ty = FRESH[s]
        pool = ["c", "b", "a"] if ty == "N" else ["z", "y", "x"]
        used = set(t_vars(t)) | set(env.values())
        for cand in pool:
            if cand not in used:
                env[s] = cand
                break
    return t_subst(t, env) if env else t


def _strs(t: Any) -> List[str]:
    if not isinstance(t, tuple):
        return [t] if isinstance(t, str) else []
    out: List[str] = []
    for a in t[1:]:
        out += _strs(a)
    return out


# ---------------------------------------------------------------------------
# 器官3: 十字のアジェンダ — 腕に候補、エネルギーが順序を決める
# ---------------------------------------------------------------------------
def match_pat(pat: Any, term: Any, b: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """?変数つきパターンの一致(一階・決定論)。"""
    if isinstance(pat, str) and pat.startswith("?"):
        v = pat[1:]
        if v in b:
            return b if b[v] == term else None
        b2 = dict(b)
        b2[v] = term
        return b2
    if isinstance(pat, tuple):
        if (not isinstance(term, tuple) or term[0] != pat[0]
                or len(term) != len(pat)):
            return None
        for p_, t_ in zip(pat[1:], term[1:]):
            b = match_pat(p_, t_, b)
            if b is None:
                return None
        return b
    return b if pat == term else None


def subst_pat(pat: Any, b: Dict[str, Any]) -> Any:
    if isinstance(pat, str) and pat.startswith("?"):
        return b[pat[1:]]
    if isinstance(pat, tuple):
        return (pat[0],) + tuple(subst_pat(a, b) for a in pat[1:])
    return pat


class TrialLedger:
    """試行の記憶。二度目は探索でなく参照(資産探索と同じ型)。"""

    def __init__(self) -> None:
        self.failed: Set[Tuple[str, str]] = set()
        self.proved: Dict[Tuple[str, str], str] = {}

    def key(self, lhs: Any, rhs: Any) -> Tuple[str, str]:
        return (term_to_str(lhs) if not isinstance(lhs, str) else lhs,
                term_to_str(rhs) if not isinstance(rhs, str) else rhs)


def energy(cand: Dict[str, Any], stuck_syms: Set[str],
           ledger: TrialLedger) -> float:
    """axis_energy と同じ形: 接地の質量 ×(1 + 被覆 + 特定性)− 台帳減点。

    エネルギーは順序だけを決める。真偽は核と接地検査と Lean が決める。
    """
    syms = t_symbols(cand["lhs"]) | t_symbols(cand["rhs"])
    cover = len(syms & stuck_syms) / max(1, len(stuck_syms))
    spec = 1.0 / (t_size(cand["lhs"]) + t_size(cand["rhs"]))
    mass = 1.0 + math.log1p(cand["ground_passed"])
    e = mass * (1.0 + 2.0 * cover + 4.0 * spec)
    if ledger.key(cand["lhs"], cand["rhs"]) in ledger.failed:
        e *= 0.25
    return round(e, 6)


def cross_agenda(cands: List[Dict[str, Any]], stuck_l: Any, stuck_r: Any,
                 ledger: TrialLedger) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """候補を ShellCross の腕に置き、エネルギー降順の試行順を返す。

    十字は探索状態の置き場: 腕の tip = 候補の左辺、面 = 由来・接地数・
    右辺・エネルギー。中心 = 詰まった段。dump はトレースに残る。
    """
    stuck_syms = t_symbols(stuck_l) | t_symbols(stuck_r)
    scored = sorted(cands, key=lambda c: (-energy(c, stuck_syms, ledger),
                                          term_to_str(c["lhs"])))
    shell = ShellCross()
    dump: Dict[str, Any] = {"center": term_to_str(stuck_l) + " = "
                            + term_to_str(stuck_r), "arms": {}}
    for axis, cand in zip(AXES, scored[:len(AXES)]):
        tip = term_to_str(cand["lhs"])
        shell.faces[axis]["tip"] = tip
        facets = [("rhs", term_to_str(cand["rhs"])
                   if not isinstance(cand["rhs"], str) else cand["rhs"]),
                  ("origin", cand["origin"]),
                  ("ground", str(cand["ground_passed"])),
                  ("energy", str(energy(cand, stuck_syms, ledger)))]
        for face, (_k, v) in zip(FACET_FACES, facets):
            shell.faces[axis][face] = v
        dump["arms"][axis] = {"tip": tip, "facets": dict(facets)}
    return scored, dump


# ---------------------------------------------------------------------------
# 証明の駆動(帰納 + 発明した補題の再帰証明)
# ---------------------------------------------------------------------------
def generalise_str(t: Any) -> str:
    """変数(閉じた表)をパターン変数へ。帰納定数 k/h0 は定数のまま。"""
    def go(u: Any) -> Any:
        if not isinstance(u, tuple):
            return "?" + u if isinstance(u, str) and u in VAR_TYPES else u
        return (u[0],) + tuple(go(a) for a in u[1:])
    g = go(t)
    return term_to_str(g) if not isinstance(g, str) else g


# ---------------------------------------------------------------------------
# LPO(lexicographic path order)— 印字でなく項の構造で向きを決める(PREREG6)
# ---------------------------------------------------------------------------
#: 優先順位: 定義される側 > 定義に使われる側。
_PREC = {"rev": 9, "len": 8, "app": 7, "monus": 7, "le": 7,
         "mul": 6, "add": 5,
         "s": 3, "cons": 2, "nil": 1, "0": 0, "true": 0, "false": 0}


def _head(t: Any) -> str:
    if isinstance(t, tuple):
        return t[0]
    return str(t)


def _prec(sym: str) -> int:
    return _PREC.get(sym, 0)


def _is_var(t: Any) -> bool:
    return isinstance(t, str) and (t in VAR_TYPES or t in FRESH)


def lpo_gt(s: Any, t: Any) -> bool:
    """s >lpo t(厳密)。変数は「s に現れる変数」の場合のみ下位。"""
    if s == t:
        return False
    if _is_var(t):
        return t in _strs(s) if not _is_var(s) else False
    if _is_var(s):
        return False
    s_args = list(s[1:]) if isinstance(s, tuple) else []
    t_args = list(t[1:]) if isinstance(t, tuple) else []
    # (a) ある引数 si ≥ t
    for a in s_args:
        if a == t or lpo_gt(a, t):
            return True
    f, g = _head(s), _head(t)
    if _prec(f) > _prec(g):
        return all(lpo_gt(s, b) for b in t_args)
    if f == g:
        # (c) 引数列の辞書式比較 + s > 各 tj
        for a, b in zip(s_args, t_args):
            if a == b:
                continue
            if lpo_gt(a, b):
                return all(lpo_gt(s, c) for c in t_args)
            return False
        return False
    return False


def orient_by_lpo(l: Any, r: Any) -> Tuple[Any, Any, str]:
    """(格納すべき lhs, rhs, 種別)。種別: normal / reversed / undirected。"""
    if lpo_gt(l, r):
        return l, r, "normal"
    if lpo_gt(r, l):
        return r, l, "reversed"
    return l, r, "undirected"


def _defs_are_lpo_decreasing() -> List[str]:
    """構築時検査(PREREG6 V3): 定義10本が全て lhs >lpo rhs であること。"""
    bad = []
    for name, l, r in DEFS:
        tl, tr = parse_term(l), parse_term(r)
        # パターン変数 ?x は変数扱いにする(閉じた変換)
        def strip(u: Any) -> Any:
            if isinstance(u, str) and u.startswith("?"):
                return u[1:]
            if isinstance(u, tuple):
                return (u[0],) + tuple(strip(a) for a in u[1:])
            return u
        if not lpo_gt(strip(tl), strip(tr)):
            bad.append(name)
    return bad


def is_symmetric(l: Any, r: Any) -> bool:
    if not isinstance(l, tuple) or not isinstance(r, tuple) or l == r:
        return False
    return (l[0] == r[0] and len(l) == len(r)
            and sorted(map(str, l[1:])) == sorted(map(str, r[1:])))


def load_mathlib_context(path: Optional[Path] = None):
    """mathlib等式断片を手渡し用の文脈規則に変換する(PREREG7)。

    昇格も蓄積もしない — IH と同じ「この文脈でだけ使ってよい規則」の席。
    演算子を署名に写し(+→add, *→mul)、向きは LPO(normal/reversed は
    無条件適用、比較不能は既存の順序門つき)。返り値:
    (rules[(name,lhs,rhs)], oriented_names)。名前は ml: 接頭辞 —
    発火の引用が台帳で mathlib 由来と分かる。
    """
    import json as _json
    p = path or (Path.home() / "Projects" / "vera-corpus" / "build"
                 / "mathlib_eq_rules.json")
    if not p.exists():
        return [], []
    rows = _json.loads(p.read_text(encoding="utf-8")).get("rules", [])

    def conv(side: str) -> Optional[str]:
        t = parse_term(side)
        if t is None:
            return None
        def go(u):
            if isinstance(u, tuple):
                op = {"+": "add", "*": "mul"}.get(u[0], u[0])
                return (op,) + tuple(go(a) for a in u[1:])
            return u
        return term_to_str(go(t))

    rules, oriented = [], []
    skipped: List[str] = []
    load_mathlib_context.skipped = skipped   # 報告用(不搭載の明示)
    for r in rows:
        pl, pr = conv(r["lhs"]), conv(r["rhs"])
        if pl is None or pr is None:
            continue
        def strip(u):
            if isinstance(u, str) and u.startswith("?"):
                return u[1:] if u[1:] in VAR_TYPES else "a"
            if isinstance(u, tuple):
                return (u[0],) + tuple(strip(x) for x in u[1:])
            return u
        tl, tr = strip(parse_term(pl)), strip(parse_term(pr))
        name = "ml:" + r["name"]
        # AC冗長規則は渡さない(PREREG9)。この証明器の「一致」は既に
        # ac_norm(比較の正規化)が担っており、可換・結合・回転の族を
        # 書き換え規則としても渡すと「一致」の意味が二重化して循環する
        # (実測: comm が 297/600 発で段が予算切れ — fork 168 の再発)。
        if ac_norm(tl) == ac_norm(tr):
            skipped.append(name)
            continue
        _l, _r, kind = orient_by_lpo(tl, tr)
        if kind == "reversed":
            pl, pr = pr, pl
        rules.append((name, pl, pr))
        if kind == "undirected":
            oriented.append(name)
    return rules, oriented


def load_list_context(path: Optional[Path] = None):
    """Lean core の List 断片(build_list_rules.py の出力)を手渡しに。

    規則は既に我々の署名で、1本ずつこの機体の Lean が VERIFIED 済み
    (verified:lean4:local)。向き(LPO)と AC 冗長門は nat 側と共通。
    """
    import json as _json
    p = path or (Path.home() / "Projects" / "vera-corpus" / "build"
                 / "mathlib_list_rules.json")
    if not p.exists():
        return [], []
    rows = _json.loads(p.read_text(encoding="utf-8")).get("rules", [])
    rules, oriented = [], []
    for r in rows:
        pl, pr = r["lhs"], r["rhs"]
        def strip(u):
            if isinstance(u, str) and u.startswith("?"):
                return u[1:] if u[1:] in VAR_TYPES else "a"
            if isinstance(u, tuple):
                return (u[0],) + tuple(strip(x) for x in u[1:])
            return u
        tl, tr = strip(parse_term(pl)), strip(parse_term(pr))
        name = "ml:" + r["name"]
        if ac_norm(tl) == ac_norm(tr):
            continue
        _l, _r, kind = orient_by_lpo(tl, tr)
        if kind == "reversed":
            pl, pr = pr, pl
        rules.append((name, pl, pr))
        if kind == "undirected":
            oriented.append(name)
    return rules, oriented


class Prover:
    def __init__(self, rules: Optional[List[Tuple[str, str, str]]] = None,
                 max_depth: int = 3, wave: int = 6,
                 proof_ledger: Any = None,
                 mathlib_context: Optional[Tuple[List, List]] = None) -> None:
        self.rules = list(rules or DEFS)
        self.oriented: List[str] = []
        self.ledger = TrialLedger()
        # 永続の台帳(verantyx.proof_ledger.ProofLedger)。渡されたときだけ
        # 書く — 実験の決定論は不変(台帳は読み書きされるが票を持たない)。
        self.persist = proof_ledger
        self.max_depth = max_depth
        self.wave = wave          # 1つの詰まりで試す候補数(十字の腕数)
        # mathlib 手渡し文脈(PREREG7)。票を持たず、昇格せず、nf の文脈に
        # だけ供給される。cited に発火した mathlib 名が溜まる。
        self.ml_rules, self.ml_oriented = mathlib_context or ([], [])
        self.stats = {"nodes": 0, "invented": 0, "refuted_pruned": 0,
                      "lemmas": [], "cited": [], "cond_fired": [],
                      "cond_refused": 0}
        self.trace: List[Dict[str, Any]] = []
        # 前セッションの失敗をエネルギー減点に流し込む — プロセスを跨いだ
        # 「二度目は探索でなく参照」。proved は規則にはしない(規則昇格は
        # このプロセスで再証明されたものだけ — 台帳は記憶であって公理でない)。
        if self.persist is not None:
            for k, status in self.persist.trials.items():
                if status == "failed" and " = " in k:
                    l, r = k.split(" = ", 1)
                    self.ledger.failed.add((l, r))

    # -- 条件付き書き換え(PREREG12) ---------------------------------------
    def _cond_try(self, t: Any, depth: int, seen: Set[Tuple[str, str]],
                  fired: List[str]) -> Optional[Any]:
        """条件付き規則を一箇所だけ適用(外側優先)。条件は先に放電。

        放電: 条件の代入例が接地なら nf で決定(構成子まで落ちる)。
        変数・新鮮定数を含むなら、新鮮定数を変数化した全称形を再帰的に
        prove する — 全称形の証明は代入例の成立を含意する(強い側への
        放電、健全)。放電できなければ発火しない — 黙った仮定は無い。
        """
        for name, lp_s, rp_s, (cl_s, cr_s) in COND_RULES:
            lp, rp = parse_term(lp_s), parse_term(rp_s)
            cl, cr = parse_term(cl_s), parse_term(cr_s)
            b = match_pat(lp, t, {})
            if b is not None:
                ci_l, ci_r = subst_pat(cl, b), subst_pat(cr, b)
                if self._discharge(ci_l, ci_r, depth, seen):
                    self.stats["cond_fired"].append(name)
                    if name not in fired:
                        fired.append("cond:" + name)
                    return subst_pat(rp, b)
                self.stats["cond_refused"] += 1
        if isinstance(t, tuple):
            for i, a in enumerate(t[1:], start=1):
                r = self._cond_try(a, depth, seen, fired)
                if r is not None:
                    return t[:i] + (r,) + t[i + 1:]
        return None

    def _discharge(self, cl: Any, cr: Any, depth: int,
                   seen: Set[Tuple[str, str]]) -> bool:
        has_var = bool(t_vars(cl) + t_vars(cr))
        has_fresh = any(s in FRESH for s in _strs(cl) + _strs(cr))
        if not has_var and not has_fresh:
            a = nf(cl, self.rules, self.oriented)
            b = nf(cr, self.rules, self.oriented)
            return a is not None and b is not None and a == b
        gl, gr = _rename_fresh(cl), _rename_fresh(cr)
        ok, _how = self.prove(gl, gr, depth + 1, seen)
        return ok

    def _cond_close(self, a: Any, b: Any, depth: int,
                    seen: Set[Tuple[str, str]], extra: List,
                    fired: List[str]) -> bool:
        """正規形どうしが割れた後、条件付き書き換えで閉じるか(有界)。"""
        if not COND_RULES:
            return False
        ta, tb = a, b
        for _ in range(4):
            changed = False
            for is_a in (True, False):
                t = ta if is_a else tb
                r = self._cond_try(t, depth, seen, fired)
                if r is not None:
                    n = nf(r, self.rules + extra + self.ml_rules,
                           self.oriented + self.ml_oriented, fired=fired)
                    if n is None:
                        continue
                    if is_a:
                        ta = n
                    else:
                        tb = n
                    changed = True
            if ta == tb:
                return True
            if not changed:
                return False
        return ta == tb

    def _cite(self, fired: List[str]) -> None:
        for n in fired:
            if n.startswith("ml:") and n not in self.stats["cited"]:
                self.stats["cited"].append(n)

    # -- 補題の昇格 --------------------------------------------------------
    def _promote(self, name: str, l: Any, r: Any) -> None:
        # 確認4で採択された形(発明時のサイズ非増大門 + 対称のみ向き付き)。
        # PREREG6 は昇格・持ち越しの向きを LPO に載せ替えたが V1 違反で
        # 棄却(B9/B10 後退) — LPO は orient_by_lpo として実装済み・未採択。
        self.rules = self.rules + [(name, generalise_str(l),
                                    generalise_str(r))]
        if is_symmetric(l, r):
            self.oriented.append(name)

    # -- 中核: 目標を証明する ----------------------------------------------
    def prove(self, lhs: Any, rhs: Any, depth: int = 0,
              seen: Optional[Set[Tuple[str, str]]] = None) -> Tuple[bool, str]:
        if isinstance(lhs, str) and lhs not in VAR_TYPES and "(" in lhs:
            lhs = parse_term(lhs)
        if isinstance(rhs, str) and rhs not in VAR_TYPES and "(" in rhs:
            rhs = parse_term(rhs)
        seen = seen or set()
        key = self.ledger.key(lhs, rhs)
        if key in self.ledger.proved:
            return True, "ledger"
        if key in seen or depth > self.max_depth:
            return False, "depth"
        seen = seen | {key}
        self.stats["nodes"] += 1
        # 目標レベルの反駁門(PREREG11): 偽の目標は「証明できない」でなく
        # REFUTED(反例を名指す)。失敗(拒否)と偽を混ぜない。
        if depth == 0:
            _gc = ground_check(lhs, rhs)
            if _gc["verdict"] == "REFUTED":
                self.ledger.failed.add(key)
                if self.persist is not None:
                    self.persist.record_trial(key[0], key[1], "refuted")
                return False, "REFUTED:" + str(_gc.get("witness"))
        if t_size(lhs) + t_size(rhs) > MAX_TERM:
            self.ledger.failed.add(key)
            return False, "too_large"

        # 直接検査にも ml 文脈を供給(PREREG10 — 同じ門を全検査で)
        _fired_d: List[str] = []
        a = nf(lhs, self.rules + self.ml_rules,
               self.oriented + self.ml_oriented, fired=_fired_d)
        b = nf(rhs, self.rules + self.ml_rules,
               self.oriented + self.ml_oriented, fired=_fired_d)
        if a is not None and b is not None and a == b:
            self._cite(_fired_d)
            self.ledger.proved[key] = "direct"
            return True, "direct"
        if a is not None and b is not None and                 self._cond_close(a, b, depth, seen, [], _fired_d):
            self._cite(_fired_d)
            self.ledger.proved[key] = "direct+cond"
            return True, "direct+cond:" + "+".join(self.stats["cond_fired"])

        # 不動点: 補題が昇格したら、失敗した変数の帰納をやり直す
        # (探索実験の「不動点が補題の順序を自動で解く」の変数版)。
        vs = t_vars(lhs) + [u for u in t_vars(rhs) if u not in t_vars(lhs)]
        for _round in range(3):
            n_rules = len(self.rules)
            for v in vs:
                ok, how = self._induct(lhs, rhs, v, depth, seen)
                if ok:
                    self.ledger.proved[key] = how
                    return True, how
            if len(self.rules) == n_rules:
                break
        self.ledger.failed.add(key)
        if self.persist is not None:
            self.persist.record_trial(key[0], key[1], "failed")
        return False, "no_induction_closed"

    def _induct(self, lhs: Any, rhs: Any, v: str, depth: int,
                seen: Set[Tuple[str, str]]) -> Tuple[bool, str]:
        self.stats["nodes"] += 1
        ty = VAR_TYPES[v]
        base_val: Any = "0" if ty == "N" else "nil"
        step_val: Any = ("s", "kn") if ty == "N" else ("cons", "h0", "kl")
        k_const = "kn" if ty == "N" else "kl"

        # mathlib 手渡し(PREREG7): 基底・段の検査の文脈にだけ供給。
        # 昇格しない。発火は cited に引用として残る。
        _fired: List[str] = []

        def _ctx_nf(t, extra):
            return nf(t, self.rules + extra + self.ml_rules,
                      self.oriented + self.ml_oriented, fired=_fired)

        b0 = _ctx_nf(t_subst(lhs, {v: base_val}), [])
        b1 = _ctx_nf(t_subst(rhs, {v: base_val}), [])
        if b0 is None or b1 is None:
            return False, "base_budget"
        if b0 != b1:
            # 基底が閉じない → 基底そのものを部分目標として証明を試みる
            ok, _ = self._close_stuck(b0, b1, depth, seen)
            if not ok:
                return False, "base_open"
            b0 = _ctx_nf(t_subst(lhs, {v: base_val}), [])
            b1 = _ctx_nf(t_subst(rhs, {v: base_val}), [])
            if b0 is None or b0 != b1:
                return False, "base_open"

        ih = [("IH", generalise_str(t_subst(lhs, {v: k_const})),
               generalise_str(t_subst(rhs, {v: k_const})))]
        s0 = _ctx_nf(t_subst(lhs, {v: step_val}), ih)
        s1 = _ctx_nf(t_subst(rhs, {v: step_val}), ih)
        if s0 is not None and s1 is not None and s0 == s1:
            self._cite(_fired)
            return True, "induction on %s" % v
        if s0 is not None and s1 is not None and                 self._cond_close(s0, s1, depth, seen, ih, _fired):
            self._cite(_fired)
            return True, "induction on %s + cond" % v
        if s0 is None or s1 is None:
            return False, "step_budget"

        # 詰まった — 候補を発明し、十字に置き、エネルギー順に証明を試みる。
        # 閉じたかの検査は IH 込みで行う(段の中では IH は使ってよい規則)。
        ok, used = self._close_stuck(s0, s1, depth, seen, extra=ih)
        if not ok:
            return False, "step_open"
        s0 = _ctx_nf(t_subst(lhs, {v: step_val}), ih)
        s1 = _ctx_nf(t_subst(rhs, {v: step_val}), ih)
        if s0 is not None and s1 is not None and s0 == s1:
            self._cite(_fired)
            return True, "induction on %s + %s" % (v, used)
        return False, "step_open_after_lemmas"

    def _close_stuck(self, s0: Any, s1: Any, depth: int,
                     seen: Set[Tuple[str, str]],
                     extra: Optional[List[Tuple[str, str, str]]] = None,
                     ) -> Tuple[bool, str]:
        """詰まりを補題の発明で閉じる。エネルギーが試行順を決める。

        extra は詰まりの文脈でだけ使ってよい規則(帰納の仮定)。候補の
        証明そのものには渡さない — 補題は文脈の外でも真でなければ
        規則に昇格できない。
        """
        # 参照が発明に先行する: mathlib 文脈だけで閉じるならここで終わり
        if self.ml_rules:
            _fired0: List[str] = []
            a0 = nf(s0, self.rules + list(extra or []) + self.ml_rules,
                    self.oriented + self.ml_oriented, fired=_fired0)
            b0_ = nf(s1, self.rules + list(extra or []) + self.ml_rules,
                     self.oriented + self.ml_oriented, fired=_fired0)
            if a0 is not None and b0_ is not None and a0 == b0_:
                self._cite(_fired0)
                return True, "mathlib_context"
        cands = invent_candidates(s0, s1, self.rules)
        self.stats["invented"] += len(cands)
        ranked, dump = cross_agenda(cands, s0, s1, self.ledger)
        self.trace.append(dump)
        used: List[str] = []
        ctx = list(extra or [])
        for cand in ranked[:self.wave]:
            ok, _how = self.prove(cand["lhs"], cand["rhs"], depth + 1, seen)
            if not ok:
                continue
            name = "L%d" % len(self.rules)
            self._promote(name, cand["lhs"], cand["rhs"])
            _ls = term_to_str(cand["lhs"])
            _rs = (term_to_str(cand["rhs"])
                   if not isinstance(cand["rhs"], str) else cand["rhs"])
            self.stats["lemmas"].append("%s: %s = %s" % (name, _ls, _rs))
            if self.persist is not None:
                self.persist.add_lemma(
                    _ls, _rs, how=_how, origin_goal=dump["center"],
                    ground_passed=cand["ground_passed"])
                self.persist.record_trial(_ls, _rs, "proved")
            used.append(name)
            _fired2: List[str] = []
            a = nf(s0, self.rules + ctx + self.ml_rules,
                   self.oriented + self.ml_oriented, fired=_fired2)
            b = nf(s1, self.rules + ctx + self.ml_rules,
                   self.oriented + self.ml_oriented, fired=_fired2)
            if a is not None and b is not None and a == b:
                self._cite(_fired2)
                return True, "+".join(used)
        return (False, "") if not used else (False, "+".join(used))
