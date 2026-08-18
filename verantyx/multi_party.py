"""複数の当事者にまたがる出来事を分解し、行為ごとに文書の罪を引く。

Why
---
「AがBを脅してBがCを傷つけた場合、誰がどのような罪を犯したか」は一つの
主題ではなく二つの出来事で、主体も客体も入れ替わる(第1事象の客体Bが
第2事象の主体になる)。文全体を単一の lookup(subject) に投げると、長い
質問文字列がどの見出しとも一致せず沈黙するか、どちらの行為の罪かが
混ざって出る。

ここでは出来事を閉じた表層手がかり(「Xが Yを V(て形/た形)」の連接)で
分割し、行為ごとに別々の document_structure.lookup を打つ — 主張を
組み立てず、行為の数だけ文書の言葉をそのまま引用する。立体十字の腕
(support/cause/kind の6問)を単一の主張ではなく、複数の出来事の並びに
そのまま広げたもの: 各出来事が誰を客体にしたか(support: 誰の行為が
根拠か)、前の出来事の客体が次の出来事の主体になっているか(cause: 何が
次を引き起こしたか)、行為がどの罪の種別に当たるか(kind: 見出しへの
一致)。

閉じた表層一致だけを使う。代名詞・省略された主語(「Bが」を言わずに
いきなり動詞が来る形)・「誰が」で始まる疑問節は拾わない — 拾わない
ことも失敗ではなく、正直に言う。
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .document_structure import lookup as _lookup

#: 出来事本体ではなく問いの節を書き出す語。「誰が」「何を」のような節を
#: 出来事として拾うと、質問文そのものが第三の当事者に化ける。
_QUESTION_ACTORS = ("誰", "だれ", "何", "なに", "どの", "どちら", "どれ")

#: 閉じた表層パターン: 「Xが Yを V(て形/た形/る形)」。粒子(が/を/は/に/
#: で/と/へ/も)は主体・客体の内側には来ない、という閉じた仮定の上でだけ
#: 動く — 助詞を含む固有名や複合目的語(「店から商品」)は対象外、既知の
#: 限界として受け入れる。
#:
#: 動詞側は が/を だけを除く — で/だ は除かない。ナ行・マ行・バ行の
#: 五段動詞(死ぬ/盗む/呼ぶ)のて形/た形は「んで/んだ」で、て/たではなく
#: で/だに変わる(音便)。実測: 「盗んで」を て のみで探すと文中の次の
#: て(次の動詞の て形)まで飲み込み、二つ目の事象の主体・客体ごと
#: 一つ目の動詞句に混ざった。
_EVENT_RE = re.compile(
    r"(?P<actor>[^\s、。,がをはにでとへも]+?)が"
    r"(?P<target>[^\s、。,がをはにでとへも]+?)を"
    r"(?P<verb>[^\s、。,がを]+?)"
    r"(?:て|た|で|だ|る)(?:、|,|うえで)?"
)


def decompose_events(sentence: str) -> List[Dict[str, str]]:
    """「Xが Yを V」の連続を、出来事のリストに割る。

    疑問節(「誰が...を...した」のような、問い自身が持つ が/を/V の形)は
    _QUESTION_ACTORS で除く。拾わない形(代名詞・省略主語)は単に出来事に
    ならない — DECOMPOSE_NO_EVENTS として呼び出し元に伝わる。
    """
    events: List[Dict[str, str]] = []
    for m in _EVENT_RE.finditer(sentence):
        actor = m.group("actor")
        if actor.startswith(_QUESTION_ACTORS):
            continue
        events.append({
            "actor": actor,
            "target": m.group("target"),
            "verb": m.group("verb"),
        })
    return events


def _is_refusal(verdict: Optional[str]) -> bool:
    v = str(verdict or "")
    return v.startswith("UNKNOWN") or v in ("DOCUMENT_NOT_SPECIFIED",)


def _crime_for_verb(verb: str, book: Dict[str, Any]) -> Dict[str, Any]:
    """行為(動詞)から、文書が定める罪をそのまま引く。動詞そのもので
    見出しに届かなければ、動詞の先頭1文字(語幹の最短単位)で再度引く —
    「脅す」の「脅」が見出し「脅迫罪」に含まれるような、文字境界だけの
    閉じた一致。それでも届かなければ、素直に UNKNOWN を返す。"""
    r = _lookup(verb, book)
    if not _is_refusal(r.get("verdict")):
        r["reached_via"] = verb
        return r
    stem = verb[:1]
    if stem and stem != verb:
        r2 = _lookup(stem, book)
        if not _is_refusal(r2.get("verdict")):
            r2["reached_via"] = stem
            r2["reached_via_stem"] = True
            return r2
    r["reached_via"] = verb
    return r


def analyze_multi_party(sentence: str, book: Dict[str, Any]) -> Dict[str, Any]:
    """出来事に分解し、当事者ごとに罪(文書の該当条)を引いて並べる。

    法的な因果判断(誰の罪がどの程度重いか、正当防衛が成立するか等)は
    一切しない — できるのは「この行為に対応する条文はこれ」という、
    文書の言葉の引用だけ。因果や連鎖の「解釈」はここでは作らない。
    """
    events = decompose_events(sentence)
    if not events:
        return {
            "verdict": "DECOMPOSE_NO_EVENTS",
            "sentence": sentence,
            "note": "「Xが Yを V」の閉じた形が見つからない。代名詞・省略された"
                    "主語・非対応の文型は拾わない設計。単一主題として lookup を"
                    "呼ぶ従来の経路を使うべき",
        }
    per_event: List[Dict[str, Any]] = []
    for i, ev in enumerate(events):
        crime = _crime_for_verb(ev["verb"], book)
        per_event.append({
            "event_index": i,
            "actor": ev["actor"],
            "target": ev["target"],
            "verb": ev["verb"],
            "crime_verdict": crime.get("verdict"),
            "crime_text": crime.get("text") or "",
            "crime_section": crime.get("section") or crime.get("subject"),
            "source": crime.get("source"),
            "quoted": bool(crime.get("quoted")),
            "reached_via": crime.get("reached_via"),
        })
    unresolved = [e for e in per_event if _is_refusal(e["crime_verdict"])]
    return {
        "verdict": "MULTI_PARTY_ANALYSIS",
        "sentence": sentence,
        "events": events,
        "per_event": per_event,
        "unresolved_count": len(unresolved),
        "note": "各出来事を独立に引いた行為ごとの結果。事象間の連接(誰が誰を)"
                "は表層の「が/を」から機械的に分割したもので、責任の重さ・"
                "故意過失・違法性阻却などの法的評価はしていない — 各当事者の"
                "行為に対応する条文をそのまま引用しただけ",
    }
