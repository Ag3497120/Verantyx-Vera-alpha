"""What the user already settled, and whether the reply still honours it.

Not comprehension. Two mechanical checks against things somebody already
said, and a proposal to paste them back in:

    covenant   a rule the user set — required terms, forbidden terms, and
               the topic it applies to. Violated when the reply is on that
               topic and the requirement is missing or a prohibition is used
    collapse   the reply is ABOUT something the conversation already
               constrained, and shares nothing with what was said about it

Both are answers to "has the model forgotten", and neither needs to know
what anything means. A covenant is a string test with a scope; a collapse is
the `attest_llm` linkage test with the CONVERSATION as the corpus instead of
a document store.

## The output is a proposal, never a verdict on the reply

A guard that blocks is wrong here. The model may be departing from an
earlier instruction because the user just changed it, and this layer cannot
tell those apart — it sees text, not intent. So a finding carries the exact
sentence to re-inject and the turn it came from, and a human or the calling
agent decides. The failure this prevents is the silent one: a window slides,
an instruction from turn 3 falls out of it, and nothing anywhere says so.

## Why the conversation is a different corpus from the knowledge store

`attest_claim` asks whether a claim matches a body of documents. This asks
whether a reply matches THIS conversation. A reply can be perfectly true and
still have forgotten what the user asked for, and that is the failure worth
catching in a long session. Same linkage test, different store, and mixing
them would let a well-attested falsehood pass as consistency.

## What it cannot do, stated because it will be deployed

It matches strings against scopes. 「TypeScriptを使う」 is caught by naming
TypeScript required and JavaScript forbidden; it cannot infer that
prohibition from the requirement. A covenant nobody wrote down is not
checked, and a paraphrase that avoids every registered term is not caught.
The register is the contract — this reports what was registered, exactly.
"""
from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple

#: Hiragana is excluded from the CONTINUATION class on purpose. Including
#: it let one run swallow its own particles — 「プロジェクトはPythonで実装
#: されています」 came back as a single term, so no subject was ever
#: recognised and `collapse` reported every reply consistent.
_RUN = re.compile(r"[㐀-䶿一-鿿ァ-ヺー々〆A-Za-z][㐀-䶿一-鿿ァ-ヺー々〆A-Za-z0-9.+#-]*")


def terms_of(text: str) -> List[str]:
    """Content runs, latin words included — a rule often names a tool."""
    out: List[str] = []
    for m in _RUN.finditer(text or ""):
        t = m.group(0)
        if len(t) >= 2 and t not in out:
            out.append(t)
    return out


@dataclass
class Covenant:
    """One thing the user settled, in the form a machine can check."""

    name: str
    requires: List[str] = field(default_factory=list)
    forbids: List[str] = field(default_factory=list)
    #: Terms that put a reply in this covenant's scope. Empty means always,
    #: which is deliberately hard to get: an always-on rule fires on replies
    #: that were never about it, and a guard that cries every turn is turned
    #: off by the second day.
    topic: List[str] = field(default_factory=list)
    said_at_turn: int = -1
    said_by: str = "user"
    quote: str = ""
    #: 退役(2026-08-21)。削除ではない — 「もう使っていいよ」と言われた
    #: 約束は check/fading から外れるが、履歴として残る。何を約束して
    #: いたか・いつ解かれたかは台帳の一部で、消すと監査が嘘になる。
    retired: bool = False
    retired_quote: str = ""
    retired_at_turn: int = -1
    #: 隔離席(2026-08-21)。閉じた抽出規則の外(婉曲・言い換え)は
    #: 規則で追いかけない — 極性regexの実測(645/661が語彙の外)と同じ
    #: 壁。代わりに LLM が候補を propose し、candidate は shadow で
    #: 照合だけされ verdict に混ざらない。adopt して初めて執行に入る。
    #: 淘汰は門。
    status: str = "adopted"
    #: 書かれていない禁止(2026-08-21)。登録・採用の時に店の siblings を
    #: 焼き込む(check 時に店を読むと 0.04s が死ぬ)。推論由来のヒットは
    #: 字面の forbids と型を分けて報じる — 弱い主張は弱い型で。
    inferred_forbids: List[str] = field(default_factory=list)
    inferred_from: str = ""

    def in_scope(self, text: str, asked: str = "") -> bool:
        """Scope is the EXCHANGE, not the reply's wording.

        A rule about the implementation language did not fire on the reply
        「Python。」 — one word, on topic, naming no scope term. Checking the
        question that prompted it fixes exactly that case, and it is the
        right reading anyway: a covenant binds what was asked and answered,
        not the vocabulary the answer happened to use.
        """
        if not self.topic:
            return True
        return any(t in text or (asked and t in asked) for t in self.topic)

    def check(self, text: str, asked: str = "",
              store: Any = None, infer_top: int = 6) -> Optional[Dict[str, Any]]:
        """``store`` lets the covenant infer prohibitions it never listed.

        Registered by hand, a rule catches only the substitution somebody
        anticipated. Given a store, the siblings of each required term are
        the alternatives the geometry already knows about — measured over
        four legal alternative sets, 11 of 14 terms recovered another member
        of their own set, most at rank one.

        Inferred hits are reported SEPARATELY from registered ones. A
        registered prohibition is what the user said; an inferred one is
        what the corpus suggests they meant, and a reader deciding whether
        to re-inject needs to know which is which.
        """
        if self.retired:
            return None
        if not self.in_scope(text, asked):
            return None
        used = [f for f in self.forbids if loosely_in(f, text)]
        # 文字クラスの照合(2026-08-21、閉じた表・1クラスのみ)。
        # 実地試験の限界2:「絵文字を使わないで」を捕まえたのは返答中の
        # **語**「絵文字」であって 🎉 そのものではなかった。禁止語が
        # クラス名のとき、そのクラスの文字自体を検査し、見つけた文字を
        # 名指す。「日本語」クラスは入れない — 識別子の構文解析が要り、
        # 日本語の返答全部に誤発火する(過検出の番人は切られる)。
        class_hits = []
        for f in self.forbids:
            chars = _class_members_in(f, text)
            if chars:
                class_hits.append({"class": f, "found": chars[:8]})
                if f not in used:
                    used.append(f)
        # 焼き込み済みの推論禁止(登録時 siblings)。字面の used とは
        # 別の型で報じる — 登録は利用者の言葉、推論は店の示唆。
        inferred_used = [f for f in self.inferred_forbids
                         if loosely_in(f, text)]
        missing = [r for r in self.requires if not loosely_in(r, text)]
        inferred: List[Dict[str, Any]] = []
        if store is not None and missing:
            for r in missing:
                for w, s in siblings(store, r, limit=infer_top):
                    if loosely_in(w, text):
                        inferred.append({"instead_of": r, "used": w,
                                         "sibling_score": s})
        if not used and not missing and not inferred_used:
            return None
        out = {
            "covenant": self.name,
            "forbidden_used": used,
            "required_missing": missing,
            "substituted": inferred,
            "said_at_turn": self.said_at_turn,
            "said_by": self.said_by,
            # What to paste back, verbatim, rather than a paraphrase of it.
            "inject": self.quote or self.name,
        }
        if class_hits:
            out["class_hits"] = class_hits
        if inferred_used:
            out["inferred_forbidden_used"] = inferred_used
            out["inferred_from"] = self.inferred_from
        return out

    def as_dict(self) -> Dict[str, Any]:
        out = {"name": self.name, "requires": self.requires,
               "forbids": self.forbids, "topic": self.topic,
               "said_at_turn": self.said_at_turn, "said_by": self.said_by,
               "quote": self.quote}
        if self.retired:
            out["retired"] = True
            out["retired_quote"] = self.retired_quote
            out["retired_at_turn"] = self.retired_at_turn
        if self.status != "adopted":
            out["status"] = self.status
        if self.inferred_forbids:
            out["inferred_forbids"] = self.inferred_forbids
            out["inferred_from"] = self.inferred_from
        return out


@dataclass
class Register:
    """The covenants in force, the turns they came from, and their history.

    ## Which harness, and when

    Re-injecting every rule every turn is what a system prompt already does,
    and it is why long sessions drift anyway: the model has seen the rule so
    often it has stopped carrying information. What carries information is a
    rule that has JUST started being broken.

    So each covenant keeps its own record of checks and breaks, and `fading`
    reports the ones whose recent behaviour differs from their history — a
    rule kept for twenty turns and broken twice in the last three is the one
    worth spending context on. A rule never broken needs no reminder, and a
    rule broken from the beginning was probably never understood and needs
    rewriting rather than repeating.
    """

    covenants: List[Covenant] = field(default_factory=list)
    #: covenant name -> the check results, oldest first, True = kept
    history: Dict[str, List[bool]] = field(default_factory=dict)
    #: tool 実行の証人(2026-08-21)。「必ずテストして」は返答の字面では
    #: 執行できない — やったの根拠は実行の記録(attest_claim の
    #: CLAIM_UNWITNESSED と同じ線)。{"boundary": True} がターン境界で、
    #: audit は最後の境界以降だけを数える(ターンを跨いだ「やった」は
    #: 別のターンの証人)。
    witnesses: List[Dict[str, Any]] = field(default_factory=list)

    def add(self, c: Covenant) -> Covenant:
        self.covenants.append(c)
        self.history.setdefault(c.name, [])
        return c

    def fading(self, window: int = 5, min_history: int = 4) -> Dict[str, Any]:
        """Covenants whose recent compliance is worse than their own past.

        Compared against ITS OWN history, not against the other rules. A
        rule that is hard to keep and always half-kept is not degrading; a
        rule kept perfectly for twenty turns and broken twice just now is,
        and only the second one is news.
        """
        rows: List[Dict[str, Any]] = []
        for c in self.covenants:
            if c.retired:
                continue          # 解かれた約束の風化を報せても雑音
            if c.status != "adopted":
                continue          # 隔離席の候補はまだ約束ではない
            h = self.history.get(c.name) or []
            if len(h) < min_history:
                continue
            recent, past = h[-window:], h[:-window]
            if not past:
                continue
            r_keep = sum(recent) / len(recent)
            p_keep = sum(past) / len(past)
            rows.append({
                "covenant": c.name, "checks": len(h),
                "kept_before": round(p_keep, 3), "kept_recently": round(r_keep, 3),
                "delta": round(r_keep - p_keep, 3),
                "quote": c.quote,
            })
        rows.sort(key=lambda r: r["delta"])
        fade = [r for r in rows if r["delta"] < 0]
        return {
            "verdict": "FADING" if fade else "HELD",
            "window": window,
            "fading": fade,
            "stable": [r for r in rows if r["delta"] >= 0],
            "advise": ([f["quote"] for f in fade[:2]] if fade else []),
            "note": "re-inject the fading ones only; a rule repeated every "
                    "turn stops carrying information, which is how a long "
                    "session drifts with the system prompt still in place",
        }

    def check(self, text: str, asked: str = "", store: Any = None) -> Dict[str, Any]:
        """``asked`` is the turn that prompted this reply, if there was one.

        ``store`` turns on sibling inference — see `Covenant.check`.
        """
        hits = []
        advisories = []
        shadow = []
        for c in self.covenants:
            if c.retired:
                continue          # 退役済みは照合も履歴も汚さない
            h = c.check(text, asked, store=store)
            # 禁止側の証拠(字面・文字クラス・焼き込み推論・置換の実使用)
            # だけが violation。required_missing の字面欠落は advisory —
            # 「必ず〜して」を字面で違反と呼ぶと誤検知だらけになる
            # (実地実測)。行為の required は audit(証人)が見る。
            positive = bool(h) and bool(
                h.get("forbidden_used") or h.get("class_hits")
                or h.get("inferred_forbidden_used") or h.get("substituted"))
            # Only in-scope checks are recorded. Counting a turn that was
            # never about the rule as a "keep" makes every rule look healthy.
            # 履歴は advisory も破りに数える — 誤検知が問題なのは遮断で
            # あって観測ではない。「TypeScriptと書かなくなってきた」は
            # 風化として報せる価値がある(遮断はしない、が線)。
            if c.in_scope(text, asked):
                self.history.setdefault(c.name, []).append(h is None)
            if h:
                if c.status != "adopted":
                    # 候補は shadow に分離 — 遮断の材料にはならず、
                    # adopt するかを決める実績になるだけ(淘汰は門)。
                    shadow.append(h)
                elif positive:
                    hits.append(h)
                else:
                    advisories.append(h)
        out = {
            # BROKEN is a finding about the REPLY, never about the user. The
            # user may have changed their mind one turn ago and this layer
            # cannot see intent, only text.
            "verdict": "BROKEN" if hits else "KEPT",
            "in_force": len([c for c in self.covenants
                             if not c.retired and c.status == "adopted"]),
            "violations": hits,
            "note": "a proposal to re-inject, not a judgment; the rule may "
                    "have been superseded and this cannot tell",
        }
        if advisories:
            out["advisories"] = advisories
        if shadow:
            out["shadow_violations"] = shadow
        return out

    def witness(self, tool: str, detail: str = "",
                turn: int = -1) -> Dict[str, Any]:
        """tool 実行を1件記録する。判定はしない — 置くだけ。"""
        row = {"tool": str(tool)[:80], "detail": str(detail)[:400],
               "turn": turn, "ts": time.time()}
        self.witnesses.append(row)
        return {"verdict": "ANSWER", "witnesses": len(self.witnesses)}

    def boundary(self, turn: int = -1) -> Dict[str, Any]:
        """ターン境界。audit はこれ以降の証人だけを数える。"""
        self.witnesses.append({"boundary": True, "turn": turn,
                               "ts": time.time()})
        # 境界より前は監査に使わないので落とす(台帳の肥大を防ぐ。
        # 履歴と違い、証人は「このターンにやったか」にしか使えない)。
        for i in range(len(self.witnesses) - 1, -1, -1):
            if self.witnesses[i].get("boundary") and i < len(self.witnesses) - 1:
                self.witnesses = self.witnesses[i + 1:]
                break
        return {"verdict": "ANSWER", "witnesses": len(self.witnesses)}

    def audit(self) -> Dict[str, Any]:
        """required 側を証人で見る — 字面では見ない。

        「必ずテストを実行して」を返答の字面で執行すると誤検知だらけに
        なる(実地試験の実測)。やったの根拠は tool 実行の記録だけ。
        照合は部分文字列(requires の語が tool 名か detail に現れる)で、
        意味の一致は主張しない。**遮断はしない** — このターンにその
        実行が要ったかは文脈で、この層には見えない。報せるだけ。
        """
        # 最後の境界以降の証人だけ
        recent = []
        for w in self.witnesses:
            if w.get("boundary"):
                recent = []
            else:
                recent.append(w)
        rows = []
        for c in self.covenants:
            if c.retired or c.status != "adopted" or not c.requires:
                continue
            for r in c.requires:
                lo = r.lower()
                hit = next((w for w in recent
                            if lo in (w.get("tool", "") + " "
                                      + w.get("detail", "")).lower()), None)
                rows.append({"covenant": c.name, "requires": r,
                             "witnessed": hit is not None,
                             "witness": hit, "inject": c.quote or c.name})
        if not rows:
            verdict = "NO_REQUIREMENTS"
        elif all(r["witnessed"] for r in rows):
            verdict = "REQUIRED_WITNESSED"
        else:
            verdict = "REQUIRED_UNWITNESSED"
        return {"verdict": verdict, "rows": rows,
                "witnesses_this_turn": len(recent),
                "note": "advisory only: whether this turn NEEDED the "
                        "execution is context this layer cannot see"}

    def propose(self, c: Covenant) -> Covenant:
        """隔離席に置く。shadow で照合はされるが執行はされない。"""
        c.status = "candidate"
        return self.add(c)

    def adopt(self, name: str) -> Optional[Dict[str, Any]]:
        """候補を採用して執行に入れる。門はここ。"""
        for c in self.covenants:
            if c.name == name and c.status == "candidate" and not c.retired:
                c.status = "adopted"
                return c.as_dict()
        return None

    def retire(self, name: str, quote: str = "",
               turn: int = -1) -> Optional[Dict[str, Any]]:
        """約束を退役させる — 削除ではない。

        実地試験の限界1: 「もう絵文字使っていいよ」と言っても番人が
        止め続けた。破棄経路が要る。ただし削除すると「かつて約束が
        あった」履歴ごと消える — 風化の測定も provenance も嘘になる。
        だから席は残し、check/fading から外れるだけにする(閉鎖は追記、
        は GAP 台帳と同じ線)。同名は最初の未退役だけを退役させる。
        """
        for c in self.covenants:
            if c.name == name and not c.retired:
                c.retired = True
                c.retired_quote = quote
                c.retired_at_turn = turn
                return c.as_dict()
        return None

    def save(self, path: Path) -> Dict[str, Any]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        # 履歴も一緒に置く — 凍結バイナリの1回呼び(guard CLI)では
        # プロセスが毎回死ぬので、履歴が残らないと風化が測れない。
        Path(path).write_text(
            json.dumps({"covenants": [c.as_dict() for c in self.covenants],
                        "history": self.history,
                        "witnesses": self.witnesses},
                       ensure_ascii=False), encoding="utf-8")
        return {"verdict": "ANSWER", "path": str(path),
                "covenants": len(self.covenants)}

    @classmethod
    def load(cls, path: Path) -> "Register":
        r = cls()
        p = Path(path)
        if p.is_file():
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, list):        # 旧形式(素のリスト)も読む
                rows, hist = data, {}
            else:
                rows = data.get("covenants", [])
                hist = data.get("history", {})
                r.witnesses = list(data.get("witnesses", []))
            for d in rows:
                r.covenants.append(Covenant(**d))
            r.history = {k: list(v) for k, v in hist.items()}
        return r


def bake_inferred(c: "Covenant", store: Any, *, limit: int = 6,
                  store_name: str = "") -> Dict[str, Any]:
    """登録・採用の時に一度だけ店を読み、書かれていない禁止を焼き込む。

    check 時に店を読むと速い道(0.04s)が死ぬ。siblings は店の幾何から
    「同じ席を占める語」を返す — TypeScript を使う、と登録された約束が
    JavaScript を捕るのは、誰かが JavaScript を挙げたからではなく、
    同じ核たちが両方を抱えているから。店に姉妹語が無ければ空のまま
    (推測しない)。provenance に店の名を残す。
    """
    found: List[str] = []
    for f in list(c.forbids) + list(c.requires):
        for w, _score in siblings(store, f, limit=limit):
            if w not in found and w not in c.forbids:
                found.append(w)
    c.inferred_forbids = found
    c.inferred_from = store_name or "store"
    return {"verdict": "ANSWER", "inferred_forbids": found,
            "inferred_from": c.inferred_from}


def siblings(store: Any, term: str, *, limit: int = 24,
             min_shared: int = 2, max_fanout: int = 60,
             max_common: float = 0.02) -> List[Tuple[str, float]]:
    """Terms that occupy the same slot as ``term``, by shared parents.

    Structural, not similar. Two terms are siblings here when the same cores
    hold both — 拘禁刑 and 罰金 are both facets of the articles that set a
    penalty, 過失 and 故意 of the ones that turn on intent. That is a fact
    about the store's geometry: no embedding, no nearest neighbour, no
    notion of meaning.

    It is what lets a covenant infer its own prohibitions. Registered by
    hand, 「TypeScriptを使う」 catches JavaScript only because somebody
    listed JavaScript, and the substitution nobody anticipated goes through.

    ## Raw co-occurrence returns hubs, and hubs are not siblings

    Unweighted, 拘禁刑 came back beside 法学, 百科, 日本, 規定 — the domain
    labels and the words every article uses. Two corrections, both the same
    idea `hierarchy.distinctive_terms` already applies:

      max_fanout   a parent holding hundreds of facets witnesses nothing;
                   a sentence that names a choice is small
      max_common   a term that is a facet of more than this share of all
                   cores is furniture, whatever it co-occurs with

    Scored by 1/fanout summed over shared parents, so agreement between two
    narrow articles outweighs agreement between two indexes.
    """
    labels = getattr(store, "source_labels", set()) or set()
    common = _too_common(store, max_common)
    # The store lowercases latin and a covenant is registered the way the
    # user wrote it, so 「TypeScript」 found no parents at all under
    # 実装言語 -> typescript. Everything downstream then worked perfectly on
    # an empty list.
    low = term.lower()
    parents = [(c, len(cr or ())) for c, cr in store.crosses.items()
               if c not in labels
               and (term in (cr or ()) or low in {f.lower() for f in (cr or ())})]
    parents = [(c, n) for c, n in parents if 0 < n <= max_fanout]
    if not parents:
        return []
    need = min(min_shared, len(parents))
    from collections import Counter
    score: Counter = Counter()
    seen: Counter = Counter()
    for c, n in parents:
        for f in (store.crosses.get(c) or ()):
            if f == term or f.lower() == low or f in labels or f in common:
                continue
            score[f] += 1.0 / n
            seen[f] += 1
    out = [(w, round(s, 4)) for w, s in score.most_common(limit * 3)
           if seen[w] >= need]
    return out[:limit]


_COMMON_CACHE: Dict[Tuple[int, float], Set[str]] = {}


#: Below this many cores, a SHARE threshold means nothing: on four cores a
#: facet in one of them is already 25%, so every candidate is furniture and
#: the sibling list comes back empty. Measured the hard way — the fixture
#: fork returned no siblings at all while the 54,244-core federation
#: returned 罰金 first.
_MIN_CORES_FOR_SHARE = 200


def _too_common(store: Any, share: float) -> Set[str]:
    """Facets of more than ``share`` of all cores. Furniture, not evidence."""
    key = (id(store), share)
    if key in _COMMON_CACHE:
        return _COMMON_CACHE[key]
    if len(store.crosses) < _MIN_CORES_FOR_SHARE:
        _COMMON_CACHE[key] = set()
        return _COMMON_CACHE[key]
    from collections import Counter
    labels = getattr(store, "source_labels", set()) or set()
    tally: Counter = Counter()
    for cross in store.crosses.values():
        for f in cross or ():
            if f not in labels:
                tally[f] += 1
    n = max(len(store.crosses), 1)
    _COMMON_CACHE[key] = {w for w, c in tally.items() if c / n > share}
    return _COMMON_CACHE[key]


def infer_forbidden(store: Any, required: Sequence[str], *,
                    limit: int = 8) -> Dict[str, List[str]]:
    """For each required term, the siblings that would substitute for it."""
    return {r: [w for w, _s in siblings(store, r)[:limit]] for r in required}


#: 文字クラスの閉じた表(2026-08-21)。1クラスのみ — 絵文字は
#: コードポイントで曖昧さなく判定できる唯一のクラス。表を増やすときは
#: 「そのクラスの検出が返答の言語に依存しないか」を先に測ること。
_CLASS_TERMS = {"絵文字", "emoji", "emojis"}
_EMOJI = re.compile(
    "["
    "\U0001F300-\U0001FAFF"   # Misc symbols/pictographs .. symbols ext
    "\U00002600-\U000027BF"   # Misc symbols / dingbats
    "\U0001F1E6-\U0001F1FF"   # regional indicators
    "\U00002B00-\U00002BFF"   # arrows/stars incl ⭐
    "\uFE0F"                    # variation selector (emoji presentation)
    "]")


def _class_members_in(term: str, text: str) -> List[str]:
    """禁止語がクラス名なら、クラスの文字そのものを探して名指す。"""
    if str(term).strip().lower() not in _CLASS_TERMS:
        return []
    seen: List[str] = []
    for ch in _EMOJI.findall(text or ""):
        if ch not in seen:
            seen.append(ch)
    return seen


def loosely_in(term: str, text: str) -> bool:
    """Is the term present, allowing the word-forms the corpus writes?

    Exact substring misses 傷害罪 in a reply that wrote 傷害, which is the
    same one-character gap that made 「殺人罪の刑は」 unanswerable until the
    judgment was quantized. `ja_morph.variants` is the machinery already
    measured for it — 98.7% of 500 morphological variants reached the
    original core.

    Latin is compared case-insensitively because the store lowercases it and
    a reply does not. Without this the inference found javascript as a
    sibling of typescript and then failed to see 「JavaScript」 in the reply
    it was checking — the whole chain worked except the last comparison.
    """
    if term in text:
        return True
    if any(c.isascii() and c.isalpha() for c in term):
        if term.lower() in text.lower():
            return True
    try:
        from .ja_morph import variants
    except Exception:
        return False
    return any(v and v in text for v in variants(term, add=True, split=False))


#: Below this share of a reply's terms linked to what the conversation said
#: about the same subject, the reply has drifted off what was established.
#: The same floor `attest_llm` measured, for the same reason — it separated
#: grounded from free generation at 64.1% against 6.4%.
LINK_FLOOR = 0.30


def collapse(
    conversation: Any,
    reply: str,
    *,
    subjects: Optional[Sequence[str]] = None,
    floor: float = LINK_FLOOR,
) -> Dict[str, Any]:
    """Subjects the reply addresses that the conversation settled otherwise.

    ``conversation`` is a `conversation.Conversation`. For each subject the
    reply and the conversation share, the reply's terms are checked against
    what the CONVERSATION recorded under that subject — so a reply that has
    quietly reverted to generic knowledge about a subject the user already
    pinned down comes back with the turn to re-inject.
    """
    mem = getattr(conversation, "memory", None)
    levels = list(getattr(mem, "levels", []) or [])
    if not levels:
        return {"verdict": "UNKNOWN_NO_CONVERSATION", "subjects": 0}

    # Every layer, not just the top one. A frozen layer still holds what was
    # said — that is the whole reason overflow freezes instead of dropping —
    # and checking only level 0 would report the model as consistent with a
    # conversation whose relevant turn has aged out of it.
    labels: Set[str] = set()
    merged: Dict[str, Set[str]] = {}
    for lv in levels:
        labels |= getattr(lv, "source_labels", set()) or set()
        for c, cr in lv.crosses.items():
            merged.setdefault(c, set()).update(cr or ())

    rterms = terms_of(reply)
    # Case-fold for latin: the store ingests TypeScript as typescript, and a
    # reply that names it exactly must still match what the user set.
    fold = {c.lower(): c for c in merged}
    cand = list(subjects) if subjects else [
        fold[t.lower()] for t in rterms if t.lower() in fold]

    found: List[Dict[str, Any]] = []
    for s in cand:
        cross = {f for f in merged.get(s, ()) if f not in labels}
        if not cross:
            continue
        others = [t for t in rterms if t != s]
        if not others:
            continue
        lower = {f.lower() for f in cross}
        linked = [t for t in others if t in cross or t.lower() in lower]
        share = len(linked) / len(others)
        if share >= floor:
            continue
        loc = conversation.locate(s)
        mentions = loc.get("mentions", [])[:4]
        # Inject the TURNS, verbatim. Rebuilding a sentence from the facets
        # gave 「プロジェクト: 使い」 — the ingest keeps what a fact is about,
        # not how it was put, so anything reassembled from a cross is a
        # paraphrase this module has no business writing. The turn text is
        # what the user actually said, and it is what should go back in.
        turns = getattr(conversation, "turns", []) or []
        quotes = [turns[m["turn"]].text for m in mentions
                  if isinstance(m.get("turn"), int) and 0 <= m["turn"] < len(turns)]
        found.append({
            "subject": s,
            "link": round(share, 3),
            "conversation_said": sorted(cross)[:10],
            "reply_used": others[:10],
            "status": loc.get("status"),
            "mentions": mentions,
            "inject": quotes or [f"{s}: " + "、".join(sorted(cross)[:10])],
        })
    return {
        "verdict": "DRIFTED" if found else "CONSISTENT",
        "subjects": len(cand),
        "drifted": found,
        "floor": floor,
        "note": "the reply is about something this conversation already "
                "settled and does not use what was settled; re-injecting is "
                "a suggestion, not a correction",
    }


# ---------------------------------------------------------------------------
# 指示文 → 約束の抽出(2026-08-21、実地試験の限界3・4を受けて器官化)
# ---------------------------------------------------------------------------
# フック側に散在していた字面規則を一元化する。閉じた規則のみ:
# ここに LLM を挟むと、逸れる側の装置に約束の抽出を任せることになる
# (実装者の言葉のまま)。読めない文からは立てない — 推測しない。
#: 日本語: 「Xを使わないで」型。禁止の対象は印の直前の「を/は」句の名詞
#: (文の内容語を丸ごと拾うと誤遮断する — 実地試験で実測済みの修理)。
# 捕獲は**名詞連のみ**(ひらがなを許すと直前の助詞 では/に を巻き込む —
# 「では絵文字」を禁止語にした実測がある)。
_JA_FORBID = re.compile(
    r"([一-龥ァ-ヺa-zA-Z0-9ー]+)(?:を|は)"
    r"(?:使わないで|入れないで|書かないで|やめて|禁止|使用しない)")
_JA_REQUIRE = re.compile(
    r"必ず([一-龥ァ-ヺa-zA-Zぁ-ん0-9ー]+?)(?:を)?"
    r"(?:して|すること|実行して|実行すること)")
#: 英語(実地試験: "Never use emojis" からは約束が立たない、の修理)。
_EN_FORBID = re.compile(
    r"\b(?:never use|don't use|do not use|stop using|avoid using|no)\s+"
    r"([A-Za-z][A-Za-z0-9_-]*)", re.I)
_EN_REQUIRE = re.compile(
    r"\b(?:always|make sure to|be sure to)\s+"
    r"(?:run|use|execute)\s+([A-Za-z][A-Za-z0-9_.-]*)", re.I)
#: 解除(2026-08-21)。実地試験の限界1は「取り下げられない」だった —
#: 破棄経路(retire)は器官にあるが、**何が解除の言葉か**もフックに
#: 散らさず器官の閉じた表に置く。ja+en 対称。読めない解除は解除しない。
_JA_RELEASE = re.compile(
    r"もう([一-龥ァ-ヺa-zA-Z0-9ー]+)(?:を)?使っていい"
    r"|([一-龥ァ-ヺa-zA-Z0-9ー]+)の禁止(?:を)?解除"
    r"|([一-龥ァ-ヺa-zA-Z0-9ー]+)(?:を)?解禁")
_EN_RELEASE = re.compile(
    r"\byou can use\s+([A-Za-z][A-Za-z0-9_-]*)\s+again"
    r"|\b([A-Za-z][A-Za-z0-9_-]*)\s+is\s+(?:fine|okay|ok|allowed)\s+"
    r"(?:now|again)"
    r"|\bgo ahead and use\s+([A-Za-z][A-Za-z0-9_-]*)"
    r"|\blift(?:ed)?\s+the\s+ban\s+on\s+([A-Za-z][A-Za-z0-9_-]*)", re.I)


def extract_releases(text: str) -> List[str]:
    """解除の言葉から解かれる対象語を拾う。読めなければ空(推測しない)。"""
    out: List[str] = []
    for pat in (_JA_RELEASE, _EN_RELEASE):
        for m in pat.finditer(text or ""):
            term = next((g for g in m.groups() if g), None)
            if term and term not in out:
                out.append(term)
    return out


def extract_covenants(text: str, turn: int = -1) -> List[Dict[str, Any]]:
    """指示文から約束の候補を立てる。読めない文からは立てない。

    返すのは**候補**であり、登録は呼び出し側(set_covenant)の仕事 —
    抽出と登録を分けておくと、フックが候補を人に見せてから登録する
    運用も選べる。quote には元の文をそのまま入れる(言い換えない)。
    """
    out: List[Dict[str, Any]] = []
    for sent in re.split(r"[。\n.!?]", text or ""):
        sent = sent.strip()
        if not sent:
            continue
        forbids = [m.group(1) for m in _JA_FORBID.finditer(sent)]
        forbids += [m.group(1) for m in _EN_FORBID.finditer(sent)]
        requires = [m.group(1) for m in _JA_REQUIRE.finditer(sent)]
        requires += [m.group(1) for m in _EN_REQUIRE.finditer(sent)]
        if not forbids and not requires:
            continue
        out.append({"name": sent[:40], "requires": requires,
                    "forbids": forbids, "quote": sent,
                    "said_at_turn": turn})
    return out
