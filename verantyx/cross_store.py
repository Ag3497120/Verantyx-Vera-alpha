"""CrossStore — core ごとに1つの十字を累積する多十字ネットワークの土台.

大量投入の欠落部品: "apple" が 1000 文に出たとき facets を **1つの apple
十字に累積・重み付け** する。殻 (6腕) の spill/上書きではなく、
core → {facet: count} を無制限に蓄え、読み出し時に上位 facet を
決定論で腕に載せる。

  add(core, facts)         → counts 累積 (mass = 出現回数)
  top_facets(core, k)      → count 降順・同数はアルファベット順 (決定論)
  ingest_rows(rows)        → 文分割 → classify_sentence → 累積
  save / load              → JSON checkpoint (途中再開可)

LM なし。retrieval + consensus 接続は consensus_store.py。
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

from .en_decompose import classify_sentence, split_sentences
from .lex_filters import (
    is_junk_core,
    is_junk_facet,
    proper_lexicon_from_stats,
    proper_runs,
    sense_key,
    update_cap_stats,
)


@dataclass
class CrossStore:
    # core → facet → count
    crosses: Dict[str, Dict[str, int]] = field(default_factory=dict)
    # core → 出現回数 (質量の元; facet ゼロ文でもカウント)
    core_count: Dict[str, int] = field(default_factory=dict)
    n_sentences: int = 0
    n_rows: int = 0
    source: str = ""
    # word → [n_cap_mid_sentence, n_lower]; 文頭大文字の二段判定に使う
    cap_stats: Dict[str, List[int]] = field(default_factory=dict)
    proper_lexicon: set = field(default_factory=set)

    # ------------------------------------------------------------------
    # accumulate
    # ------------------------------------------------------------------
    def add(self, core: str, facts: Iterable[str]) -> None:
        core = str(core).casefold().strip()
        if not core:
            return
        cross = self.crosses.setdefault(core, {})
        self.core_count[core] = self.core_count.get(core, 0) + 1
        for f in facts:
            f = str(f).casefold().strip()
            if not f or f == core:
                continue
            cross[f] = cross.get(f, 0) + 1

    def ingest_sentence(self, text: str) -> Optional[str]:
        unit = classify_sentence(text)
        if unit.core is None or is_junk_core(unit.core):
            return None
        runs = proper_runs(text)
        key, run_rest = sense_key(
            unit.core, text, runs, proper_lexicon=self.proper_lexicon
        )
        drop = set(run_rest)
        # 他の proper run に属する facts はその複合語に合流させる
        facts: List[str] = []
        for f in unit.facts:
            if f in drop or is_junk_facet(f):
                continue
            merged = None
            for run in runs:
                if f in run and unit.core not in run:
                    merged = "_".join(run) + "#p"
                    break
            facts.append(merged or f)
        self.add(key, dict.fromkeys(facts))  # preserve order, dedupe
        self.n_sentences += 1
        return key

    def scan_cap_stats(self, rows: Iterable[str]) -> int:
        """Pass 1: 大文字統計だけ累積し proper_lexicon を再構築."""
        n = 0
        for row in rows:
            for sent in split_sentences(row):
                update_cap_stats(self.cap_stats, sent)
                n += 1
        self.proper_lexicon = proper_lexicon_from_stats(self.cap_stats)
        return n

    def ingest_rows(
        self,
        rows: Iterable[str],
        *,
        max_sentences: Optional[int] = None,
        checkpoint_path: Optional[Path] = None,
        checkpoint_every: int = 5000,
    ) -> Dict[str, Any]:
        """Stream rows → sentences → accumulate. No full materialization."""
        done = 0
        for row in rows:
            self.n_rows += 1
            for sent in split_sentences(row):
                if max_sentences is not None and done >= max_sentences:
                    return self.report()
                if self.ingest_sentence(sent) is not None:
                    done += 1
                    if (
                        checkpoint_path is not None
                        and done % checkpoint_every == 0
                    ):
                        self.save(checkpoint_path)
        return self.report()

    # ------------------------------------------------------------------
    # read out
    # ------------------------------------------------------------------
    def top_facets(self, core: str, k: int = 4) -> List[Tuple[str, int]]:
        cross = self.crosses.get(str(core).casefold().strip())
        if not cross:
            return []
        items = sorted(cross.items(), key=lambda kv: (-kv[1], kv[0]))
        return items[:k]

    def mass(self, core: str) -> float:
        return float(self.core_count.get(str(core).casefold().strip(), 0))

    def has(self, core: str) -> bool:
        return str(core).casefold().strip() in self.crosses

    def n_cores(self) -> int:
        return len(self.crosses)

    def n_facet_links(self) -> int:
        return sum(len(c) for c in self.crosses.values())

    def report(self) -> Dict[str, Any]:
        return {
            "n_cores": self.n_cores(),
            "n_facet_links": self.n_facet_links(),
            "n_sentences": self.n_sentences,
            "n_rows": self.n_rows,
            "source": self.source,
        }

    # ------------------------------------------------------------------
    # checkpoint
    # ------------------------------------------------------------------
    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "crosses": self.crosses,
            "core_count": self.core_count,
            "n_sentences": self.n_sentences,
            "n_rows": self.n_rows,
            "source": self.source,
            "cap_stats": self.cap_stats,
        }
        path.write_text(json.dumps(payload, ensure_ascii=False))

    @classmethod
    def load(cls, path: Path) -> "CrossStore":
        d = json.loads(Path(path).read_text())
        st = cls(
            crosses={k: dict(v) for k, v in d.get("crosses", {}).items()},
            core_count=dict(d.get("core_count", {})),
            n_sentences=int(d.get("n_sentences", 0)),
            n_rows=int(d.get("n_rows", 0)),
            source=str(d.get("source", "")),
            cap_stats={k: list(v) for k, v in d.get("cap_stats", {}).items()},
        )
        st.proper_lexicon = proper_lexicon_from_stats(st.cap_stats)
        return st


def pour_corpus(
    *,
    source: str = "auto",
    max_rows: Optional[int] = 2000,
    max_sentences: Optional[int] = None,
    checkpoint_path: Optional[Path] = None,
    store: Optional[CrossStore] = None,
    two_pass: bool = True,
    checkpoint_every: int = 5000,
) -> Tuple[CrossStore, Dict[str, Any]]:
    """corpus_en stream → CrossStore accumulate (resumable via ``store``).

    two_pass: pass 1 で大文字統計 → proper_lexicon 構築 → pass 2 で投入
    (文頭大文字の二段判定)。イテレータは source から作り直す。
    """
    from .corpus_en import iter_english_rows

    st = store or CrossStore()
    if two_pass:
        rows1, _tag, _fb, _meta = iter_english_rows(
            source=source, max_rows=max_rows
        )
        st.scan_cap_stats(rows1)

    rows, tag, fallback, meta = iter_english_rows(source=source, max_rows=max_rows)
    st.source = tag
    rep = st.ingest_rows(
        rows,
        max_sentences=max_sentences,
        checkpoint_path=checkpoint_path,
        checkpoint_every=checkpoint_every,
    )
    rep.update(
        {
            "fallback_reason": fallback,
            "meta": meta,
            "two_pass": two_pass,
            "proper_lexicon_size": len(st.proper_lexicon),
        }
    )
    if checkpoint_path is not None:
        st.save(checkpoint_path)
    return st, rep
