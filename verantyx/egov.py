"""e-Gov 法令XML → a store you can cite from.

Japan publishes its statutes as XML through e-Gov 法令検索, with a public
API and bulk download. The XML is clean — `<Article Num>` with an
`<ArticleTitle>`, an optional `<ArticleCaption>`, and `<Sentence>` bodies —
so the work here is not parsing. It is deciding what becomes a core.

## Two attempts that did not work, and why

**Whole law as one document.** Article numbers are picked up as headings and
carried forward for a few sentences, so provisions from following articles
land on the preceding article's core. 第百八条 came back as 不動産侵奪、
不同意堕胎 — captions belonging to other articles entirely. For a citation
tool that is the worst possible failure: it points a reader at a provision
that is not the one they asked for.

**One document per article.** Attribution is then correct, but the core of
「人を殺した者は、死刑…」 is 者 — a word several hundred articles share.
The article is retrievable only through its source label, and the thing a
reader actually types (第百九十九条) is not a core at all.

## What works: reify the article

Same move as `verantyx.events` makes for a three-place fact. The article is
the entity; what it provides are its facets:

    刑法第百九十九条 → 殺人、死刑、拘禁刑、五年以上
    刑法第二百四条   → 傷害、十五年以下、拘禁刑、五十万円以下
    刑法第三十七条   → 緊急避難、危難、生命、他人

Measured on 刑法 (357 articles): the citations above are correct, and

    ask 「刑法第二百四条 殺人」 -> UNKNOWN_INSUFFICIENT_EVIDENCE

which is the point. Article 204 is 傷害, not 殺人, and the coverage gate
refuses rather than returning the article with a confident silence about
the word that was actually asked.

## What this is NOT for

`verantyx.events` must not be pointed at statute prose. Measured over 9,087
sentences of six statutes, unguarded event extraction produced 6,800
"events" — 74.8% — of which the inspected ones assigned 又 (a conjunction)
as the actor and 処 as the act. With the guard, 2.1% are read and the rest
are refused by name.

That is not a shortcoming to fix. Statutes state RULES; a fact pattern
("A, threatened by B, injured C") is not in them and never will be. Event
reification belongs on case descriptions; article reification belongs here.
"""
from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

#: Content runs shorter than this are particles and fragments; longer ones
#: are usually a whole clause the run splitter failed to break.
MIN_TERM, MAX_TERM = 2, 6

#: Terms taken from an article body. An arm shows four, and the caption
#: earns its place first, so this is the pool a placement policy chooses
#: from rather than what gets displayed.
TERMS_PER_ARTICLE = 8


def law_title(path: Path) -> str:
    """The statute's own name, for use as the core prefix."""
    root = ET.parse(Path(path)).getroot()
    for tag in ("LawTitle", "LawName"):
        el = root.find(f".//{tag}")
        if el is not None and (el.text or "").strip():
            return el.text.strip()
    return Path(path).stem


def articles(path: Path, *, law: str = "") -> List[Tuple[str, str, List[str]]]:
    """[(core, caption, terms)] for every article that has a title and a body.

    ``core`` is 法令名 + 条番号 — the string a reader types. An article
    number without its statute is not a citation key: 第一条 exists in every
    law, and five of them collapsing onto one core is a documented failure
    in this package already.
    """
    from .lang import ja_content_runs

    root = ET.parse(Path(path)).getroot()
    name = law or law_title(Path(path))
    out: List[Tuple[str, str, List[str]]] = []

    # 本則 and 附則 number their articles independently, so both contain a
    # 第一条 and they collided: 民法第一条 came back as 施行期日・公布 —
    # the supplementary provision — instead of 私権は公共の福祉に適合. The
    # main provisions are walked first and supplementary ones carry 附則 in
    # their core, which is also how a citation is written.
    seen: set = set()
    sections: List[Tuple[str, List[Any]]] = [
        ("", root.findall(".//MainProvision")),
        ("附則", root.findall(".//SupplProvision")),
    ]
    for marker, nodes in sections:
        for node in nodes:
            for art in node.iter("Article"):
                t = art.find("ArticleTitle")
                c = art.find("ArticleCaption")
                title = (t.text or "").strip() if t is not None else ""
                caption = (c.text or "").strip("（）()") if c is not None else ""
                body = "".join((s.text or "") for s in art.iter("Sentence"))
                if not (title and body):
                    continue
                core = f"{name}{marker}{title}"
                if core in seen:
                    # Several 附則 blocks (one per amendment) reuse 第一条.
                    # Numbering them keeps each retrievable instead of the
                    # last one silently overwriting the rest.
                    n = 2
                    while f"{core}の{n}" in seen:
                        n += 1
                    core = f"{core}の{n}"
                seen.add(core)
                terms = [caption] if caption else []
                terms += [r for r in ja_content_runs(body)
                          if MIN_TERM <= len(r) <= MAX_TERM][:TERMS_PER_ARTICLE]
                out.append((core, caption, list(dict.fromkeys(terms))))
    return out


def article_sentences(path: Path, *, law: str = "") -> List[str]:
    """Reified sentences an ordinary Japanese ingest can consume."""
    return [f"{core}は{term}である。"
            for core, _cap, terms in articles(Path(path), law=law)
            for term in terms]


def ingest_law(store: Any, path: Path, *, law: str = "") -> Dict[str, Any]:
    """Place one law's articles as cores. Returns what it placed.

    Goes through `ingest_documents` rather than writing crosses directly, so
    the statute gets the same provenance and gates as any other source. A
    second write path is how two readers of one corpus begin to disagree
    about what it said.
    """
    from .document_ingest import Document, ingest_documents

    name = law or law_title(Path(path))
    sentences = article_sentences(Path(path), law=name)
    if not sentences:
        return {"verdict": "UNKNOWN_UNREADABLE", "law": name, "articles": 0}
    ingest_documents(store, [Document(source=name, text="".join(sentences))])
    return {"verdict": "ANSWER", "law": name,
            "articles": len(articles(Path(path), law=name)),
            "sentences": len(sentences)}


def ingest_laws(store: Any, paths: Iterable[Path]) -> Dict[str, Any]:
    """Several laws into one store — the "one brain, many fields" shape.

    Cores carry their statute's name, so 民法第一条 and 刑法第一条 are
    different cores and stay that way however many laws are added.
    """
    placed = [ingest_law(store, Path(p)) for p in paths]
    return {"verdict": "ANSWER" if placed else "UNKNOWN_NO_INPUT",
            "laws": placed,
            "articles": sum(p.get("articles", 0) for p in placed)}
