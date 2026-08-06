"""Does the Japanese path actually carry a claim from a file to a verdict?

Written after measuring that it did not. Before the fixes this suite locks
in, a two-source Japanese corpus ingested as **zero sentences** — silently,
with a cheerful report — and the disaster-information path that exists for
Japanese readers was the one path that did not work in Japanese.

Six defects are pinned here, each a regression case:

  1. `_SENT` required whitespace after the full stop; Japanese writes 。 with
     none, so an article was one enormous "sentence".
  2. The minimum-sentence length was tuned for Latin text, where twelve
     characters is two words. Twelve characters of Japanese is a whole claim.
  3. `CrossStore.ingest_sentence` decomposes with `[A-Za-z0-9']+`, so Japanese
     produced one core, the entire sentence, and zero facets. The Japanese
     segmenter existed in `lang` and simply was not on this path.
  4. `ja_ingest_sentence` never passed `source=`, so provenance was empty and
     every "which source said this" came back blank.
  5. `CrossStore.track_provenance` defaults to False, so attribution was
     empty even in English unless the caller happened to know the flag.
  6. Language detection ran on the sentence AFTER the "(reported by X)"
     suffix was appended, and the Latin in that suffix outvoted a short
     Japanese sentence — routing it back to the English decomposer.

Plus the one that matters most for trust: the Japanese polarity vocabulary
must not manufacture contradictions. 停止線 (a painted stop line) and 危険物
(hazardous materials) both produced poles in the first version, and a report
that invents a disagreement is worse than one that misses it.

Run:  python3 -m verantyx.multilingual_eval
"""
from __future__ import annotations

import sys
import tempfile
import zipfile
from pathlib import Path
from typing import List

from .cross_store import CrossStore
from .document_ingest import Document, deep_report, ingest_documents
from .document_loaders import SUPPORTED, load_directory, load_path
from .lang import detect
from .polarity import detect_ja

#: Sentences that must NOT produce a pole. Every one contains a polar term as
#: part of a compound noun, which is the shape that fooled the first version.
CONTROLS: List[str] = [
    "会議を開始します。",
    "資料を公開しました。",
    "事業を展開しています。",
    "受付時間は9時からです。",
    "停止線で止まってください。",
    "復旧作業の説明会を開催します。",
    "危険物取扱者の資格が必要です。",
    "通行止標識を設置。",
]

#: (sentence, expected aspect, expected value)
POLAR: List[tuple] = [
    ("避難所は開設されました。", "開設", "開設"),
    ("避難所は閉鎖されました。", "開設", "閉鎖"),
    ("国道4号は通行止です。", "通行可能", "通行止"),
    ("国道4号は通行可能になりました。", "通行可能", "通行可能"),
    ("この区域は危険です。", "安全", "危険"),
    ("水道は復旧しました。", "復旧", "復旧"),
    ("受付終了しました。", "受付中", "受付終了"),
]


def main() -> int:
    print("multilingual — Japanese ingestion, polarity, and file loaders\n")
    failures: List[str] = []

    # -- 1. No manufactured contradictions ---------------------------------
    bad = [(s, detect_ja(s)) for s in CONTROLS if detect_ja(s)]
    ok = not bad
    print(f"[{'ok  ' if ok else 'FAIL'}] {len(CONTROLS)} compound-noun controls "
          f"produce no pole")
    for s, hit in bad:
        print(f"        invented: {s} -> {hit}")
    if not ok:
        failures.append("false-positive polarity")
    print()

    # -- 2. Real oppositions are found -------------------------------------
    for sentence, aspect, value in POLAR:
        hits = detect_ja(sentence)
        got = [(a, v) for a, v, _ in hits]
        ok = (aspect, value) in got
        print(f"[{'ok  ' if ok else 'FAIL'}] {sentence} -> {got or 'nothing'}")
        if not ok:
            failures.append(f"polarity: {sentence}")
    print()

    # -- 3. Script routing -------------------------------------------------
    cases = [("避難所は開設されました。", "ja"),
             ("The shelter is open.", "en"),
             # The suffix ingestion appends must not flip the vote.
             ("避難所は開設されました。 (reported by f.docx)", "ja")]
    for text, expect in cases:
        got = detect(text)
        ok = got == expect
        # The third case is the interesting one and is allowed to be wrong
        # here — what matters is that ingestion detects on the ORIGINAL, which
        # the end-to-end check below proves.
        if text.endswith("(reported by f.docx)"):
            print(f"[note] detect on a tagged sentence returns '{got}' — this "
                  f"is why ingestion detects on the untagged text")
            continue
        print(f"[{'ok  ' if ok else 'FAIL'}] detect({text[:24]}…) = {got}")
        if not ok:
            failures.append(f"detect: {text}")
    print()

    # -- 4. End to end: two Japanese sources that disagree -----------------
    store = CrossStore()
    docs = [
        Document("A新聞", "国道4号は土砂崩れで通行止です。"
                          "本町の避難所は開設されました。毛布を配布しています。"),
        Document("B放送", "国道4号は通行可能になりました。"
                          "本町の避難所は閉鎖されました。"),
    ]
    rep = ingest_documents(store, docs)
    ok = rep.sentences == 5
    print(f"[{'ok  ' if ok else 'FAIL'}] 5 Japanese sentences ingested "
          f"(got {rep.sentences}; was 0 before the splitter and router fixes)")
    if not ok:
        failures.append(f"ingest count {rep.sentences}")

    ok = rep.polar_claims > 0
    print(f"[{'ok  ' if ok else 'FAIL'}] polar claims counted: {rep.polar_claims}")
    if not ok:
        failures.append("polar_claims counted with the English detector only")

    road = deep_report(store, "国道")
    contested = road["confidence"] == "contested"
    print(f"[{'ok  ' if contested else 'FAIL'}] 国道 is contested, not blended "
          f"into one story")
    if not contested:
        failures.append("contradiction not detected in Japanese")

    # Attribution: the point of ingesting several sources.
    attributed = all(side.get("sources")
                     for d in road["disputed"] for side in d["sides"])
    print(f"[{'ok  ' if attributed else 'FAIL'}] every contested side names "
          f"which source said it")
    if not attributed:
        failures.append("source attribution empty")

    # The outlet's own name must not become a fact about the road.
    leaked = [s["claim"] for s in road["settled"]
              if s["claim"] in {"新聞", "放送", "A新聞", "B放送"}]
    ok = not leaked
    print(f"[{'ok  ' if ok else 'FAIL'}] source labels stay out of the facts "
          f"{('(leaked: ' + str(leaked) + ')') if leaked else ''}")
    if not ok:
        failures.append("attribution vocabulary leak")
    print()

    # -- 4b. Composition: long, grammatical, and never invented -------------
    from .arm_schema import ArmIndex
    from .compose import compose_digest, compose_report, unsupported_sentences

    arms = ArmIndex()
    store2 = CrossStore()
    ingest_documents(store2, docs, arms)
    road = compose_report(store2, "国道", arms=arms)

    # The two defects the first version produced, kept as regressions.
    # 「通行可能しています」came from inferring the predicate form from kanji
    # count; the polar vocabulary is closed, so the form is stated per term.
    ok = "通行可能です" in road.text and "しています" not in road.text.split("通行")[0]
    print(f"[{'ok  ' if ok else 'FAIL'}] polar terms take their real predicate "
          f"form, not one inferred from shape")
    if not ok:
        failures.append("ja predicate form")

    # A claim cannot be settled and contested in the same report. Replaying a
    # source sentence reintroduced this, because the sentence carrying the
    # disputed word is itself a real sentence.
    ok = not ("以下は確認された情報です。以下は" in road.text)
    print(f"[{'ok  ' if ok else 'FAIL'}] no empty 'settled' heading when "
          f"everything under it is contested")
    if not ok:
        failures.append("empty settled heading")

    digest = compose_digest(store2, ["国道", "本町", "毛布", "水道"], arms=arms)
    bad = unsupported_sentences(store2, digest)
    ok = not bad
    print(f"[{'ok  ' if ok else 'FAIL'}] {len(digest.text)} characters composed, "
          f"none unsupported by the store")
    for b in bad[:3]:
        print(f"        invented: {b}")
    if not ok:
        failures.append("composed sentence with no stored backing")
    print()

    # -- 5. File loaders ---------------------------------------------------
    with tempfile.TemporaryDirectory() as td:
        root = Path(td)
        (root / "a.txt").write_text("国道4号は通行止です。\n", encoding="utf-8")
        (root / "b.html").write_text(
            "<html><head><style>p{color:red}</style></head><body>"
            "<p>本町の避難所は閉鎖されました。</p><script>var x=1</script>"
            "</body></html>", encoding="utf-8")
        (root / "c.csv").write_text("name,status\n本町避難所,開設\n", encoding="utf-8")
        (root / "d.json").write_text('{"road":{"state":"通行可能"}}', encoding="utf-8")
        ns = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
        with zipfile.ZipFile(root / "e.docx", "w") as zf:
            zf.writestr("word/document.xml",
                        f'<?xml version="1.0"?><w:document xmlns:w="{ns}"><w:body>'
                        f"<w:p><w:r><w:t>南町の避難所は</w:t></w:r>"
                        f"<w:r><w:t>開設されました。</w:t></w:r></w:p>"
                        f"</w:body></w:document>")
        (root / "f.pdf").write_bytes(b"%PDF-1.4 not really")

        res = load_directory(str(root))
        ok = res["loaded"] == 5
        print(f"[{'ok  ' if ok else 'FAIL'}] 5 of 6 files loaded "
              f"(got {res['loaded']})")
        if not ok:
            failures.append(f"loaders: {res['loaded']} loaded")

        # A missing optional parser must not take the batch down with it.
        pdf_skips = [s for s in res["skipped"]
                     if s["verdict"] in {"UNKNOWN_NO_PARSER", "UNKNOWN_UNREADABLE",
                                         "UNKNOWN_EMPTY_DOCUMENT"}]
        ok = len(pdf_skips) == 1
        print(f"[{'ok  ' if ok else 'FAIL'}] the PDF is reported by name, and "
              f"the other five still load")
        if not ok:
            failures.append("pdf skip")

        # Word runs must join without separators — Word splits a sentence
        # across runs whenever formatting changes, and spaces would land
        # inside words.
        docx = [d for d in res["documents"] if d.source == "e.docx"]
        ok = bool(docx) and "南町の避難所は開設されました。" in docx[0].text
        print(f"[{'ok  ' if ok else 'FAIL'}] .docx runs join without inserting "
              f"separators inside a sentence")
        if not ok:
            failures.append("docx run joining")

        # An unreadable file is a typed refusal, not an empty Document.
        missing = load_path(str(root / "nope.txt"))
        ok = missing["verdict"] == "UNKNOWN_UNREADABLE"
        print(f"[{'ok  ' if ok else 'FAIL'}] a missing file refuses by type "
              f"rather than returning an empty document")
        if not ok:
            failures.append("missing file verdict")
    print()

    print(f"formats supported: {', '.join(sorted(SUPPORTED))}")
    print()
    if failures:
        print(f"{len(failures)} failed: {', '.join(failures)}")
        return 1
    print("Japanese documents reach typed, attributed, contested verdicts")
    return 0


if __name__ == "__main__":
    sys.exit(main())
