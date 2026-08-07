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
    # The reliability-vocabulary pairs ride the same guard; each of these
    # contains a pair member inside a compound and must stay silent.
    "運行状況を確認してください。",
    "有効期限は明日までです。",
    "無効化の手順を説明します。",
    "開館時間は9時です。",
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
    ("山手線は運休です。", "運行", "運休"),
    ("このパスは無効です。", "有効", "無効"),
    ("当館は閉館しております。", "開設", "閉館"),
    # Humble-register negation: 〜しておりません flips the pole.
    ("当館は開館しておりません。", "開設", "not_開館"),
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

    # The core is 国道4号 with the digits INTACT — the first segmenter
    # split it into 国道+号 and dropped the 4, which in a legal or medical
    # corpus is the difference between citing the right article and citing
    # nothing. The old key is asserted absent so a regression cannot pass by
    # filing facts under both.
    ok = "国道" not in store.crosses
    print(f"[{'ok  ' if ok else 'FAIL'}] digits survive: the core is 国道4号, "
          f"and the digit-stripped 国道 does not exist")
    if not ok:
        failures.append("digit-stripped core resurfaced")

    road = deep_report(store, "国道4号")
    contested = road["confidence"] == "contested"
    print(f"[{'ok  ' if contested else 'FAIL'}] 国道4号 is contested, not blended "
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
    road = compose_report(store2, "国道4号", arms=arms)

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

    # -- 4b2. Head-final cores and negation ---------------------------------
    # 「本町の避難所」is about the shelter, not the neighbourhood: Japanese is
    # head-final, so the topic's LAST noun is the core. Before this rule the
    # shelter's open/closed status was filed under 本町 and a query about
    # 避難所 found nothing.
    ok = "避難所" in store.crosses
    shelter = deep_report(store, "避難所")
    ok = ok and shelter["confidence"] == "contested"
    print(f"[{'ok  ' if ok else 'FAIL'}] head-final: the shelter's dispute is "
          f"filed under 避難所, not under its neighbourhood")
    if not ok:
        failures.append("head-final core")

    # 「安全ではありません」 was stored as 安全(+) — the pole inverted, a safety
    # claim manufactured from its own denial. The worst single defect found
    # in this module, pinned hardest.
    neg_cases = [
        ("この道は安全ではありません。", "not_安全", "-"),
        ("設備は稼働していません。", "not_稼働", "-"),
        ("この道は安全です。", "安全", "+"),
        ("設備は稼働しています。", "稼働", "+"),
    ]
    for sentence, value, pol in neg_cases:
        hits = detect_ja(sentence)
        ok = any(v == value and p == pol for _a, v, p in hits)
        print(f"[{'ok  ' if ok else 'FAIL'}] negation: {sentence} -> "
              f"{[(v, p) for _a, v, p in hits]}")
        if not ok:
            failures.append(f"negation: {sentence}")

    # Chinese routes away from the Japanese path. Before: 如果 and 配置
    # appeared as catalogue topics with Chinese facets under a "ja" label.
    zh_ok = detect("如果配置错误，网关将无法启动。") == "zh"
    ja_ok = detect("交通情報") == "ja"          # short han-only is normal Japanese
    ok = zh_ok and ja_ok
    print(f"[{'ok  ' if ok else 'FAIL'}] script split: han-without-kana prose "
          f"is zh, a short han label stays ja")
    if not ok:
        failures.append("zh/ja detection")
    print()

    # -- 4c. English prepositions must not read as state claims -------------
    # From the first real-corpus run: this project's own 12 documents produced
    # one contradiction across 251 cores and it was false — "corpus ON top"
    # against "trade-OFF". Precision 0 of 1 on the only hit there was.
    from .polarity import detect as detect_en
    prepositions = ["pour a second corpus on top and counts merge",
                    "same trade-off as any personal-secret scheme",
                    "based on the store contents"]
    bad = [t for t in prepositions if detect_en(t)]
    ok = not bad
    print(f"[{'ok  ' if ok else 'FAIL'}] on/off as prepositions produce no pole")
    for b in bad:
        print(f"        invented: {b} -> {detect_en(b)}")
    if not ok:
        failures.append("english preposition false positive")

    states = [("the switch is on", "on"), ("the light was off", "off"),
              ("turned off the feature", "off")]
    missed = [t for t, v in states
              if not any(val == v for _a, val, _p in detect_en(t))]
    ok = not missed
    print(f"[{'ok  ' if ok else 'FAIL'}] real on/off state claims still detected")
    if not ok:
        failures.append(f"english copula recall: {missed}")
    print()

    # -- 4d. A pole must belong to the core, not to whatever else the ------
    # sentence mentions. Measured on 2,633 real documents: without this gate
    # the catalogue reported 24 contested topics and the four inspected were
    # all false, every one the same way — the polar word described some other
    # noun in the sentence. With it, zero, while the Japanese disaster case
    # above still detects. Zero on a corpus with no real disputes is the
    # right answer; 24 was confidently wrong.
    from .polarity import subject_is_core
    gate = [
        # (sentence, core, word, lang, should the pole be placed)
        ("If sandbox mode is enabled but Docker is unavailable, doctor reports.",
         "sandbox", "unavailable", "en", False),
        ("The gateway surfaces one installer (brew when available).",
         "gateway", "available", "en", False),
        ("The service, which we deployed, is broken.",
         "service", "broken", "en", False),
        ("The shelter is open until midnight.", "shelter", "open", "en", True),
        ("The gateway is unavailable right now.",
         "gateway", "unavailable", "en", True),
        ("国道4号は通行止です。", "国道", "通行止", "ja", True),
        ("避難所は閉鎖されました。", "避難所", "閉鎖", "ja", True),
    ]
    for sentence, core, word, lg, expect in gate:
        got = subject_is_core(sentence, core, word, lg)
        ok = got == expect
        print(f"[{'ok  ' if ok else 'FAIL'}] subject gate {core}/{word} = {got}")
        if not ok:
            failures.append(f"subject gate: {core}/{word}")
    print()

    # -- 4e. Topics carry the language their sentences voted for ------------
    # A topic string cannot name its own language — 如果 and 結論 are both
    # two-ideograph words — but the sentences that produced it were already
    # routed per-script, so the label comes from that vote, not from a guess.
    from .catalog import build_catalog

    with tempfile.TemporaryDirectory() as td2:
        root2 = Path(td2)
        (root2 / "ja.txt").write_text(
            "国道4号は通行止です。避難所は開設されました。", encoding="utf-8")
        (root2 / "en.txt").write_text(
            "The bridge is closed for repairs. The bridge spans the river.",
            encoding="utf-8")
        (root2 / "zh.txt").write_text(
            "如果配置错误，网关将无法启动。建议使用默认设置以保证稳定运行。",
            encoding="utf-8")
        cat = build_catalog([str(root2 / n) for n in
                             ("ja.txt", "en.txt", "zh.txt")])
        got = {e.topic: e.lang for e in cat.entries}
        expect = {"国道4号": "ja", "避難所": "ja", "bridge": "en"}
        wrong = {t: (got.get(t), lg) for t, lg in expect.items()
                 if got.get(t) != lg}
        zh_ok = any(lg == "zh" for lg in got.values())
        ok = not wrong and zh_ok
        print(f"[{'ok  ' if ok else 'FAIL'}] catalogue entries labeled by their "
              f"sentences' language (ja/en/zh all present)")
        if wrong:
            print(f"        wrong: {wrong}")
        if not ok:
            failures.append("entry language labels")
    print()

    # -- 4f. Duplicates count once; .txt gets the same cleaning as .md ------
    # Measured: 1,588 of 2,633 files in the real corpus were byte-identical
    # copies living in a second repository checkout, doubling every ranking
    # signal. And a 107,752-line .txt notes file read raw contributed `py`
    # (1,471 mentions) as its top topic — the file is nominally plain text
    # and actually full of markdown and pasted code.
    with tempfile.TemporaryDirectory() as td3:
        root3 = Path(td3)
        (root3 / "a.txt").write_text(
            "設計の方針を説明します。\n```python\nimport json\n"
            "def parse(x):\n    return json.loads(x)\n```\n"
            "構造は決定論的であるべきです。", encoding="utf-8")
        (root3 / "b.txt").write_text(  # byte-identical copy of a
            (root3 / "a.txt").read_text(), encoding="utf-8")
        cat3 = build_catalog([str(root3 / "a.txt"), str(root3 / "b.txt")])
        dups = [x for x in cat3.manifest.skipped
                if x.get("verdict") == "DUPLICATE"]
        ok = len(dups) == 1 and cat3.manifest.files == 2
        print(f"[{'ok  ' if ok else 'FAIL'}] identical files count once, and "
              f"the duplicate is named in the manifest")
        if not ok:
            failures.append("dedupe")
        topics3 = {e.topic for e in cat3.entries}
        ok = "json" not in topics3 and "parse" not in topics3
        print(f"[{'ok  ' if ok else 'FAIL'}] pasted code inside a .txt file "
              f"does not become a topic")
        if not ok:
            print(f"        topics: {sorted(topics3)}")
            failures.append("txt cleaning")
    print()

    # -- 4g. Grammar as data: bundled, overlayable, loudly validated --------
    # The Japanese vocabulary ships as lang_data/ja_grammar.json and a user
    # extends it with a file beside their store — no code. The validator
    # enforces the rules this project paid for on real corpora (2-char
    # floor, one pole per term, no dangling references), and an invalid
    # overlay refuses to load with every problem named.
    from . import ja_grammar
    from .placement_explain import explain

    base = ja_grammar.status()
    ok = (base["pairs"] >= 12 and base["stopwords"] >= 50
          and base["overlay"] is None)
    print(f"[{'ok  ' if ok else 'FAIL'}] bundled grammar loads from data: "
          f"{base['pairs']} pairs, {base['stopwords']} stopwords")
    if not ok:
        failures.append("bundled grammar")

    with tempfile.TemporaryDirectory() as tdg:
        ov = Path(tdg) / "ja_grammar.json"
        ov.write_text('{"antonym_pairs": [["点灯", "消灯"]], '
                      '"predicates": {"点灯": "は点灯しています。"}}',
                      encoding="utf-8")
        info = ja_grammar.load_overlay(ov)
        hits = detect_ja("非常灯は点灯しています。")
        ok = any(v == "点灯" for _a, v, _p in hits) and info["added_pairs"] == 1
        print(f"[{'ok  ' if ok else 'FAIL'}] overlay adds a pair without code "
              f"and detection picks it up: {hits}")
        if not ok:
            failures.append("overlay round-trip")
        # The compound guard applies to overlay vocabulary too.
        ok = not detect_ja("点灯確認の手順を説明します。")
        print(f"[{'ok  ' if ok else 'FAIL'}] overlay vocabulary rides the same "
              f"compound guard (点灯確認 stays silent)")
        if not ok:
            failures.append("overlay compound guard")

        bad = Path(tdg) / "bad.json"
        bad.write_text('{"antonym_pairs": [["開", "閉"]]}', encoding="utf-8")
        try:
            ja_grammar.load_overlay(bad)
            ok = False
        except ValueError as exc:
            ok = "2 chars" in str(exc)
        print(f"[{'ok  ' if ok else 'FAIL'}] a 1-char pair is refused loudly "
              f"(開 lives inside 開始, 公開, 展開)")
        if not ok:
            failures.append("invalid overlay accepted")
    # The shipped sample domain pack must stay valid against the bundled
    # grammar — it is the document's worked example, and a worked example
    # that no longer loads is worse than none.
    import json as _json
    sample = Path(ja_grammar.BUILTIN_GRAMMAR).parent / "ja_domain_disaster.json"
    raw_sample = _json.loads(sample.read_text(encoding="utf-8"))
    raw_sample.pop("_comment", None)
    merged_sample = {
        "stopwords": sorted(ja_grammar.STOPWORDS),
        "antonym_pairs": [list(x) for x in ja_grammar.ANTONYM_PAIRS]
                         + raw_sample.get("antonym_pairs", []),
        "aspect_joins": [list(x) for x in ja_grammar.ASPECT_JOINS],
        "aliases": dict(ja_grammar.ALIASES),
        "predicates": {**ja_grammar.PREDICATES,
                       **raw_sample.get("predicates", {})},
    }
    errs2 = ja_grammar.validate(merged_sample)
    ok = not errs2
    print(f"[{'ok  ' if ok else 'FAIL'}] the shipped sample domain pack "
          f"validates against the bundled grammar")
    for e in errs2:
        print(f"        {e}")
    if not ok:
        failures.append("sample pack invalid")

    ja_grammar.load()   # restore pristine bundled state for later checks

    # The placement explainer: the adjustment loop's first step.
    exp = explain("本町の避難所は閉鎖されました。")
    ok = (exp["core"] == "避難所" and exp["core_rule"] == "head_of_topic_phrase"
          and any(p["placed"] for p in exp["poles"]))
    print(f"[{'ok  ' if ok else 'FAIL'}] explain_placement states core, rule "
          f"and pole for a Japanese sentence")
    if not ok:
        failures.append("placement explain")
    exp2 = explain("停止線の位置を確認してください。")
    ok = not exp2["poles"] and "pole_note" in exp2
    print(f"[{'ok  ' if ok else 'FAIL'}] explain_placement names WHY no pole "
          f"was placed on a compound-noun sentence")
    if not ok:
        failures.append("placement explain notes")
    print()

    # -- 4h. SQLite checkpoint: identical contents, crash-safe format -------
    # Chosen for the failure mode, not the speed: the JSON checkpoint
    # rewrites the whole store as one string (208 MB measured on an
    # accumulated store) and a crash mid-write parses as nothing. WAL rolls
    # back instead. Same store, either suffix, equal contents — asserted.
    with tempfile.TemporaryDirectory() as tds:
        pj, ps = Path(tds) / "s.json", Path(tds) / "s.sqlite"
        store.save(pj); store.save(ps)
        a, b = CrossStore.load(pj), CrossStore.load(ps)
        ok = (a.crosses == b.crosses == store.crosses
              and a.provenance == b.provenance == store.provenance
              and a.core_count == b.core_count
              and a.n_sentences == b.n_sentences)
        print(f"[{'ok  ' if ok else 'FAIL'}] JSON and SQLite checkpoints "
              f"round-trip to identical stores")
        if not ok:
            failures.append("sqlite round-trip")
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
