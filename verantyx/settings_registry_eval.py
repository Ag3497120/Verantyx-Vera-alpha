"""Does the settings registry answer, refuse, and stay in sync?

Three properties, in the order they matter:

  1. Typed behaviour — the right verdict for the right question, including
     the refusals. A support answer that is confidently wrong is worse than
     one that says "I do not have that", so the refusal cases are tests, not
     afterthoughts.
  2. Bilingual reach — the same question in Japanese and English reaches the
     same setting. Two cases below are regressions: the first scorer scored
     substrings, so "how do I change the language" came back as the
     context-window setting, and it tokenised on whitespace, so every
     Japanese question matched nothing at all.
  3. Freshness — the registry still describes the IDE that exists. Skipped
     with a stated reason when the checkout is absent, never passed silently.

Run:  python3 -m verantyx.settings_registry_eval
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Optional, Tuple

from .settings_registry import (MODE_FAMILIES, SETTINGS, TABS, all_modes,
                                lookup, search, verify_against_source)

#: (question, expected_verdict, expected_key or None)
CASES: List[Tuple[str, str, Optional[str]]] = [
    # -- English ----------------------------------------------------------
    ("how do I change the language", "ANSWER", "language"),
    ("temperature", "ANSWER", "model.temperature"),
    ("turn on the terminal tool", "ANSWER", "tools.terminal"),
    ("system prompt", "ANSWER", "agent.system_prompt"),

    # -- Japanese ---------------------------------------------------------
    ("言語を日本語にしたい", "ANSWER", "language"),
    ("ollamaのモデルを変えたい", "ANSWER", "model.ollama"),
    ("温度を下げたい", "ANSWER", "model.temperature"),
    ("ターミナルを使わせたい", "ANSWER", "tools.terminal"),
    ("エンドポイントを変更", "ANSWER", "model.ollama_endpoint"),

    # -- Refusals ---------------------------------------------------------
    # The point of the whole module: no setting exists, so none is invented.
    ("how do I enable blockchain mode", "UNKNOWN_NO_SETTING", None),
    ("ブロックチェーンを有効に", "UNKNOWN_NO_SETTING", None),
    ("", "UNKNOWN_NO_SETTING", None),
    # Genuinely undecidable — three API keys, and picking one silently would
    # send the user to the wrong field.
    ("APIキーはどこ", "UNKNOWN_AMBIGUOUS", None),
]


def main() -> int:
    print(f"settings registry — {len(SETTINGS)} settings, "
          f"{len(MODE_FAMILIES)} mode families\n")
    failures: List[str] = []

    for question, expect_verdict, expect_key in CASES:
        got = lookup(question)
        ok = got["verdict"] == expect_verdict
        if ok and expect_key is not None:
            ok = got.get("key") == expect_key
        label = question or "(empty)"
        print(f"[{'ok  ' if ok else 'FAIL'}] {label}")
        print(f"        expected {expect_verdict}"
              f"{'/' + expect_key if expect_key else ''}, "
              f"got {got['verdict']}{'/' + got['key'] if got.get('key') else ''}")
        if not ok:
            failures.append(label)
    print()

    # GUI-only settings must SAY they are GUI-only. The predecessor bot had no
    # way to express this and so invented a CLI command instead, which is the
    # exact failure this verdict exists to make impossible.
    gui_only = [s for s in SETTINGS if s.cli is None]
    bad = [s.key for s in gui_only
           if lookup(s.key).get("cli_verdict") != "UNKNOWN_NO_CLI"]
    ok = not bad
    print(f"[{'ok  ' if ok else 'FAIL'}] {len(gui_only)} GUI-only settings report "
          f"UNKNOWN_NO_CLI rather than a guessed command")
    if not ok:
        failures.append(f"cli verdict: {bad}")
    print()

    # Every GUI path must name a screen that exists.
    bad_tabs = [s.key for s in SETTINGS if s.tab not in TABS]
    ok = not bad_tabs
    print(f"[{'ok  ' if ok else 'FAIL'}] every setting points at one of the "
          f"{len(TABS)} real settings tabs")
    if not ok:
        failures.append(f"tabs: {bad_tabs}")
    print()

    # Mode consolidation: one call returns every family, and no two families
    # share a group id (which would silently merge two unrelated switches).
    modes = all_modes()
    groups = [m["group"] for m in modes]
    ok = len(modes) == len(MODE_FAMILIES) and len(set(groups)) == len(groups)
    total_options = sum(len(m["options"]) for m in modes)
    print(f"[{'ok  ' if ok else 'FAIL'}] all_modes returns {len(modes)} distinct "
          f"families covering {total_options} options")
    if not ok:
        failures.append("mode families")
    print()

    # Freshness against the real checkout. Absent tree is reported, not passed.
    ide = Path.home() / "verantyx" / "cli" / "VerantyxIDE" / "Sources" / "Verantyx"
    fresh = verify_against_source(str(ide))
    if fresh["verdict"] == "UNKNOWN_NO_SOURCE":
        print(f"[skip] freshness: {fresh['reason']} — the registry was NOT "
              f"checked against the IDE on this machine")
    else:
        ok = fresh["verdict"] == "ANSWER"
        print(f"[{'ok  ' if ok else 'FAIL'}] freshness: {fresh['checked_settings']} "
              f"keys still present across {fresh['swift_files']} Swift files")
        if not ok:
            print(f"        stale: {fresh['missing_defaults_keys']} "
                  f"{fresh['missing_mode_sources']}")
            failures.append("registry is stale against the IDE source")
    print()

    if failures:
        print(f"{len(failures)} failed: {', '.join(failures)}")
        return 1
    print("settings registry answers, refuses, and matches the source")
    return 0


if __name__ == "__main__":
    sys.exit(main())
