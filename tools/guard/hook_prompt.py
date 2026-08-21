#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""UserPromptSubmit: 指示→約束の抽出・登録、薄れた約束だけ再注入、
「もう〜していいよ」の退役。抽出は器官(extract_covenants)に一元化。"""
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import guard, read_hook_input, store_path

#: 解除の定型(閉じた表)。「もう絵文字使っていいよ」→ 対象語の約束を退役。
_RELEASE = re.compile(
    r"もう([一-龥ァ-ヺa-zA-Z0-9ー]+)(?:を)?使っていい|"
    r"([一-龥ァ-ヺa-zA-Z0-9ー]+)の禁止(?:を)?解除")


def main():
    data = read_hook_input()
    prompt = str(data.get("prompt", ""))
    store = store_path()

    # 1) 解除の検出 → 該当語を forbids に持つ約束を退役
    for m in _RELEASE.finditer(prompt):
        term = m.group(1) or m.group(2)
        listed = guard("list", {}, store=store) or {}
        for c in listed.get("covenants", []):
            if not c.get("retired") and term in c.get("forbids", []):
                guard("retire", {"name": c["name"], "quote": prompt[:80]},
                      store=store)

    # 2) 指示 → 約束の抽出と登録(器官に一元化 — フックは配管だけ)
    ex = guard("extract", {"text": prompt}, store=store) or {}
    for cand in ex.get("candidates", []):
        guard("set", cand, store=store)

    # 3) 薄れた約束だけ再注入(毎ターン全部は情報を運ばない — Vera自身の注記)
    fading = guard("fading", {}, store=store) or {}
    rows = fading.get("fading") if isinstance(fading, dict) else None
    lines = []
    for r in (rows or []):
        if isinstance(r, dict) and r.get("delta", 0) < 0:
            lines.append(f"・「{r.get('covenant')}」"
                         f"(守れていた率 {r.get('kept_before')}"
                         f" → {r.get('kept_recently')})")
    if lines:
        print(json.dumps({"hookSpecificOutput": {
            "hookEventName": "UserPromptSubmit",
            "additionalContext": "<vera-covenant-reminder>\n"
                                 + "\n".join(lines)
                                 + "\n</vera-covenant-reminder>"}},
            ensure_ascii=False))


if __name__ == "__main__":
    main()
