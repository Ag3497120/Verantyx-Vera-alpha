#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""PostToolUse(Write|Edit): ファイルに書いた内容を同じ照合に通す。

実地試験の限界4(返答テキストしか見ておらず、ファイルに禁止語を書いても
素通し)を塞ぐ覆い。新しい道具は増やさない — 同じ check をファイル内容に
向けるだけ。判定は非遮断(additionalContext で知らせる): 書き込みは既に
起きており、ここで block しても取り消せない — 嘘の遮断はしない。"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import guard, read_hook_input, store_path


def main():
    data = read_hook_input()
    tool = data.get("tool_name") or ""
    ti = data.get("tool_input") or {}
    # ④ 全 tool 実行を証人として記録(判定はしない — 置くだけ)。
    # required 側は字面でなくこの記録で監査される(audit)。
    detail = str(ti.get("command") or ti.get("file_path")
                 or ti.get("description") or "")[:400]
    guard("witness", {"tool": tool, "detail": detail}, store=store_path())
    if tool not in ("Write", "Edit"):
        return
    content = str(ti.get("content") or ti.get("new_string") or "")
    if not content:
        return
    out = guard("check", {"reply": content,
                          "asked": str(ti.get("file_path", ""))},
                store=store_path())
    if not out or out.get("verdict") != "BROKEN":
        return
    hard = [v for v in out.get("violations", []) if v.get("forbidden_used")]
    if not hard:
        return
    v = hard[0]
    found = v.get("class_hits")
    extra = (f"(文字: {found[0]['found']})" if found else "")
    print(json.dumps({"hookSpecificOutput": {
        "hookEventName": "PostToolUse",
        "additionalContext": f"<vera-covenant-warning>ファイル "
                             f"{ti.get('file_path')} に約束違反: "
                             f"「{v.get('inject')}」の禁止語 "
                             f"{v.get('forbidden_used')}{extra}"
                             f"</vera-covenant-warning>"}},
        ensure_ascii=False))


if __name__ == "__main__":
    main()
