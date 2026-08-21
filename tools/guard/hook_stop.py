#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Stop: 返答を約束と照合。BROKEN(禁止語 or 絵文字そのもの)なら遮断。

遮断は forbidden_used のみ(required_missing は字面照合で誤検知が多い
ことを実地試験が実測済み — 記録のみ)。stop_hook_active で二度は
止めない(無限ループ防止)。"""
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _common import guard, read_hook_input, store_path


def last_reply(data):
    t = data.get("last_assistant_message") or ""
    if t:
        return str(t)
    # transcript_path から末尾の assistant 発話を拾う(形式差に耐える)
    p = data.get("transcript_path")
    if not p or not os.path.exists(p):
        return ""
    try:
        lines = open(p, encoding="utf-8").read().splitlines()
        for line in reversed(lines):
            try:
                d = json.loads(line)
            except Exception:
                continue
            msg = d.get("message") or d
            if (d.get("type") == "assistant" or
                    msg.get("role") == "assistant"):
                c = msg.get("content")
                if isinstance(c, list):
                    return " ".join(x.get("text", "") for x in c
                                    if isinstance(x, dict))
                return str(c or "")
    except Exception:
        pass
    return ""


def main():
    data = read_hook_input()
    if data.get("stop_hook_active"):
        return                       # 二度は止めない
    reply = last_reply(data)
    if not reply:
        return
    out = guard("check", {"reply": reply}, store=store_path())
    if not out or out.get("verdict") != "BROKEN":
        return
    hard = [v for v in out.get("violations", [])
            if v.get("forbidden_used")]
    if not hard:
        return                       # required_missing は記録のみ
    reasons = []
    for v in hard[:3]:
        found = v.get("class_hits")
        extra = (f"(見つけた文字: {found[0]['found']})" if found else "")
        reasons.append(f"「{v.get('inject')}」の禁止語 "
                       f"{v.get('forbidden_used')} を使っている{extra}")
    print(json.dumps({"decision": "block",
                      "reason": "Vera が約束との矛盾を検出した: "
                                + " / ".join(reasons)},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()
