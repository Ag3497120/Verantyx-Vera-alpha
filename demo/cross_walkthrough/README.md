# 立体十字の中を歩く — 解説デモ

3Blue1Brown 様式(暗い背景・内部の数と構造を一歩ずつ動かす)で、
立体十字構造体の仕組みを見せる単一 HTML。

- `trace.py` — 実エンジン(`verantyx.consensus` の evaluate / _enumerate_moves /
  _apply_move)を一歩ずつ記録する。最終 verdict は `run_consensus` と一致することを
  assert する(捏造した動きではない)。出力 `traces.json`。
- `template.html` — `__TRACES__` に traces.json を埋め込む雛形。
- `cross_walkthrough.html` — 埋め込み済みの完成品(ブラウザで開くだけ)。

再生成: `python3.11 demo/cross_walkthrough/trace.py` の後、template に埋め込む。
店は 6 核・29 文の合成データ(法令調・実法令ではない)。動きは本物、規模は模型。
