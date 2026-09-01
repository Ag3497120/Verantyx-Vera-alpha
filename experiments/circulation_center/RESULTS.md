# 結果: 巡回の到達と中心の占有(PREREG.md、2026-08-31)

実行環境: Linux (クラウドセッション)。数値はすべて実行結果。

## M1 — 死線の確認(配線前)

```
first_ask                 ANSWER、locks ["+x"] が書かれる
circulation_written_keys  ["gadget"]           ← 書き込みの鍵
shell_center_at_read      null                 ← 読み出しの鍵
seed_reachable            false
second_ask                seeded_from 無し、種は届かない
```

J1 成立。静的読みどおり: 読み出し3箇所(consensus_store 298/533/538)は
`shell.center`(入口では常に None)を鍵に引いており、書き込み
(mcp_server 1692 = 表示名鍵、consensus_store 344 = core_key 鍵 —
鍵も不統一)と一度も一致しない。**巡回は書き込み専用だった。**
ja 経路(本番の主席)には circulation の席そのものが無かった。

axis_lock_fork がこれを見逃した機序も確認: 書き込み側だけを検査し、
「ロックの有無で verdict 不変」はロックが乗らなければ空虚に真。

## M2 — 到達・無害・近道(配線後)

### ① 到達 ② 無害(J2)

| 経路 | 探針 | identical (verdict・core・text) | seeded_from |
|---|---|---|---|
| EN (consensus_over_store) | 6問 | 6/6 | gadget / widget |
| ja (ja_consensus_ask) | 3問 | 3/3 | 過失 |

### ③ 近道(J3) — 移動が受理される店での再訪

```
初訪            ANSWER alpha  moves 1  escape 消費  carry {widened: true}
再訪(種なし)  ANSWER alpha  moves 1  escape 消費
再訪(種あり)  ANSWER alpha  moves 0  escape 未消費  seeded_from alpha
```

答えは同一のまま、moves 1→0、escape は未消費で残る。構想の
「時間のショートカット」が初めて数字になった。moves が増えた探針は 0件。

### ④ 同点は棄権

全 facet 同点の店: locks は書かれず(locks_written false)、
2問目に seeded_from は立たない。恣意的な配置は持ち越されない。

## J4 — fork

91/91 緑(既存89 + CIRCULATION_REACHES_THE_NEXT_SEARCH +
THE_CENTER_IS_THE_SETTLED_CORE)。

## J5 — 後方互換

circulation を渡さない呼び出し(EN 3問 + ja 1問、ANSWER と拒否の両方)を
変更前コード(git stash)と比較: **バイト単位で同一**。
carry_state の形も不変(center は結果 state 上にのみ占有され、
シリアライズには足していない)。

## 変更点(束ねず重ねる — 種は配置のみ、票には入らない)

- consensus.py: ANSWER の終端状態で `shell.center = core`(中心の占有。
  代入はこれがコードベース初)。非 ANSWER は None のまま。
- consensus_store.py: `_seed_for`(候補核の決定論的な最初の一致)で
  読み出しを復旧。EN/ja 両経路 + 配置不変の再読(同じ種を両読みに渡す —
  片方だけ種を持つと門が測るものが変わる)。ja に circulation の席と
  locks の書き戻しを追加。種が乗ったときだけ `seeded_from` を名乗る
  (不在は不在のまま)。
- engine.py: 主席(`ja_consensus_ask`)に circulation を渡す。
- mcp_server.py: 書き込み鍵を core_key に統一、locks は上書きせず合流。
  旧ファイルの表示名鍵は残るが読まれないだけで無害(削除しない)。

## M3 — CLI 扉の継続性(PREREG2.md、2026-09-01 追記)

実 CLI サブプロセス(`python -m verantyx.cli ask`)で測定:

```
側車なし   決定論的(2回バイト同一) / 側車は作られない / seeded_from 無し
側車あり   seeded_from alpha / verdict・core・text 同一
           moves 1→0 / escape 未消費 / 終端配置の書き戻し確認
```

J1(無側車の純粋さ)・J2(到達と無害)成立。会話扉が書いた
`<store>.circulation.json` を CLI の ask が引き継ぎ、書き戻す —
扉は一つ、入口は二つ(PREREG7 の線)。側車が無ければ何も作らない。

## 留保

治具は合成の小店。実ストア(89k核)はこのリポジトリに同梱されて
いないため、実データでの短縮量は未測定。答えの不変(J2)は fork が
恒久に検査するが、近道の実益の規模は実ストアで測り直すこと。
