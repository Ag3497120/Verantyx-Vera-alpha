# 結果: E1 の効果測定(PREREG.md、2026-09-01)

実行環境: Linux(クラウドセッション)。ハーネス
`verantyx/consensus_ab_eval.py` は無改変。全文は eval_output.txt。

## 数字

名指し問い(18核・無意味語20):

| arm | accuracy | honesty | invented |
|---|---|---|---|
| flat / ring / fine | 1.0 | 1.0 | 0 |
| flat / geo / fine | 1.0 | 1.0 | 0 |
| matryoshka A / fine | 1.0 | 1.0 | 0 |
| matryoshka B / fine | 0.222 | 1.0 | 0 |
| matryoshka C / fine | 1.0 | 1.0 | 0 |

facet のみの問い(核名を出さない18問):

| arm | exact | defensible | answer_rate |
|---|---|---|---|
| ring | 0.056 | 0.167 | 1.0 |
| geo | 0.056 | 0.167 | 1.0 |
| matryoshka B | 0.056 | 0.111 | 0.167 |
| matryoshka C | 0.056 | 0.111 | 0.778 |

効果(差分):

```
geometry (geo − ring)   accuracy +0.000  honesty +0.000  exact +0.000  defensible +0.000
placement (fine − coarse)                                exact +0.000  defensible +0.000
matryoshka A vs flat    ±0.000 全指標
matryoshka B vs flat    accuracy −0.778(名指し)、answer_rate 1.0→0.167(facet)
matryoshka C vs flat    defensible −0.056
```

妥当性検査: INVALID なし。名指し battery は飽和(全腕 1.0 — 順位付け
不能の天井)、facet battery は飽和せず差を測れる状態で、それでも
geo−ring は全指標 0。

## 読み

1. **E1 はこのコーパスでは決まらない。** 差が「小さい」のではなく
   全指標で 0 — 環状と幾何学的視界の唯一の差(対極の可視性)は、
   回転と一回退避を持つ探索では終端を変えなかった。既定値
   (geometric_visibility=False)は事前登録どおり変えない。
   切り替えの材料は実ストア(89k核、リポジトリ外)での再測だけが作れる。
2. **マトリョーシカの過去の実測と整合。** A/C は恒等(上層はコピー)、
   B(クエリを初層で捨てる)は名指しで −0.778、facet で answer_rate
   1.0→0.167 — 伝言ゲームの劣化がここでも出た。錨(元の問い)を
   落とすと層は誤収束ではなく棄権に流れる(honesty は 1.0 のまま)。
3. 捏造は全腕 0。無意味語 20 はどの構成でも全拒否 — 門は topology や
   層の選び方に依存していない。

## 残る問い

E1 の本判定には、断面の不一致が実際に頻発する店(候補が6腕を埋め、
質量が割れる実ストア)での同じ差分が要る。このリポジトリには実ストアが
同梱されていないため、ここまで。
