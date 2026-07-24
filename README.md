# Verantyx Vera α

**A deterministic, LM-free knowledge & reasoning engine that refuses to hallucinate.**

Vera stores knowledge as *crosses* — one per concept, with a core meaning and
accumulating factual facets — and reasons by **multi-frontier consensus
search**: several sections explore toward a center, and an answer ships only
when they agree with sufficient evidence. Everything else is a **typed
refusal**:

```text
ANSWER · AMBIGUOUS · UNKNOWN_NO_EVIDENCE · UNKNOWN_INSUFFICIENT_EVIDENCE ·
UNKNOWN_SECTION_DISAGREEMENT · UNKNOWN_BUDGET · UNKNOWN_NO_SOLUTION · …
```

No neural network. No GPU. No sampling temperature. Same input, same output,
every time — and every answer traces back to counted source sentences.

> **Status: alpha research prototype.** Vera is honest about what it cannot
> do: it does not write fluent prose, it does not chat casually, and it does
> not invent anything it was never taught. That is the point.

## Why

| LLM | Vera |
|-----|------|
| Answers everything, sometimes wrongly | Answers only what it can ground, refuses the rest |
| Knowledge baked into weights | Knowledge is data — inspect, count, **delete for real** |
| Arithmetic is probabilistic | Arithmetic is exact by construction (wire carry propagation) |
| GBs of weights + GPU | A JSON store + CPU; a 900k-concept store is ~200 MB |
| Forgetting is an open research problem | `vera forget apple` — gone |

Strong areas (in order of readiness): **hallucination-free knowledge QA**,
**persistent memory for agents (via MCP)**, **code reasoning**
(who-calls / impact analysis), exact **arithmetic / equations / term
rewriting / Kripke model checking**.

Weak by design: creative writing, small talk, free-form generation.

## Install

```bash
git clone https://github.com/Ag3497120/Verantyx-Vera-alpha.git
cd Verantyx-Vera-alpha
pip install -e .            # core (stdlib only)
pip install -e ".[hf]"      # + HuggingFace corpus pouring
pip install -e ".[mcp]"     # + MCP server
```

Python ≥ 3.9. No other core dependencies.

## Quickstart

```bash
# teach a fact — usable immediately, no training
vera remember "The bright apple is sweet ."

# ask — grounded answer with provenance counts
vera ask "what is apple"
# → ANSWER "apple bright sweet"

# ask something it was never taught
vera ask "what is quantum chromodynamics"
# → UNKNOWN_NO_EVIDENCE  (it says so, instead of making something up)

# exact math on the same substrate
vera math "solve x + 3 = 7"        # → ANSWER x=4
vera math "x * 0 = 0"              # → AMBIGUOUS (many solutions — no vote)
vera simplify "(2 + 3) * y"        # → 5 * y   (rule trace included)

# interactive session (knowledge + math + code in one REPL)
vera chat
```

## Pouring corpora (bulk knowledge)

```bash
# built-in synthetic corpus (offline smoke test)
vera pour --source synthetic --max-rows 2000

# WikiText-2 from the HuggingFace cache
vera pour --source wikitext --max-rows 40000

# any HuggingFace dataset:  hf:<name>[#config][:text_field]
vera pour --source "hf:ag_news" --max-rows 120000
vera pour --source "hf:dbpedia_14:content" --max-rows 560000
vera pour --source "hf:wikitext#wikitext-103-raw-v1" --max-rows 2000000

# a local text file (one document per line)
vera pour --source file:corpus.txt
```

Pouring is deterministic and resumable (`--store` is a JSON checkpoint;
pouring again accumulates). A two-pass capitalization scan routes proper
names to their own sense channel (`bush#p` ≠ `bush`). Reference run: WikiText-2
+ ag_news + DBpedia + WikiText-103 ≈ **870k concept crosses / 9.2M facet
links**, poured in minutes on a laptop CPU.

More detail: [docs/ADDING_KNOWLEDGE.md](docs/ADDING_KNOWLEDGE.md).

## Code reasoning

```bash
vera code ingest path/to/repo          # AST → one cross per function
vera code ask "who calls wire_add"     # reverse call edges
vera code ask "what does simplify call"
vera code ask "impact of parse_term"   # BFS: what may break if it changes
```

Unknown functions get `UNKNOWN_NO_EVIDENCE`, not a guess.
See [docs/CODE_REASONING.md](docs/CODE_REASONING.md).

## MCP server (memory & knowledge tools for LLM agents)

Vera doubles as a **hallucination-free external memory** for Claude Code /
Claude Desktop or any MCP client:

```bash
pip install -e ".[mcp]"
vera --store ~/vera_memory.json mcp
```

Tools exposed: `ask`, `remember`, `recall`, `forget`, `math`,
`code_ingest`, `code_query`, `stats`. Setup snippets:
[docs/MCP.md](docs/MCP.md).

## Lab mode (self-test forks)

Every capability is guarded by falsifiable "fork" tests — including the
refusal behaviors:

```bash
vera lab        # 30 forks: consensus gates, pouring, math, rewriting, Kripke
```

## Architecture (one page)

```text
sentence ──classify──▶ core + facets ──accumulate──▶ CrossStore
                                                (core → {facet: count})
query ──decompose──▶ retrieve candidate crosses ──▶ shell (6 arms)
      ──▶ multi-frontier consensus search
           gates:  NoImprovingMove ∧ AllSectionsAgree
                 ∧ EvidenceComplete ∧ QueryGrounded ∧ NoContradiction
      ──▶ ANSWER (facet document) | typed UNKNOWN / AMBIGUOUS
disambiguation:  sense clusters over facet co-occurrence
                 ("sun newspaper" vs "sun in the sky")
layers:          matryoshka — unresolved disagreement is handed upward
math:            digits on arms, carry as current  → exact by construction
rules:           term rewriting; rules are data, poured like knowledge
modal logic:     Kripke worlds = crosses, R = joins, □ = agreement gate
```

Deep dives: [docs/MATRYOSHKA.md](docs/MATRYOSHKA.md) (layer stacking, carry
modes A/B/C), [docs/ADDING_KNOWLEDGE.md](docs/ADDING_KNOWLEDGE.md) (nodes,
facets, sense channels, deletion).

## Honest limitations

- Output is structured facet documents, **not fluent prose**.
- English only (whitespace + elementary grammar rules); Japanese needs a
  tokenizer front-end (planned).
- Facet extraction is rule-based and shallow; noisy corpora leave noisy
  facets (they are at least *visible* and deletable).
- Same-surface homographs in the same channel can mix; sense clusters
  mitigate at query time but need specifier words.
- Kripke checking is finite-model only; no tableau validity, no proof search.
- Naturals-only arithmetic (6-digit v0); no fractions/negatives yet.

## License

MIT
