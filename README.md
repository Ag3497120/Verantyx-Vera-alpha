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

## Chat modes: lab & hybrid (a local LLM under Vera's control)

```bash
vera chat --mode lab                      # deterministic only (default)
vera chat --mode hybrid --llm qwen3.5:2b  # local Ollama model, Vera allocates
```

In **hybrid** mode Vera stays the controller; the local model is only the
language surface. Allocation is deterministic and every reply is labeled:

| Label | Meaning |
|-------|---------|
| `[math]` / `[code]` | exact routes — the LLM is **never** allowed to touch proven values |
| `[llm←vera-facts]` | Vera answered; the LLM only rephrases Vera's verified facts |
| `[llm UNVERIFIED]` | Vera has nothing; the LLM may converse, explicitly unverified |
| `UNKNOWN_*` | genuine ambiguity stays refused — the LLM cannot vote it away |

**Native memory harness** (no MCP, no triggers): in chat, every declarative
utterance is remembered automatically; questions and imperatives are not
(so "tell me something" never becomes a fake fact). Disable with
`--no-auto-memory`.

**Multi-line paste**: `vera chat` and `vera agent` capture a pasted block
(traceback, multi-sentence note, JSON) as one message instead of splitting
it line by line — plain `input()` submits on every embedded newline, which
silently mangles pastes. This uses bracketed-paste mode and needs a real
TTY; piped/non-interactive input falls back to reading one line at a time.

## First-run setup

```bash
vera setup       # arrow-key menu: pick a local Ollama model + the allocation
                 # dial (which domains Vera owns vs where the LLM may speak),
                 # saved to ~/.verantyx.json
```

## Agent mode (hands and feet)

A ReAct loop where **Vera is the controller** and tools do real work — file
edits, folder/file creation, shell commands, and web search — each mutating
action gated behind **arrow-key approval**:

```bash
vera agent "read README.md and tell me the license"
vera agent          # interactive; ↑/↓ + Enter to approve/deny each action
```

Exact math/code finishes with no LLM and no tools; web search is a stdlib
DuckDuckGo client (no API key). Full details: [docs/AGENT_MODE.md](docs/AGENT_MODE.md).

## Guided data placement

```bash
vera wizard      # arrow-key: choose a corpus + row budget, then it pours
```

## Base store from HuggingFace (no local store needed)

Vera ships no weights; the artifact is the poured store. Publish it once,
and any fresh checkout fetches it automatically:

```bash
vera push-store --repo <user>/Verantyx-Vera-base-store   # upload (needs HF login)
# later, on any machine: if no local store exists and hf_store_repo is set
# in ~/.verantyx.json, `vera ask ...` fetches the base store on first use.
```

**Live base store:** [`kofdai/Verantyx-Vera-base-store`](https://huggingface.co/datasets/kofdai/Verantyx-Vera-base-store)
— 889k concept crosses / 9.78M facet links (WikiText-2/103, ag_news, DBpedia,
SQuAD, IMDB). Set it once and any fresh checkout works with knowledge already
inside:

```bash
vera setup    # or edit ~/.verantyx.json: "hf_store_repo": "kofdai/Verantyx-Vera-base-store"
vera ask "what is football"    # auto-fetches the base store on first use
```

## Languages

The cross substrate is symbol-agnostic; segmentation is per-language:

- **English** — full elementary-grammar pipeline (richest)
- **Japanese** — tokenizer-free script-run segmentation
  (`リンゴは甘い果物です` → core リンゴ, facets 甘い/果物; recall + typed refusal)
- **Spanish / French / German** — generic content-word path with per-language
  function-word stoplists; other Latin-script languages fall back to a shared list

`vera chat --lang auto` detects per utterance; force with `--lang ja` etc.

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

**Bug localization by agreement** (`verantyx.debug_consensus.locate_bug`):
traceback, recent-diff, and failing-test sections each nominate cause
functions; a cause is asserted only when the sections agree — otherwise a
typed UNKNOWN with the disagreement map. The `DEBUG_BEATS_BASELINES` fork
shows the consensus resisting noise that fools a most-recently-changed
baseline. See [docs/CODE_REASONING.md](docs/CODE_REASONING.md).

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
vera lab        # 41 forks: consensus gates, pouring, math, rewriting, Kripke,
                # languages, router allocation, debug consensus, memory
                # provenance/contradiction, SQLite round-trip
```

## Memory with provenance & contradiction detection

```python
from verantyx import CrossStore
st = CrossStore(track_provenance=True)
st.add("server:prod-1", ["os:ubuntu"], source="infra sheet 7/24")
st.add("server:prod-1", ["os:debian"], source="slack 7/25")
st.contradictions("server:prod-1")
# → key "os" holds two values, each with counts, timestamps, and sources
```

`key:value` facets are exclusive per key: a conflicting fact is **reported as
a contradiction with both sources**, never silently overwritten. This turns
the memory from "doesn't hallucinate" into "notices when it is being lied to".

## SQLite backend (scale & fast writes)

```python
from verantyx import save_sqlite, load_sqlite, SqliteSync
save_sqlite(store, "big.db")            # distributable single file
st = load_sqlite("big.db")              # or cores_like="fn:%" for a slice
sync = SqliteSync(st, "big.db"); st.add(...); sync.flush()   # delta writes
```

Reference store poured with the same pipeline: WikiText-2 + WikiText-103 +
ag_news + DBpedia + SQuAD + IMDB ≈ **889k cores / 9.78M facet links**.

## Passive memory from AI output (quarantined, never auto-trusted)

```bash
vera propose-ai-facts "The staging DB runs postgres 14. It might also \
support replication, I'm not sure." --source ai_output:claude
# → quarantines "The staging DB runs postgres 14." only —
#   the hedged sentence never becomes a candidate
vera review-ai-facts     # arrow-key accept/reject each pending candidate
```

Nothing proposed here is queryable via `ask` until explicitly accepted —
an LLM's own text (even its final answer) can be wrong or hedged, so it
never writes directly into the trusted store. Same tools over MCP:
`propose_ai_facts` / `list_pending_ai_facts` / `accept_ai_fact` /
`reject_ai_fact`. Rationale: [docs/DESIGN.md](docs/DESIGN.md#passive-memory-from-ai-output-quarantined).

## Getting started as a builder

New here? Read [docs/ONBOARDING.md](docs/ONBOARDING.md) — a zero-to-custom
walkthrough (data-path choice, pouring best practices, verification, scale-up,
extension points). Design rationale lives in [docs/DESIGN.md](docs/DESIGN.md).

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

- Output is structured facet documents, **not fluent prose** (hybrid mode
  buys fluency from a local LLM, clearly labeled).
- English has the richest pipeline; Japanese is an elementary tokenizer-free
  recall path (no consensus decomposer yet); es/fr/de use a generic
  content-word path.
- Facet extraction is rule-based and shallow; noisy corpora leave noisy
  facets (they are at least *visible* and deletable).
- Same-surface homographs in the same channel can mix; sense clusters
  mitigate at query time but need specifier words.
- Kripke checking is finite-model only; no tableau validity, no proof search.
- Naturals-only arithmetic (6-digit v0); no fractions/negatives yet.

## License

MIT
