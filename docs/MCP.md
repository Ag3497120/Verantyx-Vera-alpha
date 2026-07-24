# MCP server — Vera as external memory for LLM agents

Vera's MCP server turns any MCP client (Claude Code, Claude Desktop, …) into
an agent with **persistent, auditable, deletable memory** that never
hallucinates: `ask` returns typed verdicts, and anything ungrounded comes
back as `UNKNOWN_*` instead of a plausible-sounding guess.

## Install & run

```bash
pip install -e ".[mcp]"
vera --store ~/vera_memory.json mcp
```

## Claude Code

```bash
claude mcp add vera -- vera --store ~/vera_memory.json mcp
```

## Claude Desktop (`claude_desktop_config.json`)

```json
{
  "mcpServers": {
    "vera": {
      "command": "vera",
      "args": ["--store", "/Users/you/vera_memory.json", "mcp"]
    }
  }
}
```

## Tools

| Tool | What it does |
|------|--------------|
| `remember(sentence)` | teach one fact; usable immediately |
| `ask(query)` | grounded QA with typed verdict + provenance counts |
| `recall(core, k)` | dump a concept's accumulated facets (both sense channels) |
| `forget(core)` | **really** delete a concept |
| `math(query)` | exact arithmetic / typed equation solving |
| `code_ingest(repo_path)` | AST-ingest a Python repo |
| `code_query(query)` | who-calls / calls-of / impact analysis |
| `stats()` | store size and provenance counters |

## Why pair an LLM with Vera

The LLM does what it is good at (language, fluency, planning); Vera holds
what the LLM is bad at holding (facts that must not drift, memories that must
survive sessions, deletions that must be real). The division is clean because
Vera is a separate deterministic process — it works identically well with no
LLM at all.
