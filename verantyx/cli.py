"""vera — CLI for Verantyx Vera α (lab + chat in one binary).

Knowledge:
  vera pour --source synthetic|wikitext|hf:<name>[#config][:field]|file:<path>
  vera remember "The bright apple is sweet ."
  vera ask "what is apple"
  vera forget apple
  vera stats
  vera chat                    interactive REPL (knowledge + math + code)

Math / logic:
  vera math "x + 3 = 7"        wire arithmetic / typed equations
  vera simplify "x + 0"        term rewriting (rules are data)

Code reasoning:
  vera code ingest <path>
  vera code ask "who calls foo"

Lab:
  vera lab                     run the fork self-test suites
  vera mcp                     start the MCP server (see docs/MCP.md)

Default store: ./vera_store.json (override with --store).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

from .code_ingest import code_ask, ingest_python_repo
from .consensus_store import consensus_over_store
from .cross_store import CrossStore, pour_corpus
from .math_sim import math_ask
from .rewrite_kernel import default_algebra_rules, simplify

DEFAULT_STORE = "vera_store.json"


def _load(path: str) -> CrossStore:
    p = Path(path)
    return CrossStore.load(p) if p.is_file() else CrossStore()


def _print(obj: Any) -> None:
    print(json.dumps(obj, indent=2, ensure_ascii=False))


def _route(store: CrossStore, query: str) -> Dict[str, Any]:
    """math → code → knowledge, refusing rather than guessing."""
    m = math_ask(query)
    if m["verdict"] != "UNKNOWN_UNPARSED":
        m["route"] = "math"
        return m
    c = code_ask(store, query)
    if c["verdict"] != "UNKNOWN_UNPARSED":
        c["route"] = "code"
        return c
    out = consensus_over_store(store, query)
    out["route"] = "knowledge"
    return out


def cmd_pour(args) -> int:
    ckpt = Path(args.store)
    prev = CrossStore.load(ckpt) if ckpt.is_file() else None
    source = args.source
    kw: Dict[str, Any] = {}
    if source.startswith("file:"):
        from .corpus_en import iter_local_rows

        rows = iter_local_rows(Path(source[5:]))
        st = prev or CrossStore()
        if not args.no_two_pass:
            st.scan_cap_stats(iter_local_rows(Path(source[5:])))
        rep = st.ingest_rows(rows, max_sentences=args.max_sentences)
        st.source = source
        st.save(ckpt)
        _print({"pour": rep, "store": str(ckpt)})
        return 0
    st, rep = pour_corpus(
        source=source,
        max_rows=args.max_rows,
        max_sentences=args.max_sentences,
        checkpoint_path=ckpt,
        store=prev,
        two_pass=not args.no_two_pass,
        checkpoint_every=args.checkpoint_every,
        **kw,
    )
    _print({"pour": rep, "store": str(ckpt)})
    return 0


def cmd_remember(args) -> int:
    st = _load(args.store)
    key = st.ingest_sentence(args.text)
    st.save(Path(args.store))
    _print({"remembered": key, "facets": st.top_facets(key or "", 8)})
    return 0 if key else 1


def cmd_forget(args) -> int:
    st = _load(args.store)
    removed = []
    for key in (args.core, args.core + "#p"):
        if key in st.crosses:
            del st.crosses[key]
            st.core_count.pop(key, None)
            removed.append(key)
    st.save(Path(args.store))
    _print({"forgot": removed})
    return 0 if removed else 1


def cmd_ask(args) -> int:
    st = _load(args.store)
    _print(_route(st, args.query))
    return 0


def cmd_stats(args) -> int:
    st = _load(args.store)
    top = sorted(st.core_count.items(), key=lambda kv: (-kv[1], kv[0]))[:10]
    _print({**st.report(), "top_cores": top})
    return 0


def cmd_chat(args) -> int:
    from .router import route as harness_route

    st = _load(args.store)
    store_path = Path(args.store)

    llm_fn = None
    if args.mode == "hybrid":
        from .llm_local import ollama_available, ollama_generate

        if ollama_available():
            model = args.llm

            def llm_fn(prompt, system):  # noqa: E731 — closure over model
                return ollama_generate(model, prompt, system=system)

            print(f"[hybrid] local model '{model}' under Vera control "
                  "(vera / llm_guided / llm_free / refused)")
        else:
            print("[hybrid] Ollama not reachable at localhost:11434 — "
                  "falling back to lab mode (deterministic only)")

    auto_mem = not args.no_auto_memory
    print("Verantyx Vera α — "
          f"mode={args.mode}, lang={args.lang}, auto-memory={'on' if auto_mem else 'off'}. "
          "Commands: :remember <text>, :forget <core>, :stats, :quit")
    while True:
        try:
            line = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in (":quit", ":q", "exit"):
            break
        if line.startswith(":remember "):
            from .lang import ingest_text

            rep = ingest_text(st, line[len(":remember "):], lang=args.lang)
            st.save(store_path)
            print(f"vera> remembered {rep['cores']} [{rep['lang']}]")
            continue
        if line.startswith(":forget "):
            core = line[len(":forget "):].strip()
            n = 0
            for key in (core, core + "#p"):
                if key in st.crosses:
                    del st.crosses[key]
                    st.core_count.pop(key, None)
                    n += 1
            st.save(store_path)
            print(f"vera> forgot {n} cross(es)")
            continue
        if line == ":stats":
            print(f"vera> {st.report()}")
            continue

        out = harness_route(
            st, line,
            llm=llm_fn,
            lang=args.lang,
            auto_memory=auto_mem,
            save=lambda: st.save(store_path),
        )
        mem = out.get("remembered")
        if mem and mem.get("cores"):
            print(f"      (remembered: {mem['cores']} [{mem['lang']}])")
        src = out.get("source", "vera")
        if src in ("llm_guided", "llm_free"):
            tag = "llm←vera-facts" if src == "llm_guided" else "llm UNVERIFIED"
            print(f"vera> {out.get('surface','')}   [{tag}]")
        elif out.get("verdict") == "ANSWER":
            body = (
                out.get("text")
                or out.get("value")
                or out.get("x")
                or out.get("callers")
                or out.get("calls")
                or out.get("impacted")
            )
            print(f"vera> {body}   [{out.get('route','?')}]")
        else:
            print(f"vera> {out.get('verdict')}   [{out.get('route','?')}] "
                  "(I do not know — no guessing)")
    return 0


def cmd_math(args) -> int:
    _print(math_ask(args.query))
    return 0


def cmd_simplify(args) -> int:
    _print(simplify(args.expr, default_algebra_rules()))
    return 0


def cmd_code(args) -> int:
    st = _load(args.store)
    if args.action == "ingest":
        rep = ingest_python_repo(st, Path(args.target))
        st.save(Path(args.store))
        _print({**rep, "store": args.store})
        return 0
    _print(code_ask(st, args.target))
    return 0


def cmd_lab(args) -> int:
    from .consensus_forks import all_consensus_forks
    from .kripke_rewrite_forks import all_kripke_rewrite_forks
    from .lang_router_forks import all_lang_router_forks
    from .math_sim_forks import all_math_sim_forks
    from .phase2_forks import all_phase2_forks
    from .pour_forks import all_pour_forks

    experiments = (
        all_consensus_forks()
        + all_pour_forks()
        + all_math_sim_forks()
        + all_kripke_rewrite_forks()
        + all_lang_router_forks()
        + all_phase2_forks()
    )
    forks = {e["fork"]: e["pass"] for e in experiments}
    all_pass = all(forks.values())
    _print({"all_pass": all_pass, "n_forks": len(forks), "forks": forks})
    return 0 if all_pass else 1


def cmd_mcp(args) -> int:
    from .mcp_server import serve

    return serve(args.store)


def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(prog="vera", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--store", default=DEFAULT_STORE)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("pour", help="stream a corpus into the store")
    p.add_argument("--source", default="synthetic")
    p.add_argument("--max-rows", type=int, default=20000)
    p.add_argument("--max-sentences", type=int, default=None)
    p.add_argument("--checkpoint-every", type=int, default=100000)
    p.add_argument("--no-two-pass", action="store_true")
    p.set_defaults(fn=cmd_pour)

    p = sub.add_parser("remember", help="teach one sentence")
    p.add_argument("text")
    p.set_defaults(fn=cmd_remember)

    p = sub.add_parser("forget", help="delete a core (really deletes)")
    p.add_argument("core")
    p.set_defaults(fn=cmd_forget)

    p = sub.add_parser("ask", help="one-shot question (typed verdict)")
    p.add_argument("query")
    p.set_defaults(fn=cmd_ask)

    p = sub.add_parser("stats", help="store statistics")
    p.set_defaults(fn=cmd_stats)

    p = sub.add_parser("chat", help="interactive REPL (lab | hybrid)")
    p.add_argument("--mode", choices=["lab", "hybrid"], default="lab",
                   help="lab: deterministic only; hybrid: local LLM under Vera control")
    p.add_argument("--llm", default="llama3.2",
                   help="Ollama model name for hybrid mode")
    p.add_argument("--lang", default="auto",
                   help="auto | en | ja | es | fr | de | latin")
    p.add_argument("--no-auto-memory", action="store_true",
                   help="disable the native always-on memory harness")
    p.set_defaults(fn=cmd_chat)

    p = sub.add_parser("math", help="wire arithmetic / typed equations")
    p.add_argument("query")
    p.set_defaults(fn=cmd_math)

    p = sub.add_parser("simplify", help="term rewriting (algebra rules)")
    p.add_argument("expr")
    p.set_defaults(fn=cmd_simplify)

    p = sub.add_parser("code", help="code reasoning: ingest / ask")
    p.add_argument("action", choices=["ingest", "ask"])
    p.add_argument("target")
    p.set_defaults(fn=cmd_code)

    p = sub.add_parser("lab", help="run fork self-test suites")
    p.set_defaults(fn=cmd_lab)

    p = sub.add_parser("mcp", help="start MCP server (stdio)")
    p.set_defaults(fn=cmd_mcp)

    args = ap.parse_args(argv)
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
