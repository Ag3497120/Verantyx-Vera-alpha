"""Agent tools — the hands and feet (手足) of agent mode.

Every tool is a plain function with declared mutability; the agent loop
gates every *mutating* tool behind arrow-key approval. Read-only tools run
freely. Vera's own faculties (ask / remember / math / code) are tools too,
so the ReAct loop can consult the deterministic core at any step.

Note on web search: the Verantyx IDE's BrowserBridge is JCross-vaulted
(deliberately opaque), so this is an independent Python counterpart —
DuckDuckGo HTML search + plain page fetch over stdlib urllib.
"""
from __future__ import annotations

import html
import json
import re
import subprocess
import urllib.parse
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from .cross_store import CrossStore

_UA = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                     "AppleWebKit/537.36 (KHTML, like Gecko)"}


@dataclass
class Tool:
    name: str
    description: str
    mutating: bool
    fn: Callable[..., Dict[str, Any]]
    args_hint: str = ""


# ---------------------------------------------------------------------------
# filesystem
# ---------------------------------------------------------------------------

def read_file(path: str, max_chars: int = 8000) -> Dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        return {"ok": False, "error": f"not_a_file: {path}"}
    text = p.read_text(errors="replace")
    return {"ok": True, "path": str(p), "chars": len(text),
            "content": text[:max_chars]}


def write_file(path: str, content: str) -> Dict[str, Any]:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content)
    return {"ok": True, "path": str(p), "chars": len(content)}


def edit_file(path: str, old: str, new: str) -> Dict[str, Any]:
    p = Path(path)
    if not p.is_file():
        return {"ok": False, "error": f"not_a_file: {path}"}
    text = p.read_text(errors="replace")
    n = text.count(old)
    if n == 0:
        return {"ok": False, "error": "old_string_not_found"}
    if n > 1:
        return {"ok": False, "error": f"old_string_not_unique ({n} matches)"}
    p.write_text(text.replace(old, new, 1))
    return {"ok": True, "path": str(p), "replaced": 1}


def list_dir(path: str = ".") -> Dict[str, Any]:
    p = Path(path)
    if not p.is_dir():
        return {"ok": False, "error": f"not_a_dir: {path}"}
    entries = sorted(x.name + ("/" if x.is_dir() else "") for x in p.iterdir())
    return {"ok": True, "path": str(p), "entries": entries[:200]}


def make_dir(path: str) -> Dict[str, Any]:
    Path(path).mkdir(parents=True, exist_ok=True)
    return {"ok": True, "path": path}


# ---------------------------------------------------------------------------
# shell
# ---------------------------------------------------------------------------

def run_command(command: str, timeout: int = 60) -> Dict[str, Any]:
    try:
        r = subprocess.run(
            command, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return {
            "ok": r.returncode == 0,
            "returncode": r.returncode,
            "stdout": r.stdout[-4000:],
            "stderr": r.stderr[-2000:],
        }
    except subprocess.TimeoutExpired:
        return {"ok": False, "error": f"timeout_{timeout}s"}


# ---------------------------------------------------------------------------
# web (Python counterpart of the vaulted BrowserBridge)
# ---------------------------------------------------------------------------

# DDG lite endpoint: <a ... href="URL" class='result-link'>TITLE</a>
_RESULT = re.compile(
    r'<a[^>]+href="([^"]+)"[^>]*class=[\'"]result-link[\'"][^>]*>(.*?)</a>', re.S
)
_SNIPPET = re.compile(r'class=[\'"]result-snippet[\'"][^>]*>(.*?)</td>', re.S)
_TAG = re.compile(r"<[^>]+>")


def web_search(query: str, k: int = 5) -> Dict[str, Any]:
    url = "https://lite.duckduckgo.com/lite/?q=" + urllib.parse.quote(query)
    req = urllib.request.Request(url, headers=_UA)
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            page = r.read().decode(errors="replace")
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    links = _RESULT.findall(page)
    snips = _SNIPPET.findall(page)
    out: List[Dict[str, str]] = []
    for i, (href, title) in enumerate(links[:k]):
        if "uddg=" in href:
            href = urllib.parse.unquote(
                href.split("uddg=", 1)[1].split("&", 1)[0]
            )
        out.append(
            {
                "title": html.unescape(_TAG.sub("", title)).strip(),
                "url": href,
                "snippet": html.unescape(_TAG.sub("", snips[i])).strip()[:200]
                if i < len(snips) else "",
            }
        )
    return {"ok": True, "query": query, "results": out}


def fetch_url(url: str, max_chars: int = 6000) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers=_UA)
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            page = r.read().decode(errors="replace")
    except Exception as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}
    text = _TAG.sub(" ", re.sub(r"<(script|style).*?</\1>", " ", page, flags=re.S))
    text = html.unescape(re.sub(r"\s+", " ", text)).strip()
    return {"ok": True, "url": url, "text": text[:max_chars]}


# ---------------------------------------------------------------------------
# registry
# ---------------------------------------------------------------------------

def build_registry(store: CrossStore, save: Callable[[], None]) -> Dict[str, Tool]:
    from .code_ingest import code_ask, ingest_python_repo
    from .consensus_store import consensus_over_store
    from .math_sim import math_ask

    def vera_ask(query: str) -> Dict[str, Any]:
        return consensus_over_store(store, query)

    def vera_remember(sentence: str) -> Dict[str, Any]:
        key = store.ingest_sentence(sentence)
        save()
        return {"remembered": key}

    def vera_recall(core: str, k: int = 8) -> Dict[str, Any]:
        hits = {
            key: store.top_facets(key, int(k))
            for key in (core, core + "#p")
            if store.has(key)
        }
        return hits or {"verdict": "UNKNOWN_NO_EVIDENCE"}

    def vera_code_ingest(path: str) -> Dict[str, Any]:
        rep = ingest_python_repo(store, Path(path))
        save()
        return rep

    def vera_code_query(query: str) -> Dict[str, Any]:
        return code_ask(store, query)

    def vera_math(query: str) -> Dict[str, Any]:
        return math_ask(query)

    tools = [
        Tool("read_file", "Read a text file", False, read_file, "path"),
        Tool("list_dir", "List a directory", False, list_dir, "path"),
        Tool("write_file", "Create/overwrite a file", True, write_file,
             "path, content"),
        Tool("edit_file", "Replace a unique string in a file", True, edit_file,
             "path, old, new"),
        Tool("make_dir", "Create a directory", True, make_dir, "path"),
        Tool("run_command", "Run a shell command", True, run_command, "command"),
        Tool("web_search", "DuckDuckGo search", False, web_search, "query"),
        Tool("fetch_url", "Fetch a web page as text", False, fetch_url, "url"),
        Tool("vera_ask", "Ask the deterministic knowledge store", False,
             vera_ask, "query"),
        Tool("vera_remember", "Store a fact permanently", True, vera_remember,
             "sentence"),
        Tool("vera_recall", "Dump a concept's facets", False, vera_recall,
             "core"),
        Tool("vera_code_ingest", "AST-ingest a Python repo", True,
             vera_code_ingest, "path"),
        Tool("vera_code_query", "who calls X / impact of X", False,
             vera_code_query, "query"),
        Tool("vera_math", "Exact arithmetic / equations", False, vera_math,
             "query"),
    ]
    return {t.name: t for t in tools}


def tools_manifest(registry: Dict[str, Tool]) -> str:
    lines = []
    for t in registry.values():
        mut = " [needs approval]" if t.mutating else ""
        lines.append(f"- {t.name}({t.args_hint}): {t.description}{mut}")
    return "\n".join(lines)
