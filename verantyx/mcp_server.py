"""MCP server — hallucination-free memory & knowledge tools over stdio.

Exposes the cross-structure store to MCP clients (Claude Code, Claude
Desktop, …). Every tool returns typed verdicts; `ask` refuses instead of
guessing, and `forget` really deletes (knowledge is not baked into weights).

Requires the official MCP Python SDK:  pip install "mcp[cli]"
Client setup: see docs/MCP.md.
"""
from __future__ import annotations

import json
from pathlib import Path

from . import boundary, domains
from .agent_tools import fetch_url, web_search
from .ai_ingest import AiFactQuarantine
from .consensus_store import consensus_over_store
from .cross_store import CrossStore
from .code_ingest import code_ask, ingest_python_repo
from .gap_graph import GapGraph, gap_graph_path
from .growth_signals import GrowthSignals, growth_signals_path
from .llm_local import ollama_available
from .math_sim import math_ask
from .module_forge import build_test_queries, draft_module
from .module_ingest import DomainModuleQuarantine
from .module_verify import verify_module


def serve(store_path: str) -> int:
    try:
        from mcp.server.fastmcp import FastMCP
    except ImportError:
        print('MCP SDK not installed. Run:  pip install "mcp[cli]"')
        return 2

    path = Path(store_path)
    store = CrossStore.load(path) if path.is_file() else CrossStore()
    qpath = path.with_name(path.stem + ".ai_quarantine.json")
    quarantine = AiFactQuarantine.load(qpath)

    pkg_dir = Path(__file__).resolve().parent
    domains.register_builtins()
    domains.register_generated(pkg_dir)
    gpath = growth_signals_path(path)
    growth = GrowthSignals.load(gpath)
    mqpath = path.with_name(path.stem + ".module_quarantine.json")
    module_quarantine = DomainModuleQuarantine.load(mqpath)
    ggpath = gap_graph_path(path)
    gap_graph = GapGraph.load(ggpath)

    def _save_gap_graph() -> None:
        gap_graph.save(ggpath)

    mcp = FastMCP("verantyx-vera")

    def _save() -> None:
        store.save(path)

    def _save_growth() -> None:
        growth.save(gpath)

    @mcp.tool()
    def ask(query: str) -> str:
        """Ask the knowledge store. Returns a typed verdict — ANSWER with
        provenance-backed facets, or UNKNOWN_* (never a guess)."""
        out = consensus_over_store(store, query)
        verdict = out.get("verdict")
        if isinstance(verdict, str) and verdict.startswith("UNKNOWN"):
            growth.record_unknown(query, verdict)
            _save_growth()
        return json.dumps(out, ensure_ascii=False)

    @mcp.tool()
    def remember(sentence: str) -> str:
        """Teach one English sentence. It is classified into core + facets
        and accumulated deterministically (usable immediately)."""
        key = store.ingest_sentence(sentence)
        _save()
        return json.dumps(
            {"remembered": key, "facets": store.top_facets(key or "", 8)},
            ensure_ascii=False,
        )

    @mcp.tool()
    def record_code_change(file_path: str, description: str) -> str:
        """Record a code change (file path + what changed) as a structured
        fact — NOT run through sentence-splitting quarantine, unlike
        propose_ai_facts. This exists because AiFactQuarantine's sentence
        splitter (split on '.', '!', '?') mangles diff/patch syntax (e.g.
        splits mid-diff at every '.py' or decimal number). A file having
        been written/edited is an already-executed, verifiable action —
        not an unverified claim — so it goes straight into the store under
        a dedicated core, distinct from the trusted natural-language facts
        `remember` accumulates."""
        key = f"code_change:{file_path}".casefold()
        store.add(key, ["changed", f"desc:{description[:300]}"])
        _save()
        return json.dumps({"recorded": key}, ensure_ascii=False)

    @mcp.tool()
    def record_verified_url(name: str, url: str) -> str:
        """Register a human- or agent-confirmed URL for a named destination
        (e.g. name="Gemini", url="https://gemini.google.com/"). Stored
        directly as a facet, NOT run through the sentence-splitting
        quarantine `remember`/`propose_ai_facts` use — a URL's periods
        would get mangled the same way `record_code_change`'s docstring
        describes for diffs. Pair with `lookup_verified_url`, which reads
        it back deterministically (no consensus threshold), so an agent
        can check for a known-good URL before guessing one from its own
        training data."""
        key = f"verified_url:{name.strip().casefold()}"
        store.add(key, [f"url:{url.strip()}"])
        _save()
        return json.dumps({"recorded": key, "url": url.strip()}, ensure_ascii=False)

    @mcp.tool()
    def lookup_verified_url(name: str) -> str:
        """Deterministic lookup for a URL registered via
        `record_verified_url` -- bypasses `ask`'s consensus/agreement
        threshold entirely (a single registration is enough), so this
        never spuriously returns UNKNOWN just because there's only one
        piece of evidence. Returns {"verdict": "ANSWER", "url": ...} or
        {"verdict": "UNKNOWN_NO_EVIDENCE"}."""
        key = f"verified_url:{name.strip().casefold()}"
        if not store.has(key):
            return json.dumps({"verdict": "UNKNOWN_NO_EVIDENCE", "name": name}, ensure_ascii=False)
        facets = store.top_facets(key, 1)
        for facet, _ in facets:
            if facet.startswith("url:"):
                return json.dumps({"verdict": "ANSWER", "name": name, "url": facet[len("url:"):]}, ensure_ascii=False)
        return json.dumps({"verdict": "UNKNOWN_NO_EVIDENCE", "name": name}, ensure_ascii=False)

    @mcp.tool()
    def record_verified_ui_element(app: str, element: str, x: float, y: float, version: str = "") -> str:
        """Register a confirmed UI element location within `app`'s window,
        as (x, y) normalized to 0-1000 relative to the window's own bounds
        (matching HiddenWindowAutomation/DesktopVisionBridge's coordinate
        convention) -- so a repeat operation can click it directly instead
        of re-running screenshot + vision analysis every time. Same
        not-sentence-split reasoning as `record_verified_url`.

        `version` (optional) is the app's bundle/build version at
        registration time. A registration REPLACES any prior coord/version
        for this exact (app, element) rather than accumulating alongside
        it -- an element's location is current-state, not a frequency-
        weighted fact, so `top_facets` must never return a stale coord
        just because it was registered more times than a corrected one.
        The caller (Verantyx) compares this stored version against the
        app's current version on lookup to decide whether a cached
        location needs re-verification."""
        key = f"ui_element:{app.strip().casefold()}:{element.strip().casefold()}"
        if key in store.crosses:
            store.crosses[key] = {
                f: c for f, c in store.crosses[key].items()
                if not (f.startswith("coord:") or f.startswith("version:"))
            }
        facets = [f"coord:{x},{y}"]
        if version.strip():
            facets.append(f"version:{version.strip()}")
        store.add(key, facets)
        _save()
        return json.dumps({"recorded": key, "x": x, "y": y, "version": version.strip()}, ensure_ascii=False)

    @mcp.tool()
    def lookup_verified_ui_element(app: str, element: str) -> str:
        """Deterministic lookup for a UI element registered via
        `record_verified_ui_element`. Returns {"verdict": "ANSWER", "x":
        ..., "y": ..., "version": ...} (version is "" if none was
        recorded) or {"verdict": "UNKNOWN_NO_EVIDENCE"}."""
        key = f"ui_element:{app.strip().casefold()}:{element.strip().casefold()}"
        if not store.has(key):
            return json.dumps({"verdict": "UNKNOWN_NO_EVIDENCE", "app": app, "element": element}, ensure_ascii=False)
        x = y = None
        version = ""
        for facet, _ in store.top_facets(key, 8):
            if facet.startswith("coord:"):
                x_str, y_str = facet[len("coord:"):].split(",", 1)
                x, y = float(x_str), float(y_str)
            elif facet.startswith("version:"):
                version = facet[len("version:"):]
        if x is None:
            return json.dumps({"verdict": "UNKNOWN_NO_EVIDENCE", "app": app, "element": element}, ensure_ascii=False)
        return json.dumps(
            {"verdict": "ANSWER", "app": app, "element": element, "x": x, "y": y, "version": version},
            ensure_ascii=False,
        )

    @mcp.tool()
    def list_verified_ui_elements(app: str) -> str:
        """Lists every UI element registered for `app` via
        `record_verified_ui_element` -- used by a re-verification pass to
        know what to re-check for a given app, without needing the caller
        to already know each element's name. Each entry includes the
        version recorded at registration time (if any)."""
        prefix = f"ui_element:{app.strip().casefold()}:"
        elements = []
        for key in store.crosses:
            if not key.startswith(prefix):
                continue
            name = key[len(prefix):]
            x = y = None
            version = ""
            for facet, _ in store.top_facets(key, 8):
                if facet.startswith("coord:"):
                    x_str, y_str = facet[len("coord:"):].split(",", 1)
                    x, y = float(x_str), float(y_str)
                elif facet.startswith("version:"):
                    version = facet[len("version:"):]
            if x is not None:
                elements.append({"element": name, "x": x, "y": y, "version": version})
        return json.dumps({"app": app, "elements": elements}, ensure_ascii=False)

    @mcp.tool()
    def forget(core: str) -> str:
        """Delete a core cross entirely. Deletion is real and immediate —
        unlike model weights, nothing lingers."""
        removed = []
        for key in (core.casefold(), core.casefold() + "#p"):
            if key in store.crosses:
                del store.crosses[key]
                store.core_count.pop(key, None)
                removed.append(key)
        _save()
        return json.dumps({"forgot": removed}, ensure_ascii=False)

    @mcp.tool()
    def recall(core: str, k: int = 8) -> str:
        """Recall the accumulated facets (with counts) for a core."""
        hits = {}
        for key in (core.casefold(), core.casefold() + "#p"):
            if store.has(key):
                hits[key] = store.top_facets(key, k)
        if not hits:
            return json.dumps({"verdict": "UNKNOWN_NO_EVIDENCE", "core": core})
        return json.dumps({"verdict": "ANSWER", "crosses": hits}, ensure_ascii=False)

    @mcp.tool()
    def math(query: str) -> str:
        """Exact wire arithmetic / typed equation solving (never approximate:
        ANSWER, AMBIGUOUS, or UNKNOWN_*)."""
        return json.dumps(math_ask(query), ensure_ascii=False)

    @mcp.tool()
    def code_ingest(repo_path: str) -> str:
        """Ingest a Python repo (AST): one cross per function with
        file/class/args/calls facets."""
        rep = ingest_python_repo(store, Path(repo_path))
        _save()
        return json.dumps(rep, ensure_ascii=False)

    @mcp.tool()
    def code_query(query: str) -> str:
        """Code reasoning: 'who calls X' | 'what does X call' | 'impact of X'."""
        return json.dumps(code_ask(store, query), ensure_ascii=False)

    @mcp.tool()
    def stats() -> str:
        """Store statistics (cores, facet links, sentences ingested)."""
        return json.dumps(store.report(), ensure_ascii=False)

    @mcp.tool()
    def graph_snapshot(limit: int = 24, facets_per_core: int = 6, focus_cores: str = "") -> str:
        """Structural snapshot of the store for visualization -- NOT for
        grounded QA (that's `ask`). Returns the top `limit` cores ranked
        by pour count, each with up to `facets_per_core` of its top
        facets (with counts). Read-only; does not mutate the store.

        `focus_cores` is an optional comma-separated list of core keys
        (as returned by `remember`'s "remembered" field) that are always
        included even if their pour count is too low to make the top
        `limit` -- otherwise a just-taught fact, which starts at a pour
        count of 1, would never show up once the store has thousands of
        long-accumulated cores ranked above it."""
        ranked = sorted(
            store.core_count.items(), key=lambda kv: kv[1], reverse=True
        )[: max(0, limit)]
        ranked_keys = {core for core, _ in ranked}

        focus_keys = [c.strip().casefold() for c in focus_cores.split(",") if c.strip()]
        for key in focus_keys:
            if key in ranked_keys or key not in store.crosses:
                continue
            ranked.append((key, store.core_count.get(key, 0)))
            ranked_keys.add(key)

        nodes = []
        for core, count in ranked:
            facets = store.top_facets(core, facets_per_core)
            nodes.append({
                "core": core,
                "pour_count": count,
                "facets": [{"facet": f, "count": c} for f, c in facets],
            })
        return json.dumps(
            {"nodes": nodes, "total_cores": len(store.crosses)},
            ensure_ascii=False,
        )

    @mcp.tool()
    def propose_ai_facts(text: str, source: str = "ai_output") -> str:
        """Quarantine sentence-level fact candidates split out of an
        assistant's FINAL reply text — never pass a thinking/chain-of-
        thought block here. Hedge-worded ("might", "probably", "I think")
        and meta-commentary ("let me check") sentences are dropped before
        they even reach quarantine. Nothing proposed here is queryable via
        ask() until a human calls accept_ai_fact — this tool can never by
        itself put an unverified claim into the trusted store."""
        added = quarantine.propose(text, source=source)
        quarantine.save(qpath)
        return json.dumps(
            {"proposed": [e.text for e in added], "quarantine": str(qpath)},
            ensure_ascii=False,
        )

    @mcp.tool()
    def list_pending_ai_facts() -> str:
        """List AI-proposed facts awaiting human review (index, text,
        source, timestamp). Use the index with accept_ai_fact /
        reject_ai_fact."""
        pend = quarantine.pending()
        return json.dumps(
            [{"index": i, **e.as_dict()} for i, e in enumerate(pend)],
            ensure_ascii=False,
        )

    @mcp.tool()
    def accept_ai_fact(index: int) -> str:
        """Promote one pending AI-proposed fact (by index from
        list_pending_ai_facts) into the trusted, queryable store. This is
        the ONLY path from quarantine into real memory — a deliberate,
        explicit human action, never automatic."""
        pend = quarantine.pending()
        if not (0 <= index < len(pend)):
            return json.dumps({"ok": False, "error": "index_out_of_range"})
        key = quarantine.accept(pend[index], store)
        quarantine.save(qpath)
        store.save(path)
        return json.dumps({"ok": True, "core": key}, ensure_ascii=False)

    @mcp.tool()
    def reject_ai_fact(index: int) -> str:
        """Discard one pending AI-proposed fact (by index) — it is marked
        rejected and never enters the trusted store."""
        pend = quarantine.pending()
        if not (0 <= index < len(pend)):
            return json.dumps({"ok": False, "error": "index_out_of_range"})
        ok = quarantine.reject(pend[index])
        quarantine.save(qpath)
        return json.dumps({"ok": ok}, ensure_ascii=False)

    @mcp.tool()
    def heartbeat(llm_model: str = "", cognition_mode: str = "normal") -> str:
        """Milestone M's autonomous growth tick (closed-domain modules,
        unchanged) plus, when cognition_mode="sleep" (Milestone O), a
        second pass over gap_graph.json's open-domain GapNodes: attempts
        acquisition (web_search/fetch_url) for actionable nodes and
        proposes results into the SAME ai_quarantine.json queue
        propose_ai_facts already uses (see ai_ingest.AiFactQuarantine) --
        never writes directly to the trusted store. cognition_mode=
        "normal"/"experiment" skip this second pass entirely (no gap
        resolution attempted, matching router.py's own no-op guarantee
        for "normal" and "experiment"'s "detect only, don't resolve"
        contract)."""
        drifted = growth.record_mass_snapshot(store)
        candidates = []
        drafted = []
        for bucket in growth.buckets.values():
            verdict = boundary.classify(bucket)
            if verdict.classification != "growth_candidate":
                continue
            candidates.append({"normalized": bucket.normalized, "reason": verdict.reason})
            if not llm_model or not ollama_available():
                continue
            draft = draft_module(bucket, llm_model)
            if not draft["ok"]:
                drafted.append({"normalized": bucket.normalized, "ok": False, "error": draft["error"]})
                continue
            test_queries = build_test_queries(bucket)
            ok, reports = verify_module(draft["source"], test_queries, store, draft["name"])
            report_dicts = [r.as_dict() for r in reports]
            if ok:
                module_quarantine.propose(
                    draft["name"], draft["source"], bucket.normalized, report_dicts,
                )
                module_quarantine.save(mqpath)
                drafted.append({"normalized": bucket.normalized, "ok": True, "name": draft["name"], "queued": True})
            else:
                drafted.append({"normalized": bucket.normalized, "ok": False, "verify_reports": report_dicts})
        _save_growth()

        gap_results: list = []
        if cognition_mode == "sleep":
            from .config import VeraConfig

            cfg = VeraConfig.load()
            for node in gap_graph.actionable(limit=cfg.gap_max_new_nodes_per_run):
                if node.status == "BLOCKED_POLICY":
                    continue
                gap_graph.set_status(node.gap_id, "ACQUIRING")
                evidence_text = None
                source_used = None
                if "web_search" in node.acquisition_methods:
                    sr = web_search(node.subject)
                    if sr.get("ok") and sr.get("results"):
                        top = sr["results"][0]
                        fr = fetch_url(top.get("url", ""))
                        if fr.get("ok"):
                            evidence_text = fr.get("text", "")[:2000]
                            source_used = top.get("url")
                if evidence_text:
                    gap_graph.set_status(node.gap_id, "EVIDENCE_COLLECTED")
                    quarantine.propose_raw(
                        f"[gap:{node.gap_id}] {node.subject}\n\n{evidence_text}",
                        source=f"sleep_mode_gap_resolution:{source_used or 'unknown'}",
                    )
                    quarantine.save(qpath)
                    gap_graph.set_status(node.gap_id, "RESOLVED",
                                          resolution=f"quarantined:{source_used}")
                    gap_results.append({"gap_id": node.gap_id, "subject": node.subject,
                                        "status": "RESOLVED", "queued_for_review": True})
                else:
                    gap_graph.set_status(node.gap_id, "BLOCKED_NO_SOURCE")
                    gap_results.append({"gap_id": node.gap_id, "subject": node.subject,
                                        "status": "BLOCKED_NO_SOURCE"})
            _save_gap_graph()

        return json.dumps(
            {"drifted_cores": drifted, "growth_candidates": candidates, "drafted": drafted,
             "gap_resolutions": gap_results},
            ensure_ascii=False,
        )

    @mcp.tool()
    def propose_domain_module(name: str, source_code: str, candidate_summary: str) -> str:
        """Manually quarantine a hand- or externally-drafted domain module
        (bypassing heartbeat's LLM step) — still runs through the same
        verification gates as heartbeat's auto-drafted candidates before
        it can be queued. Nothing here is live until accept_domain_module."""
        test_queries = [candidate_summary]
        ok, reports = verify_module(source_code, test_queries, store, name)
        report_dicts = [r.as_dict() for r in reports]
        if not ok:
            return json.dumps({"ok": False, "verify_reports": report_dicts}, ensure_ascii=False)
        module_quarantine.propose(name, source_code, candidate_summary, report_dicts)
        module_quarantine.save(mqpath)
        return json.dumps({"ok": True, "queued": True}, ensure_ascii=False)

    @mcp.tool()
    def list_pending_domain_modules() -> str:
        """List LLM-drafted domain modules awaiting human review (index,
        name, source code, verify test report, candidate summary). Use the
        index with accept_domain_module / reject_domain_module."""
        pend = module_quarantine.pending()
        return json.dumps(
            [{"index": i, **e.as_dict()} for i, e in enumerate(pend)],
            ensure_ascii=False,
        )

    @mcp.tool()
    def accept_domain_module(index: int) -> str:
        """Promote one pending domain module (by index from
        list_pending_domain_modules) into the live domain registry and
        write it to verantyx/domains/generated/. The ONLY path from
        quarantine to an active module — never automatic."""
        pend = module_quarantine.pending()
        if not (0 <= index < len(pend)):
            return json.dumps({"ok": False, "error": "index_out_of_range"})
        module_path = module_quarantine.accept(pend[index], pkg_dir)
        module_quarantine.save(mqpath)
        return json.dumps({"ok": module_path is not None, "path": module_path}, ensure_ascii=False)

    @mcp.tool()
    def reject_domain_module(index: int) -> str:
        """Discard one pending domain module (by index) — marked rejected,
        never written to disk or registered."""
        pend = module_quarantine.pending()
        if not (0 <= index < len(pend)):
            return json.dumps({"ok": False, "error": "index_out_of_range"})
        ok = module_quarantine.reject(pend[index])
        module_quarantine.save(mqpath)
        return json.dumps({"ok": ok}, ensure_ascii=False)

    @mcp.tool()
    def wake_summary(since_seconds: float = 43200.0) -> str:
        """Milestone O: 'what changed while you were away' — GapNodes whose
        status changed within the last `since_seconds` (default 12h),
        split by outcome, plus a reminder of how many items are sitting in
        the two existing review queues (list_pending_ai_facts /
        list_pending_domain_modules — this tool doesn't duplicate their
        content, just points at them)."""
        import time as _time

        cutoff = _time.time() - since_seconds
        changed = gap_graph.since(cutoff)
        resolved = [n.as_dict() for n in changed if n.status == "RESOLVED"]
        still_open = [n.as_dict() for n in changed if n.status not in ("RESOLVED",) and n.status not in
                      ("BLOCKED_POLICY", "BLOCKED_NO_SOURCE", "BLOCKED_PERMISSION",
                       "BLOCKED_BUDGET", "BLOCKED_CONTRADICTION", "STALE")]
        blocked = [n.as_dict() for n in changed if n.status.startswith("BLOCKED_") or n.status == "STALE"]
        return json.dumps({
            "resolved": resolved, "still_open": still_open, "blocked": blocked,
            "pending_fact_review": len(quarantine.pending()),
            "pending_module_review": len(module_quarantine.pending()),
        }, ensure_ascii=False)

    mcp.run()
    return 0
