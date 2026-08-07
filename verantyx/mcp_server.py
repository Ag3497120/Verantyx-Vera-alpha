"""MCP server — hallucination-free memory & knowledge tools over stdio.

Exposes the cross-structure store to MCP clients (Claude Code, Claude
Desktop, …). Every tool returns typed verdicts; `ask` refuses instead of
guessing, and `forget` really deletes (knowledge is not baked into weights).

Requires the official MCP Python SDK:  pip install "mcp[cli]"
Client setup: see docs/MCP.md.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from . import boundary, domains
from .agent_tools import build_registry, fetch_url, git_clone_scratch, git_clone_scratch_dir, web_search
from .ai_ingest import AiFactQuarantine
from .consensus_store import consensus_over_store
from .cross_store import CrossStore
from .code_ingest import code_ask, ingest_python_repo
from .arc_env_adapter import find_transferable_matches, observe_transition
from .ui_transition import observe_ui_transition
from .gap_graph import GapGraph, gap_graph_path
from .structural_similarity import find_structural_matches
from .task_bootstrap import TaskDescriptor
from .task_bootstrap import bootstrap_unknown_task as _bootstrap_unknown_task
from .task_bootstrap import select_next_action as _select_next_action
from .tool_call_quarantine import ToolCallQuarantine, tool_call_quarantine_path
from .transfer_outcomes import (
    TransferOutcomeLog,
    infer_missing_outcomes,
    record_transfer_attempt,
    record_transfer_outcome,
    transfer_outcome_log_path,
)
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
    from .capacity_ingest import CapacityQuarantine
    cqpath = path.with_name(path.stem + ".capacity_quarantine.json")
    capacity_quarantine = CapacityQuarantine.load(cqpath)
    from .pack_ingest import PackQuarantine
    # Overlay lives beside the store, so an expert's corrections travel with
    # their data rather than with the app bundle (which a reinstall replaces).
    pack_overlay_dir = path.parent / "failure_packs"
    pkpath = path.with_name(path.stem + ".pack_quarantine.json")
    pack_quarantine = PackQuarantine.load(pkpath)
    from .failure_domains import BUILTIN_PACK_DIR as _BPD, reload_from as _reload
    if pack_overlay_dir.is_dir():
        _reload([_BPD, pack_overlay_dir])
    # ── Japanese grammar overlay ─────────────────────────────────────────
    # Same pattern as the failure packs: an expert's vocabulary lives
    # beside their store, not inside the app bundle a reinstall replaces.
    # An invalid overlay refuses to load with every problem named — running
    # with half a grammar is the silent version of the same failure.
    from . import ja_grammar as _jag
    _grammar_overlay = path.parent / "ja_grammar.json"
    _grammar_overlay_error = ""
    if _grammar_overlay.is_file():
        try:
            _jag.load_overlay(_grammar_overlay)
        except (ValueError, OSError) as exc:
            _grammar_overlay_error = str(exc)

    ggpath = gap_graph_path(path)
    gap_graph = GapGraph.load(ggpath)

    def _save_gap_graph() -> None:
        gap_graph.save(ggpath)

    xopath = transfer_outcome_log_path(path)
    transfer_log = TransferOutcomeLog.load(xopath)

    def _save_transfer_log() -> None:
        transfer_log.save(xopath)

    tcqpath = tool_call_quarantine_path(path)
    tool_call_quarantine = ToolCallQuarantine.load(tcqpath)

    def _save_tool_call_quarantine() -> None:
        tool_call_quarantine.save(tcqpath)

    mcp = FastMCP("verantyx-vera")

    def _save() -> None:
        store.save(path)

    def _save_growth() -> None:
        growth.save(gpath)

    # Milestone R4: no browser_endpoint here (this is the headless MCP
    # process, not the IDE) -- jgen_reflect will just report its own
    # typed "no_jgen_endpoint_configured" if an accepted call happens to
    # be that tool. Every other tool works identically to how the agent
    # itself would have run it.
    tool_registry = build_registry(store, _save, browser_endpoint=None)

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
                # Real-usage bug found live: this loop only ever knew how
                # to try web_search, so a repo-study gap (acquisition_
                # methods = ["vera_git_clone", "vera_code_ingest", ...])
                # always fell straight through to BLOCKED_NO_SOURCE, even
                # though vera_git_clone/vera_code_ingest are both perfectly
                # able to resolve it. Cloning is safe to run automatically
                # here (writes only to Vera's own scratch dir, no trust
                # implications) -- but ingestion writes directly into the
                # TRUSTED store, so Sleep mode does NOT call it itself;
                # it queues vera_code_ingest into the SAME tool_call_
                # quarantine a human already reviews via
                # list_pending_tool_calls/accept_tool_call, exactly the
                # same "never silently promote to trusted" rule every
                # other Sleep-mode resolution in this loop already follows.
                if "vera_git_clone" in node.acquisition_methods:
                    m = re.search(r"https://github\.com/[\w.-]+/[\w.-]+", node.subject)
                    if m:
                        clone = git_clone_scratch(m.group(0), scratch_dir=git_clone_scratch_dir())
                        if clone.get("ok"):
                            entry = tool_call_quarantine.propose(
                                "vera_code_ingest", {"path": clone["path"]},
                                reason="Sleep-mode heartbeat: repo cloned, ingestion queued for review",
                                task=node.subject,
                            )
                            _save_tool_call_quarantine()
                            gap_graph.set_status(
                                node.gap_id, "RESOLUTION_PLANNED",
                                resolution=f"cloned to {clone['path']}, queued vera_code_ingest as {entry.call_id}",
                            )
                            gap_results.append({
                                "gap_id": node.gap_id, "subject": node.subject,
                                "status": "RESOLUTION_PLANNED", "queued_tool_call": entry.call_id,
                            })
                            continue
                    gap_graph.set_status(node.gap_id, "BLOCKED_NO_SOURCE")
                    gap_results.append({"gap_id": node.gap_id, "subject": node.subject,
                                        "status": "BLOCKED_NO_SOURCE"})
                    continue
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

        # Milestone R3: close out any transfer-outcome records whose target
        # gap has since settled (RESOLVED/BLOCKED_*) without an explicit
        # human judgment — this is what keeps list_transfer_outcomes from
        # staying permanently full of "unjudged" entries when nobody calls
        # record_transfer_result by hand. Runs every heartbeat regardless
        # of cognition_mode (pure read of existing gap status, no new
        # action taken, so it's as safe as wake_summary's own read-only
        # scan).
        newly_judged = infer_missing_outcomes(transfer_log, gap_graph)
        if newly_judged:
            _save_transfer_log()

        # Capacity pass — the other half of the failure taxonomy. Buckets
        # whose dominant verdict says "a limit was hit" get their own failing
        # queries re-run at scaled limits; a limit that verifiably fixes them
        # is queued for human approval, never applied. The re-run doubles as
        # a check on the classification itself: a bucket a larger limit does
        # not fix comes back "reclassify", not as a bigger number.
        from .capacity_calibration import capacity_pass
        from .config import VeraConfig as _VC
        capacity = capacity_pass(
            list(growth.buckets.values()), _VC.load(), capacity_quarantine)
        if any(r.get("queued") for r in capacity):
            capacity_quarantine.save(cqpath)

        return json.dumps(
            {"drifted_cores": drifted, "growth_candidates": candidates, "drafted": drafted,
             "capacity": capacity,
             "gap_resolutions": gap_results,
             "transfer_outcomes_inferred": len(newly_judged)},
            ensure_ascii=False,
        )

    @mcp.tool()
    def record_build_failure(source: str, log_excerpt: str) -> str:
        """Classify a build/CI/conversion failure into a typed verdict
        (UNKNOWN_SIGNING / UNKNOWN_DEPENDENCY / UNKNOWN_MODEL_GEOMETRY /
        UNKNOWN_TEST / UNKNOWN_TIMEOUT / UNKNOWN_DISK / ...) and record it
        in the shared growth-signal store, bucketed by `source` (a stable
        pipeline label like "xcodebuild", "cargo", "jgen_convert").
        Recurrences then surface through failure_stats and the boundary
        classifier like every other typed unknown. Returns the verdict and
        the evidence line so the caller can display the diagnosis."""
        from .build_failure import record_build_failure as _record
        failure = _record(growth, source, log_excerpt)
        _save_growth()
        return json.dumps(
            {"verdict": failure.verdict, "evidence": failure.evidence,
             "note": failure.note},
            ensure_ascii=False,
        )

    # Conversation context as space. One Conversation per server process,
    # persisted beside the store — turns are knowledge, not a scrollback.
    from .conversation import Conversation as _Conv
    from .layer_stack import LayeredMemory as _LM
    _conv_path = path.with_name(path.stem + ".conversation")
    _conversation = _Conv(memory=_LM.load(_conv_path))

    @mcp.tool()
    def add_conversation_turn(speaker: str, text: str) -> str:
        """Ingest one utterance into the conversation's SPACE (not a window).
        Each sentence becomes a node tagged with speaker and turn index, so
        it is retrieved by relevance rather than recency and never falls out
        of a context window. Overflow, when it happens, freezes a layer —
        the turn stays consultable and is reported as FROZEN, never silently
        dropped. Returns which cores the turn touched and whether it caused
        a layer to freeze."""
        turn = _conversation.add_turn(speaker, text)
        _conversation.memory.save(_conv_path)
        return json.dumps(
            {"turn": turn.index, "speaker": turn.speaker, "cores": turn.cores,
             "stats": _conversation.stats()}, ensure_ascii=False)

    @mcp.tool()
    def locate_conversation_topic(topic: str) -> str:
        """Is a topic still in context? The typed answer an LLM cannot give:
        ACTIVE (top layer), FROZEN (overflowed to a lower layer but intact
        and answerable), or ABSENT (never discussed). FROZEN is not lost —
        that distinction is the point. Includes who mentioned it and in which
        turns."""
        return json.dumps(_conversation.locate(topic), ensure_ascii=False)

    @mcp.tool()
    def recall_conversation(query: str, carry: str = "A") -> str:
        """Answer from the conversation's own memory across every layer,
        oldest first (extended inference). `carry` controls whether the
        original query rides along to later layers: A full query, B previous
        answer only, C intent head only. B can drift and says so —
        UNKNOWN_DRIFT rather than a confident wrong answer."""
        return json.dumps(_conversation.recall(query, carry=carry),
                          ensure_ascii=False)

    @mcp.tool()
    def conversation_stats() -> str:
        """Turns, layers, total cores, speakers."""
        return json.dumps(_conversation.stats(), ensure_ascii=False)

    # Deep search over multiple sources. The arm index lives beside the
    # store so intent gating and gap-driven follow-up survive a restart.
    from .arm_schema import ArmIndex as _ArmIndex
    _arm_path = path.with_name(path.stem + ".arms.json")
    _arms = _ArmIndex.load(_arm_path)

    @mcp.tool()
    def ingest_documents(documents_json: str) -> str:
        """Ingest several source documents about one event and PRESERVE their
        disagreements. `documents_json` is a JSON list of
        {"source": "...", "text": "...", "published": "..."}.

        Each sentence is placed with its source attached and, where the
        wording is polar (open/closed, safe/dangerous, passable/blocked...),
        on the corresponding pole. Two sources that disagree therefore end
        up on opposite poles of one aspect, which the store detects on its
        own — unlike a summary, where the disagreement dissolves into
        fluency. Deterministic: no LLM is involved, so the same documents
        always yield the same report and a disputed claim stays citable."""
        from .document_ingest import Document, ingest_documents as _ingest
        try:
            raw = json.loads(documents_json)
        except json.JSONDecodeError as e:
            return json.dumps({"error": f"documents_json is not valid JSON: {e}"},
                              ensure_ascii=False)
        if not isinstance(raw, list):
            return json.dumps({"error": "documents_json must be a JSON list"},
                              ensure_ascii=False)
        docs = [Document(source=str(d.get("source", "unknown")),
                         text=str(d.get("text", "")),
                         published=str(d.get("published", "")))
                for d in raw if isinstance(d, dict)]
        if not store.track_provenance:
            # Attribution is the whole point here; without provenance the
            # report cannot say who claimed which side.
            store.track_provenance = True
        rep = _ingest(store, docs, arms=_arms)
        _save()
        _arms.save(_arm_path)
        return json.dumps(rep.as_dict(), ensure_ascii=False)

    @mcp.tool()
    def deep_report(topic: str) -> str:
        """What is actually known about a topic, in three UNBLENDED parts:

          settled    every source that spoke agrees (with citations)
          disputed   sources disagree — which source said which side
          missing    a question the arm schema says should have an answer
                     and no source gave one, each with `next_query`: the
                     search string that would close it

        `missing` is what makes this deep search rather than summarisation —
        a typed gap is the next round's query. `confidence` is contested /
        supported / unknown, the first thing a responder reads."""
        from .document_ingest import deep_report as _report
        return json.dumps(_report(store, topic, arms=_arms), ensure_ascii=False)

    @mcp.tool()
    def arm_completeness(topic: str) -> str:
        """The six-question checklist for a topic: which arms hold knowledge
        (support/oppose, cause/effect, general/instance) and which are empty.
        An empty arm is a typed gap — unverified, untestable, untransferable
        — ready to become a GapNode."""
        return json.dumps(_arms.report(topic), ensure_ascii=False)

    @mcp.tool()
    def list_failure_domains() -> str:
        """The registered failure-domain packs: name, maturity (verified =
        validated against confirmed incidents; seeded = taxonomy written,
        awaiting real incidents), the typed verdicts each can produce, and
        each verdict's remedy (kind / owner / how it is verified). This is
        the plugin surface of the typed-failure loop — the core is shared,
        packs supply classification and remedy knowledge per field."""
        from .failure_domains import all_domains, load_errors
        out = []
        for d in all_domains():
            out.append({
                "name": d.name, "maturity": d.maturity,
                "description": d.description,
                "editable": d.source_path is not None,
                "source_path": d.source_path,
                "verdicts": [
                    {"verdict": v, "note": note,
                     "provenance": d.provenance.get(v, "code"),
                     "remedy_kind": d.remedies[v].kind,
                     "remedy_owner": d.remedies[v].owner,
                     "verify": d.remedies[v].verify,
                     "auto_calibratable": d.remedies[v].auto_calibratable}
                    for v, _, note in d.patterns
                ],
            })
        # Load errors are part of the answer: a pack an expert edited into an
        # invalid state must be visible, not merely absent.
        return json.dumps({"packs": out, "load_errors": load_errors()},
                          ensure_ascii=False)

    @mcp.tool()
    def propose_failure_verdict(
        pack: str, verdict: str, note: str, positive_examples: str,
        negative_examples: str = "", remedy_kind: str = "fix_content",
        remedy_owner: str = "", verify: str = "rerun",
        remedy_note: str = "", author: str = "unknown",
    ) -> str:
        """Add a new typed verdict to a failure pack FROM EXAMPLES, not from
        a regular expression. Paste real failure lines (one per line in
        positive_examples; optional counter-examples in negative_examples)
        and a pattern is proposed from what they share.

        The proposal is refused, with the specific counter-example, if it
        matches a negative, misses a positive, or breaks the pack's
        maturity contract; existing fixtures it would also claim are
        reported as `shadowed_fixtures`. Nothing is written — the returned
        proposed_pack goes to review. `author` is stamped into provenance
        as human:<author>, which is how a claude-seeded taxonomy stops
        claiming to be mine once a domain expert has corrected it."""
        from .pack_authoring import propose_verdict
        pos = [l for l in positive_examples.split("\n") if l.strip()]
        neg = [l for l in negative_examples.split("\n") if l.strip()]
        out = propose_verdict(
            pack_name=pack, verdict=verdict, note=note,
            positives=pos, negatives=neg,
            remedy_kind=remedy_kind, remedy_owner=remedy_owner or pack,
            verify=verify, remedy_note=remedy_note, author=author)
        # Only a proposal that passes every check reaches the queue. A
        # refused one is returned with its counter-example so the author can
        # fix it, and is NOT queued -- a review queue that fills with things
        # a reviewer must reject is a queue a reviewer stops reading.
        if out.get("ok"):
            entry = pack_quarantine.propose(
                pack_name=pack, verdict=verdict, proposed=out["proposed_pack"],
                positives=pos, negatives=neg, author=author,
                report={"pattern": out["pattern"],
                        "pattern_was_generated": out["pattern_was_generated"],
                        "shadowed_fixtures": out["shadowed_fixtures"]})
            out["queued"] = entry is not None
            if entry is not None:
                pack_quarantine.save(pkpath)
            else:
                out["queued_note"] = "an identical proposal is already pending"
            out.pop("proposed_pack", None)  # the queue holds it; keep the reply small
        return json.dumps(out, ensure_ascii=False)

    @mcp.tool()
    def list_pending_pack_verdicts() -> str:
        """Proposed failure-pack verdicts awaiting review. Each carries the
        real log examples the author pasted, the generated pattern, and any
        existing fixtures it would also claim -- the evidence a reviewer
        needs. Use the index with accept/reject_pack_verdict."""
        pend = pack_quarantine.pending()
        return json.dumps([
            {"index": i, "pack_name": e.pack_name, "verdict": e.verdict,
             "author": e.author, "positives": e.positives,
             "negatives": e.negatives, "report": e.report, "ts": e.ts}
            for i, e in enumerate(pend)
        ], ensure_ascii=False)

    @mcp.tool()
    def accept_pack_verdict(index: int) -> str:
        """Write one pending verdict into the overlay pack directory and
        reload the registry. The ONLY path from a proposal to a live
        classifier. Re-validates first: a proposal can go stale if the pack
        changed since, and writing one the loader would reject leaves the
        expert with a file that silently does nothing."""
        out = pack_quarantine.accept(index, pack_overlay_dir)
        pack_quarantine.save(pkpath)
        return json.dumps(out, ensure_ascii=False)

    @mcp.tool()
    def reject_pack_verdict(index: int) -> str:
        """Discard one pending verdict proposal."""
        out = pack_quarantine.reject(index)
        pack_quarantine.save(pkpath)
        return json.dumps(out, ensure_ascii=False)

    @mcp.tool()
    def test_failure_pack(pack: str, log_samples: str) -> str:
        """Run a pack over real log samples (one per line) and report what it
        typed, with example lines per verdict. The number that matters is
        `coverage`: a taxonomy that types 3 of 200 real failures is not yet
        a taxonomy of that field, however tidy it reads. Use before asking
        for a seeded pack to be trusted."""
        from .pack_authoring import test_pack_against_logs
        logs = [l for l in log_samples.split("\n") if l.strip()]
        return json.dumps(test_pack_against_logs(pack, logs), ensure_ascii=False)

    @mcp.tool()
    def export_failure_pack(pack: str) -> str:
        """The pack as editable JSON — the exact on-disk form. Edit it and
        save to the overlay directory (VERA_FAILURE_PACKS_DIR) to override
        the shipped version; invalid edits are refused at load and reported
        by list_failure_domains rather than silently dropped."""
        from .failure_domains import get, pack_to_dict
        dom = get(pack)
        if dom is None:
            return json.dumps({"error": f"unknown pack: {pack}"}, ensure_ascii=False)
        return json.dumps(pack_to_dict(dom), ensure_ascii=False, indent=1)

    @mcp.tool()
    def record_typed_failure(domain: str, source: str, evidence: str) -> str:
        """Classify failure evidence through the named domain pack (see
        list_failure_domains) and record the typed verdict in the shared
        growth-signal store, bucketed by domain:source. The generic sibling
        of record_build_failure — use that for build logs, this for any
        other registered field (game_qa, search_zero, data_pipeline,
        support_kb, soc_telemetry, decision_explain, math). Returns the
        verdict, its remedy, and the pack's maturity — a "seeded" verdict
        is a taxonomy match, not yet an incident-validated diagnosis."""
        from .failure_domains import get, record_typed_failure as _record
        out = _record(growth, domain, source, evidence)
        if "error" not in out:
            _save_growth()
            dom = get(domain)
            spec = dom.remedies.get(out["verdict"]) if dom else None
            if spec is not None:
                out["remedy_kind"] = spec.kind
                out["remedy_owner"] = spec.owner
                out["verify"] = spec.verify
        return json.dumps(out, ensure_ascii=False)

    # ── Settings help ─────────────────────────────────────────────────────
    # These back the in-app support bot. Deliberately deterministic: the bot
    # they replace was a language model holding a one-row lookup table, told
    # in its own prompt not to invent commands — an instruction a model cannot
    # keep when the table does not cover the question. Here "I do not have
    # that" is a return value rather than a hope.

    @mcp.tool()
    def settings_lookup(question: str) -> str:
        """Find which Verantyx setting a question is about.

        Returns one of ANSWER (with the exact Settings tab and field),
        UNKNOWN_NO_SETTING (no such setting — do not guess one),
        UNKNOWN_AMBIGUOUS (several match; ask the user which). An ANSWER for
        a GUI-only setting also carries cli_verdict=UNKNOWN_NO_CLI, which is
        the honest form of "there is no command for this"."""
        from .settings_registry import lookup
        return json.dumps(lookup(question), ensure_ascii=False)

    @mcp.tool()
    def settings_search(question: str, limit: int = 8) -> str:
        """Ranked near-matches for a settings question — use when
        settings_lookup returns UNKNOWN_AMBIGUOUS or UNKNOWN_NO_SETTING and
        you want to offer the user candidates instead of a dead end."""
        from .settings_registry import search
        return json.dumps(search(question, limit=limit), ensure_ascii=False)

    @mcp.tool()
    def list_modes() -> str:
        """Every mode family in the IDE, in one list.

        The interface shows six independent families on five screens, so each
        one looks like the only mode switch. Each option carries `when`: the
        situation it is the right choice for."""
        from .settings_registry import all_modes
        return json.dumps(all_modes(), ensure_ascii=False)

    @mcp.tool()
    def settings_guide(quickstart: bool = False) -> str:
        """The settings guide as Markdown, generated from the same registry
        the lookup tools read, so the document and the answers cannot drift
        apart. `quickstart=True` returns the short first-run path."""
        from .settings_guide import render_guide, render_quickstart
        return render_quickstart() if quickstart else render_guide()

    @mcp.tool()
    def load_documents(paths: str, ingest: bool = True) -> str:
        """Read files (or a folder) into the store — PDF, Word, HTML, CSV,
        JSON, plain text.

        `paths` is comma-separated; a directory is walked. Files that cannot
        be read are reported by name with the reason (UNKNOWN_NO_PARSER for a
        format with no loader, UNKNOWN_EMPTY_DOCUMENT for a scanned PDF that
        holds only images) and the rest still load — a batch that drops what
        it could not read reports a smaller corpus as if it were the whole
        one. Japanese and English are both segmented correctly."""
        from .document_ingest import ingest_documents
        from .document_loaders import load_directory, load_paths

        items = [x.strip() for x in (paths or "").split(",") if x.strip()]
        docs, skipped = [], []
        for item in items:
            res = (load_directory(item) if Path(item).is_dir()
                   else load_paths([item]))
            docs.extend(res["documents"])
            skipped.extend(res["skipped"])
        out = {"loaded": len(docs), "skipped": skipped,
               "sources": [d.source for d in docs]}
        if ingest and docs:
            rep = ingest_documents(store, docs)
            _save()
            out["ingested"] = rep.as_dict()
        return json.dumps(out, ensure_ascii=False)

    @mcp.tool()
    def field_report_categories(lang: str = "ja") -> str:
        """The posting categories and their closed status sets, with how many
        minutes each takes to go stale.

        Statuses are CHOSEN, never typed: measured on ten realistic resident
        postings, free-text status yielded a usable state zero times, because
        residents write 「ここ給水やってる」 and the engine reads formal
        announcements. Choosing removes the parse instead of climbing it."""
        from .field_reports import category_list
        return json.dumps(category_list(lang), ensure_ascii=False)

    @mcp.tool()
    def field_report_needs(lang: str = "ja") -> str:
        """What someone might need, and which categories answer it.

        Keyed on NEED, never on who the person is. Asking whether someone is
        elderly or disabled classifies people in order to help them, and that
        fails three ways: some will not answer, some do not recognise
        themselves in the label, and holding the answer creates a duty to
        protect it."""
        from .field_reports import need_list
        return json.dumps(need_list(lang), ensure_ascii=False)

    @mcp.tool()
    def assess_field_reports(reports_json: str, now: int,
                             needs: str = "") -> str:
        """Turn structured field reports into typed findings.

        `reports_json` is a list of {place, category, status, at, reporter,
        note, official}; `now` and `at` are minutes on one shared clock.
        Pass `needs` (comma-separated) to select by what someone needs.

        Verdicts: CONFIRMED (several reporters, recently), REPORTED (one),
        CONFLICT (fresh reports disagree — no side preferred, none outvoted,
        including the official one), SUPERSEDED, EXPIRED (nothing recent
        enough to stand behind — NOT the same as closed), UNKNOWN_NO_REPORT.

        No confidence score is returned, deliberately: a number performs a
        precision nobody has and a reader takes it as permission."""
        from .field_reports import Report, assess, for_needs, validate
        try:
            raw = json.loads(reports_json)
        except json.JSONDecodeError as exc:
            return json.dumps({"verdict": "UNKNOWN_BAD_INPUT",
                               "reason": str(exc)}, ensure_ascii=False)
        reports, errors = [], []
        for i, d in enumerate(raw if isinstance(raw, list) else []):
            r = Report(place=str(d.get("place", "")),
                       category=str(d.get("category", "")),
                       status=str(d.get("status", "")),
                       at=int(d.get("at", 0)),
                       reporter=str(d.get("reporter", "resident")),
                       note=str(d.get("note", "")),
                       official=bool(d.get("official", False)))
            errs = validate(r)
            if errs:
                errors.append({"index": i, "errors": errs})
            else:
                reports.append(r)
        want = [n.strip() for n in (needs or "").split(",") if n.strip()]
        if want:
            out = {"findings": for_needs(reports, int(now), want)}
        else:
            places = sorted({(r.place, r.category) for r in reports})
            out = {"findings": [assess(reports, int(now), p, c).as_dict()
                                for p, c in places]}
        # Rejected postings are named rather than dropped: a board that
        # silently ignores malformed input reports a smaller world as if it
        # were the whole one.
        if errors:
            out["rejected"] = errors
        return json.dumps(out, ensure_ascii=False)

    @mcp.tool()
    def explain_placement(sentence: str) -> str:
        """Where would this sentence land, and why — core, facets, poles,
        the subject gate's verdict, and the arm, each with its reason.

        The adjustment interface for "arrange my data": placement is a pure
        function of text plus grammar data, so the way to change it is to
        see the decision, extend the grammar overlay, and explain again.
        There is deliberately no hand-reordering — hand-placed facts cannot
        be re-derived and would break reproducibility."""
        from .placement_explain import explain
        return json.dumps(explain(sentence), ensure_ascii=False)

    @mcp.tool()
    def grammar_status() -> str:
        """The Japanese grammar currently in force: bundled counts, which
        overlay (if any) loaded from beside the store, and any overlay error
        — an invalid overlay refuses to load with every problem named."""
        out = _jag.status()
        if _grammar_overlay_error:
            out["overlay_error"] = _grammar_overlay_error
        return json.dumps(out, ensure_ascii=False)

    @mcp.tool()
    def own_ai_guide() -> str:
        """The build-your-own-AI guide: how placement works (core / facet /
        pole / arm), the explain→overlay→re-explain adjustment loop, the
        overlay file format with validation rules, and the traps the system
        guards against. Bilingual, rendered from code so it cannot drift."""
        from .own_ai_guide import render
        return render()

    @mcp.tool()
    def goal_recipe(question: str) -> str:
        """Turn "what I want to do" into the ordered settings that get there.

        Use this BEFORE settings_lookup when the user describes an outcome
        ("build my own AI", "keep everything offline", "run across two Macs")
        rather than naming a setting — a newcomer does not know the setting
        is called inference_mode, so they cannot ask for it by name.

        Returns ANSWER with numbered steps, each carrying the settings tab to
        open, the value to set, why it matters, and whether the app may set it
        for the user. Otherwise UNKNOWN_NO_RECIPE (with the list of goals) or
        UNKNOWN_AMBIGUOUS_GOAL."""
        from .task_recipes import match_goal
        return json.dumps(match_goal(question), ensure_ascii=False)

    @mcp.tool()
    def list_goals() -> str:
        """Every task recipe available — the "what can I actually do with
        this" list, for a first-run screen or when goal_recipe misses."""
        from .task_recipes import list_goals as _list
        return json.dumps(_list(), ensure_ascii=False)

    @mcp.tool()
    def failure_stats() -> str:
        """Histogram of typed failures across all recorded UNKNOWN buckets,
        plus what the boundary detector currently makes of each bucket.
        This is the 'which kind of failure dominates' view: verdict counts
        say what keeps going wrong, classifications say what the system
        thinks should be done about it (needs_more_facts /
        needs_more_capacity / growth_candidate / reject_open_domain)."""
        from . import boundary as _boundary
        verdict_hist: dict = {}
        class_hist: dict = {}
        rows = []
        for bucket in growth.buckets.values():
            for v, n in bucket.verdict_counts.items():
                verdict_hist[v] = verdict_hist.get(v, 0) + n
            cls = _boundary.classify(bucket).classification
            class_hist[cls] = class_hist.get(cls, 0) + 1
            rows.append({"normalized": bucket.normalized,
                         "total": bucket.total(),
                         "dominant": bucket.dominant_verdict(),
                         "classification": cls})
        rows.sort(key=lambda r: -r["total"])
        return json.dumps(
            {"verdicts": verdict_hist, "classifications": class_hist,
             "buckets": rows[:50]},
            ensure_ascii=False,
        )

    @mcp.tool()
    def list_pending_capacity_limits() -> str:
        """List calibrated limit increases awaiting human review. Each entry
        carries the probe evidence: which failing queries were re-run, at
        which multipliers, and that every one answered. Use the index with
        accept_capacity_limit / reject_capacity_limit."""
        pend = capacity_quarantine.pending()
        return json.dumps(
            [{"index": i, **e.as_dict()} for i, e in enumerate(pend)],
            ensure_ascii=False,
        )

    @mcp.tool()
    def accept_capacity_limit(index: int) -> str:
        """Apply one pending limit increase (by index) to VeraConfig. The
        ONLY path by which a proposed limit becomes a running limit — the
        math domain reads config per query, so it takes effect immediately,
        no restart."""
        out = capacity_quarantine.accept(index)
        capacity_quarantine.save(cqpath)
        return json.dumps(out, ensure_ascii=False)

    @mcp.tool()
    def reject_capacity_limit(index: int) -> str:
        """Discard one pending limit increase (by index)."""
        out = capacity_quarantine.reject(index)
        capacity_quarantine.save(cqpath)
        return json.dumps(out, ensure_ascii=False)

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

    @mcp.tool()
    def find_similar_gaps(gap_id: str, limit: int = 5) -> str:
        """Structural (not text/vector) similarity against other GapNodes'
        role/failure_type/input_type/output_type/expected_transition/
        observed_transition fields — same shape, possibly totally
        different subject (e.g. a save-button bug and a 3D-export bug
        sharing "trigger fires, expected transition never observed").
        Requires the target and candidates to have these fields set (most
        gaps created by router.py/agent.py don't set them yet — this tool
        is useful once callers start populating them). NOT_COMPARABLE
        candidates are dropped entirely, not returned with a low score."""
        target = gap_graph.get(gap_id)
        if target is None:
            return json.dumps({"ok": False, "error": "unknown_gap_id"})
        matches = find_structural_matches(target, list(gap_graph.nodes.values()), limit=limit)
        return json.dumps({
            "ok": True,
            "matches": [
                {"gap_id": m.gap_id, "level": m.level, "scores": m.scores,
                 "matched_dimensions": m.matched_dimensions, "different_dimensions": m.different_dimensions,
                 "resolution": gap_graph.get(m.gap_id).resolution if gap_graph.get(m.gap_id) else None}
                for m in matches
            ],
        }, ensure_ascii=False)

    @mcp.tool()
    def log_transfer_attempt(source_gap_id: str, target_gap_id: str, resolution_applied: str) -> str:
        """Milestone R3: record that a structurally-matched resolution
        (from find_similar_gaps' source_gap_id) was actually tried on
        target_gap_id. `success` starts unset — call
        record_transfer_outcome once you know the result, or let
        heartbeat's automatic status-based inference fill it in later.
        This tool ONLY records; it never applies anything itself."""
        target = gap_graph.get(target_gap_id)
        source = gap_graph.get(source_gap_id)
        if target is None or source is None:
            return json.dumps({"ok": False, "error": "unknown_gap_id"})
        matches = find_structural_matches(target, [source], limit=1)
        if not matches:
            return json.dumps({"ok": False, "error": "not_structurally_comparable"})
        record = record_transfer_attempt(
            transfer_log, matches[0], target_gap_id=target_gap_id, resolution_applied=resolution_applied,
        )
        _save_transfer_log()
        return json.dumps({"ok": True, "outcome_id": record.outcome_id}, ensure_ascii=False)

    @mcp.tool()
    def record_transfer_result(outcome_id: str, success: bool, judged_by: str = "human") -> str:
        """Explicitly judge a previously-logged transfer attempt
        (log_transfer_attempt's outcome_id). Explicit judgment always
        takes priority over heartbeat's later automatic inference."""
        record = record_transfer_outcome(transfer_log, outcome_id, success=success, judged_by=judged_by)
        if record is None:
            return json.dumps({"ok": False, "error": "unknown_outcome_id"})
        _save_transfer_log()
        return json.dumps({"ok": True, **record.as_dict()}, ensure_ascii=False)

    @mcp.tool()
    def list_transfer_outcomes(only_unjudged: bool = False) -> str:
        """The raw log this milestone exists to build up — every recorded
        structural-match reuse attempt and whether it worked. Nothing
        analyzes this yet (deliberately not part of this pass); it's the
        evidence a future calibration step would need to tell "just needs
        more knowledge" apart from "the comparison shape itself is
        insufficient" (see this module's own docstring)."""
        records = transfer_log.unjudged() if only_unjudged else transfer_log.records
        return json.dumps([r.as_dict() for r in records], ensure_ascii=False)

    def _csv(s: str) -> list:
        return [x.strip() for x in s.split(",") if x.strip()] if s else []

    @mcp.tool()
    def bootstrap_unknown_task(
        name: str, description: str = "", user_goal: str = "",
        available_inputs: str = "", available_tools: str = "", known_affordances: str = "",
        success_criteria: str = "", allowed_sources: str = "", constraints: str = "",
        cognition_mode: str = "normal",
    ) -> str:
        """Milestone R2: turn an unfamiliar task (ARC-AGI-3, an unknown CLI/
        library, an unknown repository, or anything else — same entry point
        for all of them) into 6 structural slots (IDENTITY/GOAL/AFFORDANCES/
        INPUTS/SUCCESS_CRITERIA/CONSTRAINTS). Unknown slots become typed
        GapNodes; the result also surfaces past tasks with the same known/
        unknown SHAPE (structural_matches) and one recommended next
        acquisition action. List-shaped args are comma-separated (e.g.
        allowed_sources="web,local_repository"). This tool only structures
        the task — it never searches or acts on its own. Respects the same
        normal/experiment/sleep contract as record_ui_transition (Milestone
        S): "normal" is a guaranteed no-op, no GapNodes written."""
        if cognition_mode == "normal":
            return json.dumps({"ok": True, "skipped": "normal_mode"})
        descriptor = TaskDescriptor(
            name=name, description=description, user_goal=user_goal,
            available_inputs=_csv(available_inputs), available_tools=_csv(available_tools),
            known_affordances=_csv(known_affordances), success_criteria=_csv(success_criteria),
            allowed_sources=_csv(allowed_sources), constraints=_csv(constraints),
        )
        result = _bootstrap_unknown_task(gap_graph, descriptor)
        _save_gap_graph()
        action = _select_next_action(result, gap_graph)
        return json.dumps({
            "task_id": result.task_id, "known_slots": result.known_nodes,
            "gap_ids": result.gap_nodes, "executable": result.executable,
            "structural_matches": [
                {"gap_id": m.gap_id, "level": m.level} for m in result.structural_matches
            ],
            "next_action": {
                "intent_type": action.intent_type, "target_gap_id": action.target_gap_id,
                "preferred_tools": action.preferred_tools, "allowed_sources": action.allowed_sources,
                "query": action.required_evidence[0] if action.required_evidence else None,
            } if action else None,
        }, ensure_ascii=False)

    @mcp.tool()
    def arc_observe_transition(session_id: str, level_id: str, action: str,
                                frame_before: str, frame_after: str) -> str:
        """Milestone R1: feed one action->observation step from a 2D
        dynamic-puzzle environment (ARC-AGI-3 or similar). frame_before/
        frame_after are JSON 2D grids of ints (e.g. "[[0,0],[1,0]]"). If
        the naive "nothing changes" hypothesis was wrong, creates a typed
        GapNode and immediately checks it against every past environment
        gap for a structural match (same failure shape, possibly a
        totally different game)."""
        try:
            before = json.loads(frame_before)
            after = json.loads(frame_after)
        except json.JSONDecodeError as e:
            return json.dumps({"ok": False, "error": f"bad_json: {e}"})
        node = observe_transition(
            gap_graph, session_id=session_id, level_id=level_id, action=action,
            frame_before=before, frame_after=after,
        )
        if node is None:
            return json.dumps({"ok": True, "gap_created": False})
        _save_gap_graph()
        matches = find_transferable_matches(node, gap_graph)
        return json.dumps({
            "ok": True, "gap_created": True, "gap_id": node.gap_id,
            "transferable_matches": [{"gap_id": m.gap_id, "level": m.level} for m in matches],
        }, ensure_ascii=False)

    @mcp.tool()
    def record_ui_transition(session_id: str, action_label: str, changed: bool,
                              cognition_mode: str = "normal") -> str:
        """Milestone S: the IDE's Swift side calls this once per recorded
        UI-automation step (the same call site that already writes to
        UITestVectorTrace) to record a causal action -> observation pair
        into GapGraph -- model-independent, unlike UITestVectorTrace's own
        JGEN-embedding record. Respects the same normal/experiment/sleep
        contract as every other GapNode-creating path: "normal" is a
        guaranteed no-op, matching router.py's own guarantee. v1 only
        records what was observed (see ui_transition.py's own docstring
        for why expected_transition is deliberately left unset) --
        mismatch detection across accumulated observations is future
        work, not this tool's job."""
        if cognition_mode == "normal":
            return json.dumps({"ok": True, "skipped": "normal_mode"})
        node = observe_ui_transition(
            gap_graph, session_id=session_id, action_label=action_label, changed=changed,
        )
        _save_gap_graph()
        return json.dumps({"ok": True, "gap_id": node.gap_id, "status": node.status}, ensure_ascii=False)

    @mcp.tool()
    def list_pending_tool_calls() -> str:
        """Milestone R4: mutating tool calls the Vera-harness chat (IDE)
        proposed but could not run without a human — same shape as
        list_pending_ai_facts/list_pending_domain_modules. Use the index
        with accept_tool_call/reject_tool_call.

        Real bug found live: the Vera-harness chat runs as a SEPARATE OS
        process (`vera-memory ... serve`, launched by VeraAgentClient.swift)
        from this MCP server (`vera-memory ... mcp`, launched by
        MCPEngine.swift) -- two independent processes with their own
        memory, sharing state only through tool_call_quarantine.json on
        disk. Reading the in-memory `tool_call_quarantine` this function
        used to use was only ever a snapshot from whenever THIS process
        started, so a call queued by the other process was invisible
        forever. Reloading from disk on every call is the fix -- cheap
        (small JSON file, human-paced call frequency), and correct for
        two independent writers in a way an in-memory cache never can be."""
        nonlocal tool_call_quarantine
        tool_call_quarantine = ToolCallQuarantine.load(tcqpath)
        pend = tool_call_quarantine.pending()
        return json.dumps([{"index": i, **e.as_dict()} for i, e in enumerate(pend)], ensure_ascii=False)

    @mcp.tool()
    def accept_tool_call(index: int) -> str:
        """Actually RUN one pending tool call (by index from
        list_pending_tool_calls) now, with current state — not a replay
        of state from when it was proposed. The ONLY path from a queued
        proposal to an executed mutating action."""
        nonlocal tool_call_quarantine
        tool_call_quarantine = ToolCallQuarantine.load(tcqpath)
        pend = tool_call_quarantine.pending()
        if not (0 <= index < len(pend)):
            return json.dumps({"ok": False, "error": "index_out_of_range"})
        result = tool_call_quarantine.accept(pend[index], tool_registry)
        _save_tool_call_quarantine()
        return json.dumps({"executed": True, "result": result}, ensure_ascii=False)

    @mcp.tool()
    def reject_tool_call(index: int) -> str:
        """Discard one pending tool call (by index) — never runs."""
        nonlocal tool_call_quarantine
        tool_call_quarantine = ToolCallQuarantine.load(tcqpath)
        pend = tool_call_quarantine.pending()
        if not (0 <= index < len(pend)):
            return json.dumps({"ok": False, "error": "index_out_of_range"})
        ok = tool_call_quarantine.reject(pend[index])
        _save_tool_call_quarantine()
        return json.dumps({"ok": ok}, ensure_ascii=False)

    mcp.run()
    return 0
