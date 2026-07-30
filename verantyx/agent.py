"""Agent mode — a ReAct loop where Vera is the controller.

Vera is the brain that decides and remembers; the local LLM (optional) is
the language cortex that proposes the next tool call in a constrained JSON
format. Every mutating tool is gated behind arrow-key approval on the CLI.

Turn protocol (one ReAct step):

  1. Vera checks its deterministic faculties first (math/code/knowledge).
     A confident ANSWER can finish the task with no LLM and no tools.
  2. Otherwise the LLM proposes ONE action as JSON:
       {"thought": "...", "tool": "name", "args": {...}}
     or {"thought": "...", "final": "answer"}
  3. Mutating tools require approval (approve / always / deny).
  4. The observation is appended; loop until `final` or max steps.

Without an LLM, agent mode still runs deterministic tools via explicit
`!tool {json}` lines — the loop and approvals work solo.
"""
from __future__ import annotations

import json
import re
import time
from typing import Any, Callable, Dict, List, Optional

from .agent_tools import Tool, build_registry, jgen_reflect, tools_manifest
from .cognitive_interventions import (
    InterventionRecord,
    build_intervention_plan,
    new_intervention_id,
)
from .cross_store import CrossStore
from .router import _vera_answer

_SYSTEM = """You are the planning cortex of Verantyx Vera, a deterministic
reasoning engine. Vera holds verified facts and exact math; you only choose
the next action. Reply with ONE JSON object and nothing else:
  {"thought": "...", "tool": "<name>", "args": {...}}
or when the task is done:
  {"thought": "...", "final": "<answer for the user>"}
Prefer vera_ask / vera_math / vera_code_query for facts — they never
hallucinate. Use web_search only when Vera lacks the knowledge. Never invent
tool results. Available tools:
%s
"""

_JSON = re.compile(r"\{.*\}", re.S)

# Acquisition methods that only fetch raw material for a LATER step (e.g.
# vera_git_clone gets a repo onto disk, but doesn't itself add any
# knowledge) -- succeeding at one of these must NOT close the gap on its
# own, or a gap like "study this repo" would be marked RESOLVED the
# instant the clone finishes, before anything was actually learned.
_NON_TERMINAL_ACQUISITION_METHODS = frozenset({"vera_git_clone"})


def _parse_action(text: str) -> Optional[Dict[str, Any]]:
    m = _JSON.search(text or "")
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
        return d if isinstance(d, dict) else None
    except json.JSONDecodeError:
        return None


class Agent:
    def __init__(
        self,
        store: CrossStore,
        *,
        llm: Optional[Callable[[str, Optional[str]], Dict[str, Any]]] = None,
        save: Callable[[], None] = lambda: None,
        approver: Optional[Callable[[Tool, Dict[str, Any]], str]] = None,
        allocation: Optional[Dict[str, str]] = None,
        max_steps: int = 12,
        auto_approve: bool = False,
        browser_endpoint: Optional[str] = None,
        gap_graph: Optional[Any] = None,
        cognition_mode: str = "normal",
        intervention_log: Optional[Any] = None,
        tool_call_quarantine: Optional[Any] = None,
    ):
        self.store = store
        self.llm = llm
        self.save = save
        self.registry = build_registry(store, save, browser_endpoint=browser_endpoint)
        self.approver = approver
        self.allocation = allocation or {}
        self.max_steps = max_steps
        self.auto_approve = auto_approve
        self.always: set = set()
        self.transcript: List[Dict[str, Any]] = []
        # Milestone O: same no-op-unless-opted-in contract as router.route()'s
        # own gap_graph/cognition_mode params -- this is the actual entry
        # point the IDE's Vera-harness chat (vera_server.py -> Agent.run())
        # uses, which calls _vera_answer() directly and never goes through
        # router.route() at all, so the hook has to live here too or
        # Milestone O would silently never fire for real IDE usage.
        self.gap_graph = gap_graph
        self.cognition_mode = cognition_mode
        # Milestone P5: browser_endpoint doubles as the JGEN reflect
        # endpoint (JGenAgentServer serves both /browser/fetch and
        # /jgen/inject_multi_layer on the same port -- see Milestone N's
        # own "one IDE-side daemon" decision). intervention_log is
        # optional/no-op like gap_graph: omit it and P5 still runs the
        # probe, it just doesn't persist provenance anywhere.
        self.browser_endpoint = browser_endpoint
        self.intervention_log = intervention_log
        # Milestone R4: same optional/no-op contract as gap_graph/
        # intervention_log -- omit it and behavior is unchanged (mutating
        # tools with no interactive approver still get denied, exactly
        # like before this milestone).
        self.tool_call_quarantine = tool_call_quarantine
        self._current_task: str = ""

    def _maybe_record_gap(self, task: str, vera: Dict[str, Any]) -> Optional[Any]:
        if self.gap_graph is None or self.cognition_mode not in ("experiment", "sleep"):
            return None
        verdict = vera.get("verdict")
        if not (isinstance(verdict, str) and verdict.startswith("UNKNOWN")):
            return None
        from .gap_severity import classify as classify_gap, is_repo_study_intent

        gap_class = classify_gap(task)
        acquisition_methods = ["web_search", "fetch_url", "vera_ask"]
        if is_repo_study_intent(task):
            acquisition_methods = ["vera_git_clone", "vera_code_ingest"] + acquisition_methods
        node = self.gap_graph.create(
            gap_type=gap_class.gap_type, subject=task[:200], scope=f"query:{task[:100]}",
            severity=gap_class.severity,
            status="BLOCKED_POLICY" if gap_class.blocked_policy else "DETECTED",
            acquisition_methods=acquisition_methods,
            allowed_sources=["web", "local_repository"],
        )
        return node

    def _maybe_reflect(self, task: str, gap_node: Optional[Any], vera: Dict[str, Any]) -> Optional[str]:
        """Milestone P5 — Vera decides to probe JGEN's hidden state on its
        own, without an LLM choosing jgen_reflect as a tool call. Only
        fires once per gap (status == "DETECTED", i.e. this is the turn
        the gap was first seen) so a long-lived unresolved gap doesn't get
        re-probed every single turn it comes up. Returns a short text
        summary to seed the ReAct history with, or None if nothing fired
        (no gap, no endpoint configured, not in experiment/sleep mode,
        or the probe itself failed) -- every one of those is a silent
        no-op, matching every other Milestone O/P hook's contract."""
        if (
            gap_node is None or self.browser_endpoint is None
            or self.cognition_mode not in ("experiment", "sleep")
            or gap_node.status != "DETECTED"
        ):
            return None

        core = vera.get("core") or ""
        confirmed_facts = [f for f, _count in self.store.top_facets(core, k=3)] if core else []
        specs = build_intervention_plan(gap_node, confirmed_facts=confirmed_facts)
        observe_layers = sorted({s.layer for s in specs} | {max(s.layer for s in specs) + 2})

        result = jgen_reflect(
            self.browser_endpoint, task,
            [s.as_dict() for s in specs], observe_layers,
        )
        observations = result.get("observations") if isinstance(result, dict) else None
        if not isinstance(observations, dict) or not observations:
            # No JGEN endpoint reachable / no model loaded / request failed
            # -- same "this needs a real model on the user's machine" limit
            # as every other Milestone P verification note. Not an error
            # worth surfacing to the ReAct loop, just nothing to add.
            return None

        record = InterventionRecord(
            intervention_id=new_intervention_id(),
            source_nodes=[gap_node.gap_id],
            model="jgen",
            specs=[s.as_dict() for s in specs],
            observe_layers=observe_layers,
            observations={str(k): str(v) for k, v in observations.items()},
            expected_effect=f"prioritize_missing_evidence:{gap_node.gap_type}",
            ts=time.time(),
        )
        if self.intervention_log is not None:
            self.intervention_log.append(record)

        if self.gap_graph is not None and gap_node.status == "DETECTED":
            self.gap_graph.set_status(
                gap_node.gap_id, "RESOLUTION_PLANNED",
                resolution=f"jgen_probe:{record.intervention_id}",
            )

        obs_text = "; ".join(f"layer {k}: {v}" for k, v in record.observations.items())
        return (
            f"Vera probed JGEN's hidden state about this gap "
            f"({gap_node.gap_type}: {gap_node.subject[:100]}) -- observed: {obs_text[:400]}"
        )

    def _run_tool(self, name: str, args: Dict[str, Any], *, thought: str = "") -> Dict[str, Any]:
        tool = self.registry.get(name)
        if tool is None:
            return {"ok": False, "error": f"unknown_tool: {name}"}
        if tool.mutating and not self.auto_approve and name not in self.always:
            if self.approver is not None:
                decision = self.approver(tool, args)
                if decision == "always":
                    self.always.add(name)
                elif decision != "approve":
                    return {"ok": False, "error": "denied_by_user"}
            elif self.tool_call_quarantine is not None:
                # Milestone R4: no interactive approver reachable (the
                # Vera-harness HTTP path never has one) -- queue the call
                # for human review through the SAME propose/pending/accept
                # gate as Vera's own memory (AiFactQuarantine), instead of
                # unconditionally denying it. The tool does NOT run now.
                entry = self.tool_call_quarantine.propose(
                    name, args, reason=thought, task=self._current_task,
                )
                return {
                    "ok": False, "queued_for_approval": True, "call_id": entry.call_id,
                    "note": (
                        f"'{name}' requires human approval and has been queued "
                        f"(call_id={entry.call_id}). It has NOT run yet. Continue "
                        "with what you already know, or end the turn -- the "
                        "actual result will only exist after a human accepts it."
                    ),
                }
            else:
                return {"ok": False, "error": "denied_by_user"}
        try:
            return tool.fn(**args)
        except TypeError as e:
            return {"ok": False, "error": f"bad_args: {e}"}

    def step_solo(self, action_line: str) -> Dict[str, Any]:
        """Deterministic manual step: '!tool {json-args}' or a plain query."""
        if action_line.startswith("!"):
            body = action_line[1:].strip()
            name, _, rest = body.partition(" ")
            args = json.loads(rest) if rest.strip() else {}
            obs = self._run_tool(name, args)
            self.transcript.append({"tool": name, "args": args, "obs": obs})
            return {"tool": name, "observation": obs}
        vera = _vera_answer(self.store, action_line, "auto", self.allocation)
        return {"vera": vera}

    def run(
        self, task: str,
        on_step: "Optional[Callable[[Dict[str, Any]], None]]" = None,
    ) -> Dict[str, Any]:
        """Full ReAct loop (requires an LLM for autonomous planning).

        ``on_step`` (optional): called after each transcript entry is
        finalized (action + observation, or the terminal vera_direct/
        vera_only/llm_error/final/max_steps event) — lets a caller like
        vera_server.py's SSE endpoint (Milestone N) push live progress
        without needing to poll or re-implement this loop. Purely additive:
        omitting it reproduces the exact prior behavior (cli.py's
        cmd_agent doesn't pass one)."""
        self._current_task = task
        # 1. deterministic shortcut
        vera = _vera_answer(self.store, task, "auto", self.allocation)
        gap_node = self._maybe_record_gap(task, vera)
        if vera.get("verdict") == "ANSWER" and vera.get("route") in ("math", "code"):
            result = {"final": vera, "steps": 0, "source": "vera_direct"}
            if on_step:
                on_step(result)
            return result
        if self.llm is None:
            result = {
                "final": vera,
                "steps": 0,
                "source": "vera_only",
                "note": "no LLM configured; use step_solo for manual tools",
            }
            if on_step:
                on_step(result)
            return result

        manifest = tools_manifest(self.registry)
        system = _SYSTEM % manifest
        history: List[str] = [f"Task: {task}",
                              f"Vera's initial read: {json.dumps(vera, ensure_ascii=False)[:400]}"]
        # Milestone P5: Vera's own automatic hidden-state probe (no LLM
        # tool choice involved) gets folded into the SAME initial history
        # the LLM's first prompt is built from -- this is the actual
        # "closed loop" part: the probe happens before the LLM ever picks
        # an action, and its observation becomes part of what the LLM
        # reasons from on step 0, not a tool result bolted on afterward.
        reflect_note = self._maybe_reflect(task, gap_node, vera)
        if reflect_note:
            history.append(reflect_note)
        for step in range(self.max_steps):
            prompt = "\n".join(history) + "\nNext action (JSON only):"
            r = self.llm(prompt, system)
            if not r.get("ok"):
                result = {"final": {"error": r.get("error")}, "steps": step,
                          "source": "llm_error"}
                if on_step:
                    on_step(result)
                return result
            action = _parse_action(r["text"])
            if action is None:
                history.append(f"(unparseable action; reply JSON only)")
                continue
            self.transcript.append({"step": step, "action": action})
            if "final" in action:
                result = {"final": action["final"], "steps": step + 1,
                          "source": "react", "transcript": self.transcript}
                if on_step:
                    on_step(result)
                return result
            name = action.get("tool", "")
            args = action.get("args", {}) or {}
            obs = self._run_tool(name, args, thought=str(action.get("thought", "")))
            history.append(f"Action: {name}({json.dumps(args, ensure_ascii=False)})")
            history.append(f"Observation: {json.dumps(obs, ensure_ascii=False)[:600]}")
            self.transcript[-1]["observation"] = obs
            # Milestone O follow-up: a tool call succeeding in the SAME
            # turn as a recorded gap is itself a resolution -- without
            # this, the gap stayed DETECTED forever even when e.g.
            # vera_code_ingest had already done the actual work via
            # ordinary ReAct tool use, orphaning the node.
            # Not every tool returns an "ok" key (e.g. vera_code_ingest just
            # returns {"n_files": ..., "n_functions": ...} on success), so
            # success is "didn't fail" (no error key / ok not explicitly
            # False), not "has ok: True".
            tool_succeeded = isinstance(obs, dict) and obs.get("ok", True) is not False and not obs.get("error")
            if (
                gap_node is not None and self.gap_graph is not None
                and tool_succeeded and name in gap_node.acquisition_methods
                and name not in _NON_TERMINAL_ACQUISITION_METHODS
                and gap_node.status not in ("RESOLVED",)
            ):
                self.gap_graph.set_status(
                    gap_node.gap_id, "RESOLVED",
                    resolution=f"tool:{name} succeeded in the same run",
                )
            if on_step:
                on_step({"step": step, "action": action, "observation": obs, "source": "react_step"})
        # Ran out of step budget with real observations already gathered
        # (confirmed against a real run: 11 web_search/fetch_url/run_command
        # steps on a repo-analysis task, budget exhausted before the model
        # ever emitted `{"final": ...}`) -- a bare "max_steps_reached" error
        # throws away everything Vera actually found. One last forced
        # synthesis turn (no more tool calls allowed) salvages a real
        # answer from the transcript instead.
        synthesis_prompt = (
            "\n".join(history)
            + "\nYou are out of steps. Do not call any tool. Reply with "
              "ONLY {\"thought\": \"...\", \"final\": \"<your best answer "
              "for the user from what you found above>\"}."
        )
        r = self.llm(synthesis_prompt, system)
        action = _parse_action(r.get("text", "")) if r.get("ok") else None
        if action and "final" in action:
            result = {"final": action["final"], "steps": self.max_steps,
                      "source": "react_forced_synthesis", "transcript": self.transcript}
        else:
            result = {"final": {"error": "max_steps_reached"}, "steps": self.max_steps,
                      "source": "react", "transcript": self.transcript}
        if on_step:
            on_step(result)
        return result
