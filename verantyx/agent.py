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
from typing import Any, Callable, Dict, List, Optional

from .agent_tools import Tool, build_registry, tools_manifest
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
    ):
        self.store = store
        self.llm = llm
        self.save = save
        self.registry = build_registry(store, save)
        self.approver = approver
        self.allocation = allocation or {}
        self.max_steps = max_steps
        self.auto_approve = auto_approve
        self.always: set = set()
        self.transcript: List[Dict[str, Any]] = []

    def _approve(self, tool: Tool, args: Dict[str, Any]) -> bool:
        if not tool.mutating or self.auto_approve or tool.name in self.always:
            return True
        if self.approver is None:
            return False
        decision = self.approver(tool, args)
        if decision == "always":
            self.always.add(tool.name)
            return True
        return decision == "approve"

    def _run_tool(self, name: str, args: Dict[str, Any]) -> Dict[str, Any]:
        tool = self.registry.get(name)
        if tool is None:
            return {"ok": False, "error": f"unknown_tool: {name}"}
        if not self._approve(tool, args):
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

    def run(self, task: str) -> Dict[str, Any]:
        """Full ReAct loop (requires an LLM for autonomous planning)."""
        # 1. deterministic shortcut
        vera = _vera_answer(self.store, task, "auto", self.allocation)
        if vera.get("verdict") == "ANSWER" and vera.get("route") in ("math", "code"):
            return {"final": vera, "steps": 0, "source": "vera_direct"}
        if self.llm is None:
            return {
                "final": vera,
                "steps": 0,
                "source": "vera_only",
                "note": "no LLM configured; use step_solo for manual tools",
            }

        manifest = tools_manifest(self.registry)
        system = _SYSTEM % manifest
        history: List[str] = [f"Task: {task}",
                              f"Vera's initial read: {json.dumps(vera, ensure_ascii=False)[:400]}"]
        for step in range(self.max_steps):
            prompt = "\n".join(history) + "\nNext action (JSON only):"
            r = self.llm(prompt, system)
            if not r.get("ok"):
                return {"final": {"error": r.get("error")}, "steps": step,
                        "source": "llm_error"}
            action = _parse_action(r["text"])
            if action is None:
                history.append(f"(unparseable action; reply JSON only)")
                continue
            self.transcript.append({"step": step, "action": action})
            if "final" in action:
                return {"final": action["final"], "steps": step + 1,
                        "source": "react", "transcript": self.transcript}
            name = action.get("tool", "")
            args = action.get("args", {}) or {}
            obs = self._run_tool(name, args)
            history.append(f"Action: {name}({json.dumps(args, ensure_ascii=False)})")
            history.append(f"Observation: {json.dumps(obs, ensure_ascii=False)[:600]}")
            self.transcript[-1]["observation"] = obs
        return {"final": {"error": "max_steps_reached"}, "steps": self.max_steps,
                "source": "react", "transcript": self.transcript}
