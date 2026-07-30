"""Milestone N — Vera-as-harness HTTP+SSE daemon.

Additive to (not a replacement for) mcp_server.py: MCP stays exactly as-is
for other clients (Claude Desktop etc). This is a second, IDE-facing
transport that inverts the calling direction — instead of the IDE calling
Vera as an MCP tool, Vera runs Agent.run()'s ReAct loop as the primary
controller and pushes live progress to the IDE over Server-Sent Events, so
the IDE no longer has to poll.

Deliberately stdlib-only (http.server, threading, queue, uuid) — same
"zero required third-party dependencies" principle Milestone H's
vera-memory freeze already established. No aiohttp/websockets.

Local-only, no auth: v1 assumes 127.0.0.1 and a trusted local caller, same
as Ollama's own default. Do not bind 0.0.0.0 without adding auth first.
"""
from __future__ import annotations

import json
import queue
import threading
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Callable, Dict, Optional
from urllib.parse import urlparse, parse_qs

from .agent import Agent
from .cross_store import CrossStore


class RunState:
    def __init__(self) -> None:
        self.events: "queue.Queue[Dict[str, Any]]" = queue.Queue()
        self.done = threading.Event()
        self.result: Optional[Dict[str, Any]] = None


def _make_llm_fn(model: str, backend: str, jgen_endpoint: Optional[str]) -> Optional[Callable]:
    if backend == "jgen":
        if not jgen_endpoint:
            return None
        return _jgen_llm_fn(jgen_endpoint)
    from .llm_local import ollama_generate

    def llm_fn(prompt: str, system: Optional[str]) -> Dict[str, Any]:
        return ollama_generate(model, prompt, system=system, timeout=180)

    return llm_fn


def _jgen_llm_fn(endpoint: str) -> Callable:
    """Points Agent's llm callback at the IDE's JGenAgentServer (N4) instead
    of Ollama — this is the concrete "IDE as tool provider" inversion:
    Vera is still the caller/controller, JGEN is just a subordinate tool
    reachable over loopback HTTP, same shape as Ollama's own API."""
    import urllib.error
    import urllib.request

    def llm_fn(prompt: str, system: Optional[str]) -> Dict[str, Any]:
        payload = json.dumps({"prompt": prompt, "system": system or ""}).encode()
        req = urllib.request.Request(
            endpoint.rstrip("/") + "/jgen/generate",
            data=payload, headers={"Content-Type": "application/json"},
        )
        try:
            with urllib.request.urlopen(req, timeout=180) as resp:
                d = json.loads(resp.read())
            return {"ok": True, "text": d.get("text", "")}
        except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
            return {"ok": False, "error": f"jgen_endpoint_error: {e}"}

    return llm_fn


def make_handler(store: CrossStore, save: Callable[[], None], default_model: str,
                  jgen_endpoint: Optional[str]):
    runs: Dict[str, RunState] = {}
    runs_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, fmt: str, *args: Any) -> None:  # quiet by default
            pass

        def _json(self, code: int, body: Dict[str, Any]) -> None:
            data = json.dumps(body, ensure_ascii=False).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self) -> None:  # noqa: N802 — BaseHTTPRequestHandler's own naming convention
            parsed = urlparse(self.path)
            if parsed.path != "/agent/run":
                self._json(404, {"ok": False, "error": "not_found"})
                return
            length = int(self.headers.get("Content-Length", "0") or "0")
            try:
                body = json.loads(self.rfile.read(length) or b"{}")
            except json.JSONDecodeError:
                self._json(400, {"ok": False, "error": "bad_json"})
                return
            task = body.get("task", "")
            if not task:
                self._json(400, {"ok": False, "error": "missing_task"})
                return
            model = body.get("model") or default_model
            backend = body.get("backend", "ollama")
            llm_fn = _make_llm_fn(model, backend, jgen_endpoint)
            if llm_fn is None:
                self._json(400, {"ok": False, "error": "jgen_backend_requested_but_no_endpoint_configured"})
                return

            run_id = uuid.uuid4().hex
            state = RunState()
            with runs_lock:
                runs[run_id] = state

            def on_step(event: Dict[str, Any]) -> None:
                state.events.put(event)

            def worker() -> None:
                agent = Agent(store, llm=llm_fn, save=save, auto_approve=False)
                result = agent.run(task, on_step=on_step)
                state.result = result
                state.done.set()
                state.events.put({"__terminal__": True})

            threading.Thread(target=worker, daemon=True).start()
            self._json(202, {"ok": True, "run_id": run_id})

        def do_GET(self) -> None:  # noqa: N802
            parsed = urlparse(self.path)
            qs = parse_qs(parsed.query)

            if parsed.path == "/events":
                run_id = (qs.get("run_id") or [""])[0]
                with runs_lock:
                    state = runs.get(run_id)
                if state is None:
                    self._json(404, {"ok": False, "error": "unknown_run_id"})
                    return
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                while True:
                    try:
                        event = state.events.get(timeout=30)
                    except queue.Empty:
                        # keep-alive comment line — standard SSE idiom
                        try:
                            self.wfile.write(b": ping\n\n")
                            self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError):
                            return
                        continue
                    if event.get("__terminal__"):
                        try:
                            self.wfile.write(b"event: done\ndata: {}\n\n")
                            self.wfile.flush()
                        except (BrokenPipeError, ConnectionResetError):
                            pass
                        return
                    try:
                        line = f"data: {json.dumps(event, ensure_ascii=False)}\n\n".encode()
                        self.wfile.write(line)
                        self.wfile.flush()
                    except (BrokenPipeError, ConnectionResetError):
                        return
                return

            if parsed.path.startswith("/agent/run/"):
                run_id = parsed.path.rsplit("/", 1)[-1]
                with runs_lock:
                    state = runs.get(run_id)
                if state is None:
                    self._json(404, {"ok": False, "error": "unknown_run_id"})
                    return
                if not state.done.is_set():
                    self._json(200, {"ok": True, "status": "running"})
                    return
                self._json(200, {"ok": True, "status": "done", "result": state.result})
                return

            self._json(404, {"ok": False, "error": "not_found"})

    return Handler


def serve(store: CrossStore, save: Callable[[], None], *, port: int = 8765,
          default_model: str = "", jgen_endpoint: Optional[str] = None) -> int:
    handler = make_handler(store, save, default_model, jgen_endpoint)
    httpd = ThreadingHTTPServer(("127.0.0.1", port), handler)
    print(f"[vera serve] listening on http://127.0.0.1:{port} "
          f"(model={default_model or '(unset)'}, jgen_endpoint={jgen_endpoint or '(none)'})")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0
