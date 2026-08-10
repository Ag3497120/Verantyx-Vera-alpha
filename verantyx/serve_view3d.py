"""Serve the 3D view next to the engine and stream what a query touches.

A published artifact cannot do this. Its CSP blocks every external host and
neither runtime capability the platform offers (`downloads`, `mcp`) opens
one, so a page on claude.ai has no way to reach a Python process on this
machine. Saying otherwise would be a promise the browser refuses to keep.

Served from here it is a same-origin fetch and works. The page detects
which case it is in and says so rather than pretending — "未接続 — 静的表示"
against "接続 — 推論を可視化中".

What is streamed is what the engine actually consults, not a replay: each
setting of `GradedJudge` is asked in turn and the cores its rungs scored are
emitted as they are scored. A visualisation that animated a plausible path
instead would be the same failure as a generated sentence arriving where a
citation is expected.

    vera-view3d --root ~/Projects/vera-corpus --page <dir>
    open http://localhost:8790/vera3d.html
"""
from __future__ import annotations

import argparse
import json
import queue
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any, Dict, List, Optional
from urllib.parse import parse_qs, urlparse

CLIENTS: "List[queue.Queue]" = []
STATE: Dict[str, Any] = {}


def emit(msg: Dict[str, Any]) -> None:
    dead = []
    for q in CLIENTS:
        try:
            q.put_nowait(msg)
        except Exception:
            dead.append(q)
    for q in dead:
        if q in CLIENTS:
            CLIENTS.remove(q)


def load(root: Path) -> None:
    import pickle

    from .cross_store import CrossStore
    from .graded import GradedJudge

    doms = pickle.loads((root / "build" / "federation.pkl").read_bytes())
    st = CrossStore()
    for d in doms:
        for s in doms[d].values():
            st.source_labels |= getattr(s, "source_labels", set())
            for c, cr in s.crosses.items():
                st.crosses.setdefault(c, {}).update(cr)
                st.core_count[c] = st.core_count.get(c, 0) + 1
    STATE["store"] = st
    STATE["judge"] = GradedJudge().build(st)
    STATE["domains"] = doms


def run_query(query: str) -> Dict[str, Any]:
    """Ask, streaming each setting's reading as it is produced."""
    from .lang import ja_content_runs
    from .resolution import ask as rung_ask

    j, st = STATE["judge"], STATE["store"]
    terms = ja_content_runs(query)
    emit({"type": "ask", "query": query, "terms": terms})
    if not terms:
        emit({"type": "verdict", "verdict": "UNKNOWN_UNPARSED", "item": None})
        return {"verdict": "UNKNOWN_UNPARSED"}

    # The query's own terms, wherever the store holds them.
    seen = [t for t in terms if t in st.crosses]
    if seen:
        emit({"type": "touch", "nodes": seen, "why": "query"})

    readings: Dict[str, Optional[str]] = {}
    for name, _cfg in j.settings:
        r = rung_ask(j.ladders[name], terms)
        item = r["item"] if r["verdict"] == "ANSWER" else None
        readings[name] = item
        emit({"type": "reading", "setting": name,
              "verdict": r["verdict"], "item": item,
              "answered": r.get("answered", 0), "concord": r.get("concord", 0)})
        if item:
            emit({"type": "touch", "nodes": [item], "why": name})
        time.sleep(.16)          # so a reader can watch the settings differ

    out = j.ask(query)
    detail = f"{out.get('agreeing', 0)}/{out.get('of', 0)}"
    emit({"type": "verdict", "verdict": out["verdict"],
          "item": out.get("item"), "detail": detail, "readings": readings})
    return out


class H(BaseHTTPRequestHandler):
    page_dir = Path(".")

    def log_message(self, *a):        # quiet
        pass

    def _send(self, code, ctype, body: bytes):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        u = urlparse(self.path)
        if u.path == "/events":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.end_headers()
            q: "queue.Queue" = queue.Queue(maxsize=400)
            CLIENTS.append(q)
            try:
                while True:
                    try:
                        m = q.get(timeout=15)
                        self.wfile.write(
                            b"data: " + json.dumps(m, ensure_ascii=False).encode()
                            + b"\n\n")
                    except queue.Empty:
                        self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
            except Exception:
                pass
            finally:
                if q in CLIENTS:
                    CLIENTS.remove(q)
            return

        if u.path == "/ask":
            qs = parse_qs(u.query).get("q", [""])[0]
            # `BaseHTTPRequestHandler` hands the request line back decoded as
            # latin-1, so a Japanese query arrives as mojibake and parses to
            # zero terms — UNKNOWN_UNPARSED for a question the store can
            # answer. Re-encode and read it as what it was sent as.
            try:
                qs = qs.encode("latin-1").decode("utf-8")
            except (UnicodeEncodeError, UnicodeDecodeError):
                pass
            threading.Thread(target=run_query, args=(qs,), daemon=True).start()
            self._send(200, "application/json",
                       json.dumps({"verdict": "ACCEPTED", "query": qs},
                                  ensure_ascii=False).encode())
            return

        name = u.path.lstrip("/") or "vera3d.html"
        f = (self.page_dir / name).resolve()
        if not str(f).startswith(str(self.page_dir.resolve())) or not f.is_file():
            self._send(404, "text/plain; charset=utf-8", b"not found")
            return
        ct = ("text/html; charset=utf-8" if f.suffix == ".html"
              else "application/json; charset=utf-8" if f.suffix == ".json"
              else "application/octet-stream")
        self._send(200, ct, f.read_bytes())


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--root", default=str(Path.home() / "Projects" / "vera-corpus"))
    ap.add_argument("--page", required=True, help="directory holding vera3d.html")
    ap.add_argument("--port", type=int, default=8790)
    a = ap.parse_args(argv)

    print("連合を読み込み中…", flush=True)
    load(Path(a.root))
    print(f"核 {len(STATE['store'].crosses):,}  → http://localhost:{a.port}/vera3d.html",
          flush=True)
    print(f"問う:  curl 'http://localhost:{a.port}/ask?q=正当防衛とは'", flush=True)
    H.page_dir = Path(a.page)
    ThreadingHTTPServer(("127.0.0.1", a.port), H).serve_forever()
    return 0


if __name__ == "__main__":
    sys.exit(main())
