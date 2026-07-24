"""Local LLM client (Ollama) — the language surface Vera controls.

Vera never depends on this: everything deterministic works without it.
When present, a local model supplies fluent phrasing while Vera supplies
facts, memory, math, and the decision of *when the LLM is allowed to speak*.

Only stdlib (urllib) — no extra dependencies. Default endpoint is the
standard Ollama daemon at http://localhost:11434.
"""
from __future__ import annotations

import json
import urllib.error
import urllib.request
from typing import Any, Dict, Optional

OLLAMA_URL = "http://localhost:11434"


def ollama_available(url: str = OLLAMA_URL, timeout: float = 1.5) -> bool:
    try:
        with urllib.request.urlopen(f"{url}/api/tags", timeout=timeout):
            return True
    except (urllib.error.URLError, OSError):
        return False


def ollama_generate(
    model: str,
    prompt: str,
    *,
    system: Optional[str] = None,
    url: str = OLLAMA_URL,
    timeout: float = 180.0,
    temperature: float = 0.2,
    num_predict: int = 256,
) -> Dict[str, Any]:
    payload: Dict[str, Any] = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": temperature, "num_predict": num_predict},
    }
    if system:
        payload["system"] = system
    req = urllib.request.Request(
        f"{url}/api/generate",
        data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            d = json.loads(resp.read())
        return {"ok": True, "text": d.get("response", "").strip(), "model": model}
    except (urllib.error.URLError, OSError, json.JSONDecodeError) as e:
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "model": model}
