"""User configuration — LLM choice, store path, Vera↔LLM allocation dial.

Persisted at ~/.verantyx.json. The allocation dial decides which domains
Vera answers itself and where the LLM is allowed to speak:

  math      "vera" (exact, default)      | "llm" (why would you)
  code      "vera" (default)             | "llm"
  known     "vera" | "llm_guided" (default: LLM rephrases Vera's facts)
  unknown   "refuse" (default) | "llm_free" (labeled unverified chat)

`vera setup` edits all of this with arrow keys.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

CONFIG_PATH = Path.home() / ".verantyx.json"

DEFAULT_ALLOCATION: Dict[str, str] = {
    "math": "vera",
    "code": "vera",
    "known": "llm_guided",
    "unknown": "refuse",
}


@dataclass
class VeraConfig:
    llm_model: str = ""              # empty = no LLM (solo)
    store: str = "vera_store.json"
    allocation: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_ALLOCATION))
    hf_store_repo: str = ""          # e.g. "user/Verantyx-Vera-base-store"

    def save(self, path: Path = CONFIG_PATH) -> None:
        path.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False))

    @classmethod
    def load(cls, path: Path = CONFIG_PATH) -> "VeraConfig":
        if not Path(path).is_file():
            return cls()
        d = json.loads(Path(path).read_text())
        alloc = dict(DEFAULT_ALLOCATION)
        alloc.update(d.get("allocation", {}))
        return cls(
            llm_model=d.get("llm_model", ""),
            store=d.get("store", "vera_store.json"),
            allocation=alloc,
            hf_store_repo=d.get("hf_store_repo", ""),
        )


def list_ollama_models() -> List[str]:
    import json as _json
    import urllib.request

    try:
        with urllib.request.urlopen("http://localhost:11434/api/tags", timeout=2) as r:
            return [m["name"] for m in _json.loads(r.read()).get("models", [])]
    except Exception:
        return []


def run_setup_menu(cfg: Optional[VeraConfig] = None) -> VeraConfig:
    """Interactive setup: local LLM, allocation dial, store, HF repo."""
    from .tui import select

    cfg = cfg or VeraConfig.load()

    models = list_ollama_models()
    opts = ["(none — Vera solo, deterministic only)"] + models + ["(keep current: %s)" % (cfg.llm_model or "none")]
    i = select("Local LLM (Ollama):", opts, default=len(opts) - 1)
    if i is not None and i < len(opts) - 1:
        cfg.llm_model = "" if i == 0 else models[i - 1]

    dial: Dict[str, List[str]] = {
        "math": ["vera", "llm"],
        "code": ["vera", "llm"],
        "known": ["vera", "llm_guided"],
        "unknown": ["refuse", "llm_free"],
    }
    desc = {
        "math": "exact wire arithmetic vs LLM guessing",
        "code": "AST facts vs LLM recall",
        "known": "raw facet answer vs LLM phrasing Vera's facts",
        "unknown": "typed refusal vs labeled-unverified LLM chat",
    }
    for domain, choices in dial.items():
        cur = cfg.allocation.get(domain, choices[0])
        j = select(
            f"Allocation — {domain} ({desc[domain]}):",
            choices,
            default=choices.index(cur) if cur in choices else 0,
        )
        if j is not None:
            cfg.allocation[domain] = choices[j]

    cfg.save()
    print(f"saved → {CONFIG_PATH}")
    return cfg
