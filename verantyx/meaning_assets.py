"""Lazy, cached loaders for the meaning-layer sidecars the MCP doors need.

The doors (vera_diff / vera_typo) need the same four assets every call:
predicate profiles (107MB json), the alias map (48MB, 941,604 entries),
the sense inventory (62MB, 122,988 surfaces), and the attested-word
lattice (built from writer.json's vocabulary, long window included).
Loading any of them per call would make the first tool answer the only
tool answer, so each loads once per process and stays.

What deliberately does NOT load here: the shallow shelf (912MB) and the
definition sidecar (250MB). A door that costs the host a gigabyte of
resident json to render two shelf layers is the wrong trade until those
sidecars move to an indexed store; `diff` is called with an empty shelf
and its coverage field says honestly that layers ④/⑥ abstained. The
measurement scripts, which run once and exit, keep loading the real
shelf — the 11/30 bank number is theirs, not the door's.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

BUILD = Path.home() / "Projects" / "vera-corpus" / "build"

_cache: Dict[str, Any] = {}


def profiles() -> Dict[str, Any]:
    if "profiles" not in _cache:
        from .predicate_profile import load
        _cache["extractor"], _cache["profiles"] = load()
    return _cache["profiles"]


def extractor() -> str:
    profiles()
    return _cache["extractor"]


def aliases() -> Dict[str, str]:
    if "aliases" not in _cache:
        _cache["aliases"] = json.loads(
            (BUILD / "jawiki_aliases.json").read_text(encoding="utf-8"))
    return _cache["aliases"]


def senses() -> Dict[str, Any]:
    if "senses" not in _cache:
        d = json.loads(
            (BUILD / "jawiki_senses.json").read_text(encoding="utf-8"))
        _cache["senses"] = d.get("senses", d)
    return _cache["senses"]


def lattice() -> Any:
    if "lattice" not in _cache:
        from .lattice import build
        from .writer import Writer
        vocab = Writer.load(BUILD / "writer.json").vocab
        _cache["vocab"] = set(vocab.attested)
        _cache["lattice"] = build(_cache["vocab"])
    return _cache["lattice"]


def vocab() -> Any:
    lattice()
    return _cache["vocab"]


def empty_shelf() -> Any:
    if "empty_shelf" not in _cache:
        from .cross_store import CrossStore
        _cache["empty_shelf"] = CrossStore()
    return _cache["empty_shelf"]
