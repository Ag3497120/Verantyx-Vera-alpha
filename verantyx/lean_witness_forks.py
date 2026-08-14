"""Forks for the Lean witness type — the sorry catch above all."""
from __future__ import annotations

from typing import Any, Dict, List


def lean_witness_types_fork() -> Dict[str, Any]:
    """The three absences stay apart, and `sorry` never verifies.

    VERIFIED must carry a toolchain-versioned witness facet (the
    citation), a failing proof must land in UNPROVEN without any claim
    of falsity, and a `sorry` proof — which the kernel accepts with
    exit 0 and a warning — must be caught by name. The smoke test that
    built this module caught exactly that: the warning's quoting
    changed between Lean releases and `sorry` verified quietly.

    Skipped (reported, never counted as a pass) on machines without a
    lean executable.
    """
    from .lean_witness import lean_binary, verify

    name = "LEAN_WITNESS_TYPES"
    if lean_binary() is None:
        return {"experiment": "lean_witness", "fork": name,
                "skipped": "no lean toolchain on this machine"}

    ok_true = verify("theorem t : 1 + 1 = 2 := rfl")
    ok_false = verify("theorem t : 1 + 1 = 3 := rfl")
    ok_sorry = verify("theorem t : 1 + 1 = 2 := sorry")

    ok = (ok_true["verdict"] == "VERIFIED"
          and str(ok_true.get("witness", "")).startswith("verified:lean4:")
          and ok_false["verdict"] == "UNPROVEN"
          and "false" not in str(ok_false.get("note", "")).split("not a claim")[0]
          and ok_sorry["verdict"] == "UNPROVEN_SORRY")
    return {"experiment": "lean_witness", "fork": name, "pass": bool(ok),
            "result": {"true": ok_true["verdict"],
                       "witness": ok_true.get("witness"),
                       "false": ok_false["verdict"],
                       "sorry": ok_sorry["verdict"]}}


def all_lean_witness_forks() -> List[Dict[str, Any]]:
    return [lean_witness_types_fork()]
