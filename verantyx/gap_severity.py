"""Milestone O2 — severity classification for a detected gap.

Deterministic, not LLM-based (same "harden the judgment first" principle
as boundary.py). Deliberately conservative in this first pass: the design
discussion's own conclusion was that blocking too aggressively makes Vera
"answer almost nothing" -- so a totally-unanswered question defaults to
QUALITY, not CRITICAL. CRITICAL is reserved for cases where a specific
required piece of information for a MUTATING action is missing (unknown
target file, unclear execution target) -- situations where proceeding
would be actively unsafe, not just incomplete.

Policy-blocked requests (copyright, asset generation) are NOT a severity
level at all -- they get their own status (BLOCKED_POLICY) via
boundary.py's existing _out_of_scope_subcategory, since "knowledge is
missing" and "this must not be done regardless of knowledge" are
different failure modes (this file's own docstring point, taken directly
from the design discussion).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from . import boundary


@dataclass
class GapClassification:
    severity: str          # CRITICAL | QUALITY  (OPTIONAL unused in this pass)
    gap_type: str
    blocked_policy: bool = False
    policy_subcategory: Optional[str] = None


# Substrings that indicate a mutating tool call is missing a required,
# safety-relevant parameter (as opposed to merely "no answer available").
# Small and deliberately conservative -- false negatives fall through to
# QUALITY (safe default), never silently to a lower bar than CRITICAL was
# meant to catch.
_CRITICAL_MARKERS = (
    "unknown_target", "missing_path", "target file is unclear",
    "execution target unknown", "which file", "which environment",
)


def classify(query: str, *, mutating_tool_context: Optional[str] = None) -> GapClassification:
    """`mutating_tool_context`: pass the tool name/args-so-far when this
    gap arose while trying to prepare a mutating tool call (write_file,
    run_command, edit_file, ...) -- None for a plain unanswered question."""
    sub = boundary._out_of_scope_subcategory(query)
    if sub:
        return GapClassification(
            severity="QUALITY", gap_type="POLICY_TRANSFORMATION_REQUIRED",
            blocked_policy=True, policy_subcategory=sub,
        )

    if mutating_tool_context:
        lowered = mutating_tool_context.lower()
        if any(marker in lowered for marker in _CRITICAL_MARKERS):
            return GapClassification(severity="CRITICAL", gap_type="MISSING_REQUIRED_PARAMETER")

    return GapClassification(severity="QUALITY", gap_type="MISSING_KNOWLEDGE")
