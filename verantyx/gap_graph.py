"""Milestone O — GapNode: a persistent, typed "this is missing" state.

The core idea (from design discussion): the fact that an answer's required
structure isn't closed is itself kept as a persistent unresolved state, and
that state is what drives the next action — not curiosity, but mechanically
the same role. This module is the data layer only; classification lives in
gap_severity.py, detection hooks live in router.py, and resolution lives in
mcp_server.py's heartbeat() (Sleep mode) — all gated through the EXISTING
quarantine systems (ai_ingest.AiFactQuarantine, module_ingest.
DomainModuleQuarantine) so nothing this module tracks ever becomes a
trusted fact/module without a human's explicit accept.

Deliberately does NOT replace growth_signals.py/boundary.py (Milestone
M's closed-domain loop) — those keep working unchanged. A GapNode gets
created from a growth_signals bucket once it crosses recurrence threshold
(see router.py's hook), so Milestone M's existing plumbing is the closed-
domain half of this graph, not a separate system.
"""
from __future__ import annotations

import json
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

# Terminal/blocked states a node can land in besides RESOLVED.
BLOCKED_STATUSES = frozenset({
    "BLOCKED_POLICY", "BLOCKED_NO_SOURCE", "BLOCKED_PERMISSION",
    "BLOCKED_BUDGET", "BLOCKED_CONTRADICTION", "STALE",
})
# Forward progress states, in the order a node is expected to move through.
LIFECYCLE_STATUSES = (
    "DETECTED", "SCOPED", "RESOLUTION_PLANNED", "ACQUIRING",
    "EVIDENCE_COLLECTED", "VERIFIED", "RESOLVED",
)
ALL_STATUSES = frozenset(LIFECYCLE_STATUSES) | BLOCKED_STATUSES
SEVERITIES = frozenset({"CRITICAL", "QUALITY", "OPTIONAL"})

# A node in one of these states is a candidate for Sleep-mode resolution
# attempts; anything else (RESOLVED, BLOCKED_*, STALE) is left alone.
ACTIONABLE_STATUSES = frozenset({"DETECTED", "SCOPED", "RESOLUTION_PLANNED"})


@dataclass
class GapNode:
    gap_id: str
    gap_type: str            # free-form: MISSING_KNOWLEDGE, MISSING_WORK_SUMMARY, POLICY_TRANSFORMATION_REQUIRED, ...
    subject: str
    scope: str               # "query:<normalized>" | "repo:<path>" | "file:<path>"
    severity: str            # CRITICAL | QUALITY | OPTIONAL
    status: str = "DETECTED"
    blocks: List[str] = field(default_factory=list)      # gap_ids (or "answer") this node blocks
    caused_by: List[str] = field(default_factory=list)   # upstream gap_ids; empty = this IS the root cause
    acquisition_methods: List[str] = field(default_factory=list)
    allowed_sources: List[str] = field(default_factory=list)
    max_depth: int = 1
    resolution: Optional[str] = None
    verified_by: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)

    def as_dict(self) -> Dict[str, Any]:
        return {
            "gap_id": self.gap_id, "gap_type": self.gap_type, "subject": self.subject,
            "scope": self.scope, "severity": self.severity, "status": self.status,
            "blocks": self.blocks, "caused_by": self.caused_by,
            "acquisition_methods": self.acquisition_methods, "allowed_sources": self.allowed_sources,
            "max_depth": self.max_depth, "resolution": self.resolution,
            "verified_by": self.verified_by, "created_at": self.created_at, "updated_at": self.updated_at,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "GapNode":
        return cls(
            gap_id=d["gap_id"], gap_type=d["gap_type"], subject=d["subject"],
            scope=d["scope"], severity=d["severity"], status=d.get("status", "DETECTED"),
            blocks=list(d.get("blocks", [])), caused_by=list(d.get("caused_by", [])),
            acquisition_methods=list(d.get("acquisition_methods", [])),
            allowed_sources=list(d.get("allowed_sources", [])),
            max_depth=d.get("max_depth", 1), resolution=d.get("resolution"),
            verified_by=list(d.get("verified_by", [])),
            created_at=d.get("created_at", 0.0), updated_at=d.get("updated_at", 0.0),
        )


class GapGraph:
    """One store per Vera store_path, alongside growth_signals.json etc.
    Resolved/blocked nodes are kept (never deleted) so the same scope/
    subject combination doesn't get re-detected from scratch — matches
    the design's explicit "keep as history, reuse next time" requirement.
    """

    def __init__(self) -> None:
        self.nodes: Dict[str, GapNode] = {}

    def create(
        self, gap_type: str, subject: str, scope: str, severity: str,
        *, blocks: Optional[List[str]] = None, caused_by: Optional[List[str]] = None,
        acquisition_methods: Optional[List[str]] = None, allowed_sources: Optional[List[str]] = None,
        max_depth: int = 1, status: str = "DETECTED",
    ) -> GapNode:
        if severity not in SEVERITIES:
            raise ValueError(f"bad severity: {severity}")
        if status not in ALL_STATUSES:
            raise ValueError(f"bad status: {status}")
        # Reuse: an existing node for the exact same (scope, subject) is
        # returned as-is rather than duplicated -- this is what lets a
        # RESOLVED node from a prior session short-circuit re-detection.
        existing = self.find_by_scope_subject(scope, subject)
        if existing is not None:
            return existing
        node = GapNode(
            gap_id=f"gap_{uuid.uuid4().hex[:8]}", gap_type=gap_type, subject=subject,
            scope=scope, severity=severity, status=status,
            blocks=list(blocks or []), caused_by=list(caused_by or []),
            acquisition_methods=list(acquisition_methods or []),
            allowed_sources=list(allowed_sources or []), max_depth=max_depth,
        )
        self.nodes[node.gap_id] = node
        return node

    def find_by_scope_subject(self, scope: str, subject: str) -> Optional[GapNode]:
        for node in self.nodes.values():
            if node.scope == scope and node.subject == subject:
                return node
        return None

    def get(self, gap_id: str) -> Optional[GapNode]:
        return self.nodes.get(gap_id)

    def set_status(self, gap_id: str, status: str, *, resolution: Optional[str] = None,
                    verified_by: Optional[List[str]] = None) -> Optional[GapNode]:
        if status not in ALL_STATUSES:
            raise ValueError(f"bad status: {status}")
        node = self.nodes.get(gap_id)
        if node is None:
            return None
        node.status = status
        node.updated_at = time.time()
        if resolution is not None:
            node.resolution = resolution
        if verified_by:
            node.verified_by.extend(verified_by)
        return node

    def actionable(self, *, limit: Optional[int] = None) -> List[GapNode]:
        """Nodes eligible for Sleep-mode resolution attempts, oldest first
        (simple FIFO -- no utility scoring in this pass, see plan's
        explicitly-deferred scope)."""
        out = sorted(
            (n for n in self.nodes.values() if n.status in ACTIONABLE_STATUSES),
            key=lambda n: n.created_at,
        )
        return out[:limit] if limit else out

    def since(self, ts: float) -> List[GapNode]:
        return [n for n in self.nodes.values() if n.updated_at >= ts]

    def save(self, path: Path) -> None:
        data = {gap_id: n.as_dict() for gap_id, n in self.nodes.items()}
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))

    @classmethod
    def load(cls, path: Path) -> "GapGraph":
        graph = cls()
        if not path.is_file():
            return graph
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError):
            return graph
        graph.nodes = {gap_id: GapNode.from_dict(d) for gap_id, d in data.items()}
        return graph


def gap_graph_path(store_path: Path) -> Path:
    return store_path.parent / "gap_graph.json"
