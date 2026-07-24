"""Tiny terminal UI — arrow-key selection with a dumb-terminal fallback.

No dependencies. `select(...)` renders a menu navigated with ↑/↓ (or j/k),
Enter confirms, Esc/q cancels. When stdin is not a TTY (CI, pipes, forks)
it falls back to reading an index number, so everything stays scriptable
and deterministic in tests.
"""
from __future__ import annotations

import sys
from typing import List, Optional, Sequence


def _is_tty() -> bool:
    try:
        return sys.stdin.isatty() and sys.stdout.isatty()
    except Exception:
        return False


def _read_key() -> str:
    import termios
    import tty

    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
        if ch == "\x1b":  # escape sequence
            nxt = sys.stdin.read(1)
            if nxt == "[":
                arrow = sys.stdin.read(1)
                return {"A": "up", "B": "down"}.get(arrow, "esc")
            return "esc"
        return ch
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)


def select(
    title: str,
    options: Sequence[str],
    *,
    default: int = 0,
    descriptions: Optional[Sequence[str]] = None,
) -> Optional[int]:
    """Return the chosen index, or None if cancelled."""
    if not options:
        return None
    if not _is_tty():
        print(title)
        for i, o in enumerate(options):
            print(f"  [{i}] {o}")
        try:
            raw = input(f"select 0-{len(options)-1} (default {default}): ").strip()
        except EOFError:
            return default
        if raw == "":
            return default
        try:
            i = int(raw)
            return i if 0 <= i < len(options) else None
        except ValueError:
            return None

    idx = max(0, min(default, len(options) - 1))
    n_lines = len(options) + 1
    print(title)
    first = True
    while True:
        if not first:
            sys.stdout.write(f"\x1b[{n_lines - 1}A")
        first = False
        for i, o in enumerate(options):
            marker = "▶" if i == idx else " "
            desc = ""
            if descriptions and i == idx and descriptions[i]:
                desc = f"  — {descriptions[i]}"
            line = f" {marker} {o}{desc}"
            sys.stdout.write("\x1b[2K" + line[:120] + "\n")
        sys.stdout.flush()
        key = _read_key()
        if key in ("up", "k"):
            idx = (idx - 1) % len(options)
        elif key in ("down", "j"):
            idx = (idx + 1) % len(options)
        elif key in ("\r", "\n"):
            return idx
        elif key in ("esc", "q", "\x03"):
            return None


def confirm_action(
    summary: str,
    *,
    detail: str = "",
    allow_always: bool = True,
) -> str:
    """Approval gate for agent actions → 'approve' | 'always' | 'deny'."""
    print(f"\n┌─ approval required ─────────────────────────")
    print(f"│ {summary}")
    for line in (detail or "").splitlines()[:12]:
        print(f"│   {line[:110]}")
    print(f"└─────────────────────────────────────────────")
    opts: List[str] = ["Approve (once)"]
    keys = ["approve"]
    if allow_always:
        opts.append("Always allow this tool (this session)")
        keys.append("always")
    opts.append("Deny")
    keys.append("deny")
    i = select("", opts, default=0)
    return keys[i] if i is not None else "deny"
