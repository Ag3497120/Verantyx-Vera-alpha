"""The refusal log as a standing work queue — growth where questions failed.

Measured before it was automated: 34 subjects the engine refused during one
week of operation, fetched by name, raised subject accuracy to 13/14 across
every suffix, moved 時効 from a one-document accident to a ranked answer
with three witnesses, and lifted ranked leads from 7/12 to 10/12. The fetch
list was not curated — it was derived from a stated rule, and that is what
makes the growth honest: the corpus thickens where the QUESTIONS said it
was thin, not where anyone thought it should.

    queue   vera.ask appends the subject of every UNKNOWN_NOT_PRESENT to
            $VERA_QUEUE (one JSON line each), when that variable is set.
            Off by default: an ask that writes files is a side effect a
            library must opt into, not perform silently
    grow    python3 -m verantyx.grow --queue refusals.jsonl
            dedupes, drops subjects the store meanwhile holds, fetches the
            rest by title with the rule recorded in the manifest, rebuilds,
            re-exports, verifies

Two verdicts stay OUT of the queue on purpose (see `remedy`): TIME_DEPENDENT
does not close by registration and NO_SUBJECT should not be closed — a
queue that accumulates unclosable items is a queue nobody trusts.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

#: The one verdict whose remedy is "register documents about the subject".
QUEUEABLE = "UNKNOWN_NOT_PRESENT"


def log_refusal(result: Dict[str, Any], path: Optional[str] = None) -> bool:
    """Append a queueable refusal to the queue file. Returns whether it did."""
    path = path or os.environ.get("VERA_QUEUE")
    if not path or result.get("verdict") != QUEUEABLE:
        return False
    subject = result.get("subject")
    if not subject:
        return False
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"subject": subject,
                            "asked": time.strftime("%Y-%m-%d")},
                           ensure_ascii=False) + "\n")
    return True


def pending(queue: Path, store: Any) -> List[str]:
    """Deduplicated subjects the store still does not hold, oldest first."""
    seen: List[str] = []
    for line in Path(queue).read_text(encoding="utf-8").splitlines():
        try:
            s = json.loads(line).get("subject")
        except Exception:
            continue
        if s and s not in seen and s not in store.crosses:
            seen.append(s)
    return seen


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--queue", required=True)
    ap.add_argument("--root", default=str(Path.home() / "Projects" / "vera-corpus"))
    ap.add_argument("--dry-run", action="store_true",
                    help="show what would be fetched, fetch nothing")
    ap.add_argument("--limit", type=int, default=50)
    a = ap.parse_args(argv)
    root = Path(a.root)

    from .export_sqlite import load

    store = load(root / "build" / "vera.db")["ja"]
    subjects = pending(Path(a.queue), store)[:a.limit]
    if not subjects:
        print(json.dumps({"verdict": "ANSWER", "pending": 0,
                          "note": "every queued subject is now held"},
                         ensure_ascii=False))
        return 0
    if a.dry_run:
        print(json.dumps({"verdict": "ANSWER", "pending": len(subjects),
                          "would_fetch": subjects, "dry_run": True},
                         ensure_ascii=False, indent=1))
        return 0

    from .corpus_wikipedia import fetch_titles

    stamp = time.strftime("%Y%m%d")
    manifest = Path("corpora") / f"wikipedia_ja_queue_{stamp}.json"
    got = fetch_titles(
        subjects, root / "wikipedia_named", manifest,
        rule=("subjects refused as UNKNOWN_NOT_PRESENT, drawn from the "
              "operational queue %s; nothing curated" % a.queue),
        label="ja.wikipedia 拒否キュー由来 %s" % stamp)

    # Rebuild through the same front doors everything else uses.
    from .build_ja import main as build_main
    from .export_sqlite import main as export_main

    build_main(["--root", str(root), "--rebuild"])
    code = export_main(["--root", str(root), "--verify"])
    print(json.dumps({"verdict": "ANSWER" if code == 0 else "DRIFTED",
                      "fetched": got["articles"], "missing": got["missing"],
                      "manifest": str(manifest)}, ensure_ascii=False, indent=1))
    return code


if __name__ == "__main__":
    sys.exit(main())
