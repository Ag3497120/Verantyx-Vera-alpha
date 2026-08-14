"""Build the full shallow shelf: every jawiki lead, one new sovereign.

The slice measurement (tools/measure_shallow_shelf.py) priced this:
ingest is seconds per 10k leads, the sequential bz2 walk is the cost,
and hole-closing scales linearly with coverage (72% -> 68% from a 0.5%
slice, proportional to its probe-pool share). So the full build is a
background walk of the 4.4GB dump, batched ingest, and one save.

Atlas-widening only: the shelf lands in its own store file
(vera-corpus/build/jawiki_shallow.json), joins coverage.closing_domains
as one more shelf, and never pools into any census — the abstract-noun
warning (technical prose, 0% contradiction precision) stands; breadth
is for the map, not for the vote.
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from verantyx.cross_store import CrossStore
from verantyx.document_ingest import Document, ingest_documents

sys.path.insert(0, str(Path(__file__).resolve().parent))
from measure_shallow_shelf import pages  # noqa: E402  (same dump reader)

OUT = Path.home() / "Projects" / "vera-corpus" / "build" / "jawiki_shallow.json"
BATCH = 20_000

t0 = time.time()
shelf = CrossStore()
batch = []
n_docs = 0
for ti, lead in pages():
    if len(lead) < 30:
        continue
    batch.append(Document(source="jawiki-lead:%s" % ti,
                          text="%sは、%s" % (ti, lead[:280])))
    if len(batch) >= BATCH:
        ingest_documents(shelf, batch)
        n_docs += len(batch)
        batch = []
        print("ingested %d docs, %d cores, %.0fs"
              % (n_docs, len(shelf.crosses), time.time() - t0), flush=True)
if batch:
    ingest_documents(shelf, batch)
    n_docs += len(batch)

shelf.save(OUT)
print(json.dumps({
    "docs": n_docs, "cores": len(shelf.crosses),
    "out": str(OUT), "seconds": round(time.time() - t0, 1),
}, ensure_ascii=False))
