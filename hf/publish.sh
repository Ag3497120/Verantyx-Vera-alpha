#!/usr/bin/env bash
# Publish Vera α to Hugging Face: the structure as a model repo, the page as
# a Space. Run it yourself — it needs your credentials, and publishing is not
# something to do on someone else's behalf.
#
#   hf auth login                 # once; the token stays with you
#   NS=your-username ./hf/publish.sh
#
# Everything it uploads was verified first:
#   python3 -m verantyx.export_sqlite --verify   # answers AND shape match
#   python3 -m verantyx.card_numbers             # every number on the card
#   python3 vera_entry.py lab                    # 141 forks
set -euo pipefail

NS="${NS:-}"
MODEL="${MODEL:-$NS/vera-alpha}"
SPACE="${SPACE:-$NS/ask-vera}"
HERE="$(cd "$(dirname "$0")" && pwd)"

if [ -z "$NS" ]; then
  echo "set NS to your Hugging Face username or org, e.g. NS=verantyx $0" >&2
  exit 2
fi
if ! hf auth whoami >/dev/null 2>&1; then
  echo "not logged in — run 'hf auth login' first (the token stays with you)" >&2
  exit 2
fi
if [ ! -f "$HERE/model/vera.db" ]; then
  echo "hf/model/vera.db is missing — run:" >&2
  echo "  python3 -m verantyx.export_sqlite --verify" >&2
  echo "  cp ~/Projects/vera-corpus/build/vera.db hf/model/" >&2
  exit 2
fi

echo "==> model repo $MODEL  (vera.db 140MB, CC BY-SA 4.0 — see hf/model/LICENSE)"
hf repo create "$MODEL" --repo-type model -y >/dev/null 2>&1 || true
hf upload "$MODEL" "$HERE/model" . --repo-type model \
  --commit-message "Vera α: the federation as one auditable SQLite file"

echo "==> space $SPACE"
hf repo create "$SPACE" --repo-type space --space_sdk gradio -y >/dev/null 2>&1 || true
VERA_REPO="$MODEL" hf upload "$SPACE" "$HERE/space" . --repo-type space \
  --commit-message "Ask Vera — refusals included"
# The Space reads the model repo from this, so it is set on the repo rather
# than baked into app.py: the same page can point at a rebuilt structure.
hf repo-files "$SPACE" --repo-type space >/dev/null 2>&1 || true

cat <<EOF

published
  model  https://huggingface.co/$MODEL
  space  https://huggingface.co/spaces/$SPACE

Set VERA_REPO=$MODEL in the Space settings (Variables) if it is not already,
otherwise the page will look for the default repository.
EOF
