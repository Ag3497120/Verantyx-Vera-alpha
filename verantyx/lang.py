"""Multilingual front-end — deterministic, rule-based, honest about depth.

Vera's cross substrate is language-agnostic (cores and facets are just
symbols); what differs per language is segmentation and function-word
filtering. This module provides:

  detect(text)              → "ja" | "en" | "latin"  (script heuristic)
  ingest_text(store, text)  → routes to the right elementary classifier
  ja_ask(store, query)      → elementary Japanese recall path

Depth per language (v0, no external tokenizers):
  en     — full elementary grammar pipeline (en_decompose)
  ja     — script-run segmentation (kanji/katakana runs = content,
           particles/hiragana = function). Recall path, no consensus yet.
  latin  — generic content-word window with per-language function stoplists
           (es / fr / de shipped; others fall back to the shared list)
"""
from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Tuple

from .cross_store import CrossStore

# ---------------------------------------------------------------------------
# script detection
# ---------------------------------------------------------------------------

_RE_JA = re.compile(r"[぀-ヿ一-鿿]")
_RE_LATIN = re.compile(r"[A-Za-zÀ-ɏ]")


def detect(text: str) -> str:
    t = text or ""
    n_ja = len(_RE_JA.findall(t))
    n_lat = len(_RE_LATIN.findall(t))
    if n_ja > n_lat:
        return "ja"
    if n_lat == 0:
        return "latin"
    # English vs other latin languages is decided by the caller via lang=;
    # auto default treats latin as English (richest pipeline).
    return "en"


# ---------------------------------------------------------------------------
# Japanese (elementary, tokenizer-free)
# ---------------------------------------------------------------------------

# content run: kanji/katakana sequence, optionally trailing い/な (adjectives)
_JA_RUN = re.compile(r"[゠-ヿ]+|[一-鿿]+[いな]?")
_JA_STOP = {"事", "物", "為", "様", "何", "誰", "所"}
_JA_QUESTION = ("何", "誰", "どこ", "いつ", "なぜ", "どう", "ですか", "とは")


def ja_content_runs(text: str) -> List[str]:
    return [r for r in _JA_RUN.findall(text or "") if r not in _JA_STOP]


def ja_ingest_sentence(store: CrossStore, text: str) -> Optional[str]:
    """First content run = core, remaining runs = facets (accumulating)."""
    runs = ja_content_runs(text)
    if not runs:
        return None
    core, facets = runs[0], [r for r in runs[1:] if r != runs[0]]
    # `source=` is what CrossStore records as provenance, and the English
    # path has always passed it. Without it, anything reading provenance —
    # document_ingest's source attribution, which is the entire point of
    # ingesting several sources — got an empty answer for Japanese and could
    # not tell that it was empty because the language was unsupported rather
    # than because the sources agreed.
    store.add(core, dict.fromkeys(facets), source=text.strip())
    store.n_sentences += 1
    return core


def ja_ask(store: CrossStore, query: str, *, k: int = 5) -> Dict[str, Any]:
    """Elementary recall: first known content run answers with its facets.

    Honest scope: this is recall + refusal, not multi-frontier consensus
    (the consensus decomposer is English-only in v0).
    """
    runs = ja_content_runs(query)
    for r in runs:
        for key in (r, r + "#p"):
            if store.has(key):
                facets = [f for f, _ in store.top_facets(key, k)]
                return {
                    "verdict": "ANSWER",
                    "core": r,
                    "facets": facets,
                    "text": r + "は" + "、".join(facets) if facets else r,
                    "lang": "ja",
                    "mode": "recall",
                }
    if runs:
        return {
            "verdict": "UNKNOWN_NO_EVIDENCE",
            "core": None,
            "queried": runs,
            "lang": "ja",
        }
    return {"verdict": "UNKNOWN_UNPARSED", "lang": "ja"}


# ---------------------------------------------------------------------------
# generic latin languages (es / fr / de + fallback)
# ---------------------------------------------------------------------------

_FUNCTION_LATIN: Dict[str, set] = {
    "es": {"el", "la", "los", "las", "un", "una", "unos", "unas", "de", "del",
           "en", "y", "o", "es", "son", "que", "por", "para", "con", "su",
           "se", "al", "lo", "como", "más", "muy", "no", "este", "esta"},
    "fr": {"le", "la", "les", "un", "une", "des", "de", "du", "en", "et",
           "ou", "est", "sont", "que", "qui", "pour", "avec", "son", "sa",
           "ses", "au", "aux", "ce", "cette", "très", "ne", "pas", "plus"},
    "de": {"der", "die", "das", "ein", "eine", "einen", "einem", "und",
           "oder", "ist", "sind", "von", "im", "in", "mit", "für", "auf",
           "zu", "den", "dem", "des", "sich", "nicht", "sehr", "auch"},
}
_FUNCTION_SHARED = {"the", "a", "an", "is", "are", "of", "and", "or", "in"}

_LATIN_WORD = re.compile(r"[A-Za-zÀ-ɏ]+")


def latin_ingest_sentence(
    store: CrossStore, text: str, lang: str = "latin"
) -> Optional[str]:
    """Generic: first content word = core, other content words = facets."""
    stop = _FUNCTION_LATIN.get(lang, set()) | _FUNCTION_SHARED
    words = [w.casefold() for w in _LATIN_WORD.findall(text or "")]
    content = [w for w in words if w not in stop and len(w) > 1]
    if not content:
        return None
    core, facets = content[0], [w for w in content[1:] if w != content[0]]
    store.add(core, dict.fromkeys(facets))
    store.n_sentences += 1
    return core


# ---------------------------------------------------------------------------
# unified entry
# ---------------------------------------------------------------------------

def ingest_text(
    store: CrossStore, text: str, *, lang: str = "auto"
) -> Dict[str, Any]:
    """Sentence-split & ingest in the detected/forced language."""
    if lang == "auto":
        lang = detect(text)
    n = 0
    cores: List[str] = []
    if lang == "ja":
        for sent in re.split(r"[。！？\n]", text or ""):
            key = ja_ingest_sentence(store, sent)
            if key:
                cores.append(key)
                n += 1
    elif lang == "en":
        from .en_decompose import split_sentences

        for sent in split_sentences(text or ""):
            key = store.ingest_sentence(sent)
            if key:
                cores.append(key)
                n += 1
    else:
        for sent in re.split(r"[.!?\n]", text or ""):
            key = latin_ingest_sentence(store, sent, lang)
            if key:
                cores.append(key)
                n += 1
    return {"lang": lang, "n_sentences": n, "cores": cores}


_TASK_LABEL = re.compile(
    r"^\s*(problem|exercise|question|task|q|puzzle)\s*\d*\s*[:.\)]", re.I
)
# Cue words that mark "asking to do something" regardless of position —
# needed once multi-line paste is supported: a labeled academic problem
# ("Problem 1: ... prove or disprove ...") defeats a first-word-only check,
# since the first token is the label ("problem"), not the verb.
_TASK_CUE_WORDS = {
    "prove", "disprove", "derive", "verify", "compute", "calculate",
    "determine", "evaluate", "solve", "show", "demonstrate",
}


def is_question(text: str, lang: str = "auto") -> bool:
    """Declarative vs interrogative/imperative/task — gate for auto-memory
    in chat (a task must never be mistaken for a fact to remember)."""
    t = (text or "").strip()
    if lang == "auto":
        lang = detect(t)
    if t.endswith("?") or t.endswith("？"):
        return True
    if lang == "ja":
        return any(m in t for m in _JA_QUESTION)
    if _TASK_LABEL.match(t):
        return True
    words = set(t.casefold().split())
    if words & _TASK_CUE_WORDS:
        return True
    head = t.casefold().split()[:1]
    # interrogatives AND request/imperative heads: neither is a fact to
    # remember ("tell me something" must not become a knowledge cross)
    return bool(head) and head[0] in {
        "what", "who", "where", "when", "why", "how", "which", "is", "are",
        "do", "does", "did", "can", "could", "should", "would", "solve",
        "tell", "give", "show", "explain", "describe", "list", "write",
        "help", "please", "find", "search", "make", "let",
    }
