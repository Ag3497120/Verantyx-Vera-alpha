"""Verantyx Vera α — deterministic cross-structure knowledge & reasoning engine.

No language model. Knowledge lives as crosses (core + facet counts), reasoning
is multi-frontier consensus with typed verdicts (ANSWER / AMBIGUOUS /
UNKNOWN_*). Arithmetic, term rewriting, Kripke model checking, and Python
code-reasoning run on the same substrate.
"""
from .cross_store import CrossStore, pour_corpus
from .consensus import (
    ConsensusConfig,
    ConsensusResult,
    matryoshka_consensus,
    run_consensus,
)
from .consensus_store import (
    candidates_for_query,
    consensus_over_store,
    probe_coverage,
)
from .sense_split import facet_clusters, select_cluster, sense_facets
from .math_sim import eval_expr, math_ask, solve_equation, wire_add, wire_mul, wire_sub
from .rewrite_kernel import (
    RuleStore,
    default_algebra_rules,
    default_logic_rules,
    parse_term,
    simplify,
    term_to_str,
)
from .kripke import KripkeModel, check as kripke_check, parse_formula
from .code_ingest import calls_of, code_ask, impact, ingest_python_repo, who_calls

__version__ = "0.1.0a1"

__all__ = [
    "CrossStore",
    "pour_corpus",
    "ConsensusConfig",
    "ConsensusResult",
    "run_consensus",
    "matryoshka_consensus",
    "consensus_over_store",
    "candidates_for_query",
    "probe_coverage",
    "facet_clusters",
    "select_cluster",
    "sense_facets",
    "math_ask",
    "eval_expr",
    "solve_equation",
    "wire_add",
    "wire_sub",
    "wire_mul",
    "RuleStore",
    "simplify",
    "parse_term",
    "term_to_str",
    "default_algebra_rules",
    "default_logic_rules",
    "KripkeModel",
    "kripke_check",
    "parse_formula",
    "ingest_python_repo",
    "code_ask",
    "who_calls",
    "calls_of",
    "impact",
    "__version__",
]
