"""Action tokenizer featureset for Forge.

Turns the canonical ``Frame.action`` stream into discrete tokens and back
(``fit -> encode -> decode``), with proven strategies shipped in-box, a registry
for extension, and a comparator that benchmarks strategies on your dataset.
"""

from __future__ import annotations

# Import strategies so they self-register in the registry.
from forge.tokenize import strategies as _strategies  # noqa: F401
from forge.tokenize.analyzer import TokenizerComparator
from forge.tokenize.base import (
    ActionTokenizer,
    BaseTokenizer,
    TokenizerError,
    TokenizerNotFittedError,
    load_tokenizer,
)
from forge.tokenize.models import ComparisonReport, TokenizerStats
from forge.tokenize.registry import TokenizerRegistry

__all__ = [
    "ActionTokenizer",
    "BaseTokenizer",
    "TokenizerError",
    "TokenizerNotFittedError",
    "load_tokenizer",
    "TokenizerRegistry",
    "TokenizerComparator",
    "ComparisonReport",
    "TokenizerStats",
]
