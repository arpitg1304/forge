"""Registry for action tokenizer strategies.

Mirrors :class:`forge.formats.registry.FormatRegistry`: strategies register
themselves with a decorator and are instantiated by name.

    @TokenizerRegistry.register("openvla-bins")
    class OpenVLABinTokenizer(BaseTokenizer):
        ...

    tok = TokenizerRegistry.create("openvla-bins", num_bins=256)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from forge.core.exceptions import UnsupportedFormatError

if TYPE_CHECKING:
    from forge.tokenize.base import ActionTokenizer, BaseTokenizer


class TokenizerRegistry:
    """Central registry mapping strategy names to tokenizer classes."""

    _strategies: dict[str, type[BaseTokenizer]] = {}

    @classmethod
    def register(cls, name: str):
        """Class decorator that registers a tokenizer under ``name``."""

        def decorator(tok_cls: type[BaseTokenizer]) -> type[BaseTokenizer]:
            tok_cls.name = name
            cls._strategies[name] = tok_cls
            return tok_cls

        return decorator

    @classmethod
    def get(cls, name: str) -> type[BaseTokenizer]:
        """Return the tokenizer class registered under ``name``."""
        if name not in cls._strategies:
            raise UnsupportedFormatError(name, available_formats=cls.list_strategies())
        return cls._strategies[name]

    @classmethod
    def create(cls, name: str, **config) -> ActionTokenizer:
        """Instantiate the tokenizer registered under ``name``."""
        return cls.get(name)(**config)

    @classmethod
    def list_strategies(cls) -> list[str]:
        """Return registered strategy names, sorted."""
        return sorted(cls._strategies.keys())

    @classmethod
    def clear(cls) -> None:
        """Clear all registered strategies. Primarily for testing."""
        cls._strategies.clear()
