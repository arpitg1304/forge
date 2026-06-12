"""Core interface for action tokenizers.

An action tokenizer converts a stream of continuous action vectors
(``Frame.action``) into discrete integer tokens and back, following a
``fit -> encode -> decode`` lifecycle. The :class:`ActionTokenizer` Protocol
defines the contract; :class:`BaseTokenizer` provides shared save/load plumbing
on top of ``get_params``/``from_params``.
"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from pathlib import Path
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from forge.core.exceptions import ForgeError

if TYPE_CHECKING:
    from numpy.typing import NDArray


class TokenizerError(ForgeError):
    """Base error for the tokenizer featureset."""


class TokenizerNotFittedError(TokenizerError):
    """Raised when encode/decode is called before fit()."""

    def __init__(self, name: str = "tokenizer"):
        super().__init__(
            f"{name} must be fitted before encode/decode. Call .fit(actions) first."
        )


@runtime_checkable
class ActionTokenizer(Protocol):
    """Protocol every action tokenizer satisfies."""

    name: str

    @property
    def vocab_size(self) -> int: ...

    @property
    def granularity(self) -> str: ...

    @property
    def is_fitted(self) -> bool: ...

    def fit(self, actions: NDArray) -> ActionTokenizer: ...

    def encode(self, actions: NDArray) -> NDArray: ...

    def decode(self, tokens: NDArray) -> NDArray: ...

    def get_params(self) -> dict: ...

    @classmethod
    def from_params(cls, params: dict) -> ActionTokenizer: ...


class BaseTokenizer(ABC):
    """Abstract base implementing shared save/load on top of params.

    Subclasses set the class attribute ``name`` (usually via the registry
    decorator) and implement ``vocab_size``, ``fit``, ``encode``, ``decode``,
    ``get_params`` and ``from_params``.
    """

    name: str = ""

    def __init__(self) -> None:
        self._fitted = False

    @property
    @abstractmethod
    def vocab_size(self) -> int:
        """Number of distinct token ids per dimension."""

    @property
    def granularity(self) -> str:
        """``"per_step"`` (one token per dim per frame) or ``"chunk"``."""
        return "per_step"

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @abstractmethod
    def fit(self, actions: NDArray) -> BaseTokenizer:
        """Learn parameters from an ``(N, D)`` action corpus. Returns self."""

    @abstractmethod
    def encode(self, actions: NDArray) -> NDArray:
        """Map ``(T, D)`` float actions to integer tokens."""

    @abstractmethod
    def decode(self, tokens: NDArray) -> NDArray:
        """Inverse of :meth:`encode`; returns ``(T, D)`` floats."""

    @abstractmethod
    def get_params(self) -> dict:
        """Return JSON-serializable fitted parameters."""

    @classmethod
    @abstractmethod
    def from_params(cls, params: dict) -> BaseTokenizer:
        """Reconstruct a fitted tokenizer from :meth:`get_params` output."""

    def _check_fitted(self) -> None:
        if not self._fitted:
            raise TokenizerNotFittedError(self.name or type(self).__name__)

    def save(self, path: str | Path) -> None:
        """Serialize the fitted tokenizer to JSON (embeds the registry name)."""
        self._check_fitted()
        payload = {"name": self.name, "params": self.get_params()}
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(payload, f, indent=2)


def load_tokenizer(path: str | Path) -> ActionTokenizer:
    """Load a tokenizer saved with :meth:`BaseTokenizer.save`.

    Dispatches to the registered strategy via the embedded ``name`` and
    rebuilds it with ``from_params``.
    """
    from forge.tokenize.registry import TokenizerRegistry

    with open(path) as f:
        payload = json.load(f)
    cls = TokenizerRegistry.get(payload["name"])
    return cls.from_params(payload["params"])
