"""Data models for the dataset linter.

A :class:`LintIssue` is a single rule violation; a :class:`LintReport` collects
every issue found across a dataset and summarizes pass/warn/fail status.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum


class Severity(str, Enum):
    """How serious a lint violation is.

    ``ERROR`` breaks training or downstream tooling; ``WARN`` is a quality
    smell that degrades policies but still loads; ``INFO`` is advisory.
    """

    ERROR = "error"
    WARN = "warn"
    INFO = "info"


@dataclass
class LintIssue:
    """A single rule violation.

    Attributes:
        code: Stable machine-readable id (e.g. ``"task.too_short"``).
        severity: :class:`Severity` level.
        message: Human-readable description of the problem.
        scope: ``"dataset"`` for dataset-wide issues, otherwise the episode id.
        hint: Optional remediation suggestion.
    """

    code: str
    severity: Severity
    message: str
    scope: str = "dataset"
    hint: str | None = None

    def to_dict(self) -> dict:
        return {
            "code": self.code,
            "severity": self.severity.value,
            "message": self.message,
            "scope": self.scope,
            "hint": self.hint,
        }


@dataclass
class LintReport:
    """Aggregated lint results for one dataset."""

    dataset_path: str
    num_episodes: int = 0
    issues: list[LintIssue] = field(default_factory=list)

    def add(self, issue: LintIssue) -> None:
        self.issues.append(issue)

    @property
    def errors(self) -> list[LintIssue]:
        return [i for i in self.issues if i.severity is Severity.ERROR]

    @property
    def warnings(self) -> list[LintIssue]:
        return [i for i in self.issues if i.severity is Severity.WARN]

    @property
    def infos(self) -> list[LintIssue]:
        return [i for i in self.issues if i.severity is Severity.INFO]

    @property
    def passed(self) -> bool:
        """A dataset passes lint when it has no ERROR-level issues."""
        return len(self.errors) == 0

    def counts_by_code(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for issue in self.issues:
            out[issue.code] = out.get(issue.code, 0) + 1
        return out

    def to_dict(self) -> dict:
        return {
            "dataset_path": self.dataset_path,
            "num_episodes": self.num_episodes,
            "passed": self.passed,
            "summary": {
                "errors": len(self.errors),
                "warnings": len(self.warnings),
                "infos": len(self.infos),
            },
            "counts_by_code": self.counts_by_code(),
            "issues": [i.to_dict() for i in self.issues],
        }
