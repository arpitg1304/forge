"""Dataset hygiene linter for LeRobot-style robotics datasets.

Enforces Hugging Face's published recording guidelines against the defect
classes maintainers document as common on the Hub. Complements
:mod:`forge.quality`: quality scores *content*, lint checks *hygiene*.
"""

from forge.lint.config import LintConfig
from forge.lint.linter import DatasetLinter
from forge.lint.models import LintIssue, LintReport, Severity

__all__ = [
    "DatasetLinter",
    "LintConfig",
    "LintIssue",
    "LintReport",
    "Severity",
]
