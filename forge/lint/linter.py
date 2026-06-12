"""Dataset linter — enforce Hugging Face's LeRobot recording guidelines.

:class:`DatasetLinter` walks a dataset's episodes and flags the defect classes
that LeRobot maintainers document as rampant on the Hub: vague or missing task
annotations, ambiguous camera naming, broken (1–2 frame) episodes, and
inconsistent action/state dimensions across episodes of the same dataset.

Pure metadata inspection — no video decode, no proprioception number-crunching.
Complements :mod:`forge.quality` (which scores *content*); lint checks *hygiene*.
"""

from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING

from forge.lint.config import LintConfig
from forge.lint.models import LintIssue, LintReport, Severity

if TYPE_CHECKING:
    from forge.core.models import DatasetInfo, Episode


class DatasetLinter:
    """Run hygiene checks over a dataset.

    Two entry points read from two sources:

    * :meth:`lint_metadata` (the default for :meth:`lint_dataset`) reads a
      reader's inspected :class:`DatasetInfo` — cameras, language coverage and
      schemas — which is reliable across every format. This is what the
      ``forge lint`` CLI uses.
    * :meth:`lint_episodes` reads per-episode fields directly; use it when a
      reader actually populates ``Episode.cameras`` / ``language_instruction``
      (some formats don't), or for in-memory episodes.
    """

    def __init__(self, config: LintConfig | None = None, reader=None) -> None:
        self.config = config or LintConfig()
        self._reader = reader

    # -- public entry points ------------------------------------------------ #

    def lint_dataset(self, path: str | Path) -> LintReport:
        """Inspect ``path`` and lint its dataset-level metadata.

        Uses the reader's ``inspect`` (not a full episode scan), so it is cheap
        and correct even for readers that leave per-episode fields empty.
        """
        reader = self._reader or self._resolve_reader(path)
        info = reader.inspect(Path(path))
        return self.lint_metadata(info, dataset_path=str(path))

    def lint_metadata(
        self, info: DatasetInfo, dataset_path: str | None = None
    ) -> LintReport:
        """Lint a dataset's inspected :class:`DatasetInfo`."""
        cfg = self.config
        report = LintReport(
            dataset_path=dataset_path or str(info.path),
            num_episodes=info.num_episodes,
        )

        if info.num_episodes == 0:
            report.add(
                LintIssue(
                    code="dataset.empty",
                    severity=Severity.ERROR,
                    message="Dataset contains no episodes.",
                )
            )

        self._check_language_coverage(info, report)
        self._check_sample_task(info, report)
        self._check_metadata_cameras(info, report)
        self._check_schemas(info, report)
        return report

    # -- metadata rules ----------------------------------------------------- #

    def _check_language_coverage(
        self, info: DatasetInfo, report: LintReport
    ) -> None:
        if not info.has_language:
            report.add(
                LintIssue(
                    code="task.missing",
                    severity=Severity.WARN,
                    message="Dataset has no language / task instructions.",
                    hint="VLA training needs per-episode instructions; add them.",
                )
            )
        elif info.language_coverage < 1.0:
            report.add(
                LintIssue(
                    code="task.partial_coverage",
                    severity=Severity.WARN,
                    message=(
                        f"Only {info.language_coverage:.0%} of episodes have a "
                        f"task instruction."
                    ),
                    hint="Annotate the remaining episodes.",
                )
            )

    def _check_sample_task(self, info: DatasetInfo, report: LintReport) -> None:
        cfg = self.config
        task = info.sample_language
        if not task or not task.strip():
            return
        stripped = task.strip()
        normalized = stripped.lower().rstrip(".!").strip()
        if normalized in cfg.task_generic_phrases:
            report.add(
                LintIssue(
                    code="task.generic",
                    severity=Severity.WARN,
                    message=f"Sample task is a meaningless placeholder: {stripped!r}.",
                    hint="Describe the actual goal, e.g. 'Pick up the red cube'.",
                )
            )
        elif len(stripped) < cfg.task_min_chars:
            report.add(
                LintIssue(
                    code="task.too_short",
                    severity=Severity.WARN,
                    message=(
                        f"Sample task is only {len(stripped)} chars "
                        f"(< {cfg.task_min_chars}): {stripped!r}."
                    ),
                    hint=f"Aim for {cfg.task_min_chars}–{cfg.task_max_chars} chars.",
                )
            )

    def _check_metadata_cameras(
        self, info: DatasetInfo, report: LintReport
    ) -> None:
        cfg = self.config
        cams = info.cameras
        if not cams:
            report.add(
                LintIssue(
                    code="camera.none",
                    severity=Severity.WARN,
                    message="Dataset declares no cameras.",
                )
            )
            return

        if len(cams) < cfg.min_camera_views:
            report.add(
                LintIssue(
                    code="camera.too_few_views",
                    severity=Severity.INFO,
                    message=(
                        f"Only {len(cams)} camera view(s); HF recommends "
                        f">= {cfg.min_camera_views}."
                    ),
                )
            )

        for name, cam in cams.items():
            leaf = name.split(".")[-1].lower()
            if leaf in cfg.ambiguous_cam_names:
                report.add(
                    LintIssue(
                        code="camera.ambiguous_name",
                        severity=Severity.WARN,
                        message=(
                            f"Camera {name!r} doesn't say where it is "
                            f"(third-person vs wrist?)."
                        ),
                        hint="Use <modality>.<location>, e.g. 'observation.images.wrist'.",
                    )
                )
            if cam.height < cfg.min_cam_height or cam.width < cfg.min_cam_width:
                report.add(
                    LintIssue(
                        code="camera.low_resolution",
                        severity=Severity.INFO,
                        message=(
                            f"Camera {name!r} is {cam.width}x{cam.height}; "
                            f"below {cfg.min_cam_width}x{cfg.min_cam_height}."
                        ),
                    )
                )

    def _check_schemas(self, info: DatasetInfo, report: LintReport) -> None:
        if info.action_schema is None:
            report.add(
                LintIssue(
                    code="action.missing",
                    severity=Severity.WARN,
                    message="Dataset has no action field; can't train a policy on it.",
                )
            )

    def lint_episodes(
        self, episodes: Iterable[Episode], dataset_path: str = "<episodes>"
    ) -> LintReport:
        """Lint an iterable of episodes."""
        report = LintReport(dataset_path=dataset_path)

        # Dataset-wide accumulators for cross-episode consistency checks.
        action_dims: dict[int, int] = {}
        state_dims: dict[int, int] = {}
        camera_name_sets: set[frozenset[str]] = set()

        for ep in episodes:
            report.num_episodes += 1
            scope = ep.episode_id or f"episode_{report.num_episodes - 1}"

            self._check_task(ep, scope, report)
            self._check_episode_length(ep, scope, report)
            self._check_cameras(ep, scope, report)

            if ep.action_dim is not None:
                action_dims[ep.action_dim] = action_dims.get(ep.action_dim, 0) + 1
            if ep.state_dim is not None:
                state_dims[ep.state_dim] = state_dims.get(ep.state_dim, 0) + 1
            if ep.cameras:
                camera_name_sets.add(frozenset(ep.cameras.keys()))

        self._check_dimension_consistency(action_dims, "action", report)
        self._check_dimension_consistency(state_dims, "state", report)
        self._check_camera_consistency(camera_name_sets, report)

        if report.num_episodes == 0:
            report.add(
                LintIssue(
                    code="dataset.empty",
                    severity=Severity.ERROR,
                    message="Dataset contains no episodes.",
                )
            )
        return report

    # -- per-episode rules -------------------------------------------------- #

    def _check_task(self, ep: Episode, scope: str, report: LintReport) -> None:
        cfg = self.config
        task = ep.language_instruction
        if task is None or not task.strip():
            report.add(
                LintIssue(
                    code="task.missing",
                    severity=Severity.WARN,
                    message="Episode has no task / language instruction.",
                    scope=scope,
                    hint="VLA training needs per-episode instructions; add one.",
                )
            )
            return

        stripped = task.strip()
        normalized = stripped.lower().rstrip(".!").strip()
        if normalized in cfg.task_generic_phrases:
            report.add(
                LintIssue(
                    code="task.generic",
                    severity=Severity.WARN,
                    message=f"Task is a placeholder with no specific meaning: {stripped!r}.",
                    scope=scope,
                    hint="Describe the actual goal, e.g. 'Pick up the red cube'.",
                )
            )
        elif len(stripped) < cfg.task_min_chars:
            report.add(
                LintIssue(
                    code="task.too_short",
                    severity=Severity.WARN,
                    message=(
                        f"Task is only {len(stripped)} chars "
                        f"(< {cfg.task_min_chars}): {stripped!r}."
                    ),
                    scope=scope,
                    hint=f"Aim for {cfg.task_min_chars}–{cfg.task_max_chars} chars.",
                )
            )
        elif len(stripped) > cfg.task_max_chars:
            report.add(
                LintIssue(
                    code="task.too_long",
                    severity=Severity.INFO,
                    message=(
                        f"Task is {len(stripped)} chars "
                        f"(> {cfg.task_max_chars}); consider trimming."
                    ),
                    scope=scope,
                )
            )

    def _check_episode_length(
        self, ep: Episode, scope: str, report: LintReport
    ) -> None:
        n = ep.num_frames
        if n is not None and n < self.config.min_episode_frames:
            report.add(
                LintIssue(
                    code="episode.too_short",
                    severity=Severity.ERROR,
                    message=f"Episode has only {n} frame(s); likely a recording glitch.",
                    scope=scope,
                    hint="Drop empty/aborted episodes before training.",
                )
            )

    def _check_cameras(self, ep: Episode, scope: str, report: LintReport) -> None:
        cfg = self.config
        cams = ep.cameras
        if not cams:
            report.add(
                LintIssue(
                    code="camera.none",
                    severity=Severity.WARN,
                    message="Episode declares no cameras.",
                    scope=scope,
                )
            )
            return

        if len(cams) < cfg.min_camera_views:
            report.add(
                LintIssue(
                    code="camera.too_few_views",
                    severity=Severity.INFO,
                    message=(
                        f"Only {len(cams)} camera view(s); HF recommends "
                        f">= {cfg.min_camera_views}."
                    ),
                    scope=scope,
                )
            )

        for name, cam in cams.items():
            leaf = name.split(".")[-1].lower()
            if leaf in cfg.ambiguous_cam_names:
                report.add(
                    LintIssue(
                        code="camera.ambiguous_name",
                        severity=Severity.WARN,
                        message=(
                            f"Camera {name!r} doesn't say where it is "
                            f"(third-person vs wrist?)."
                        ),
                        scope=scope,
                        hint="Use <modality>.<location>, e.g. 'observation.images.wrist'.",
                    )
                )
            if cam.height < cfg.min_cam_height or cam.width < cfg.min_cam_width:
                report.add(
                    LintIssue(
                        code="camera.low_resolution",
                        severity=Severity.INFO,
                        message=(
                            f"Camera {name!r} is {cam.width}x{cam.height}; "
                            f"below {cfg.min_cam_width}x{cfg.min_cam_height}."
                        ),
                        scope=scope,
                    )
                )

    # -- dataset-wide rules ------------------------------------------------- #

    @staticmethod
    def _check_dimension_consistency(
        dims: dict[int, int], kind: str, report: LintReport
    ) -> None:
        if len(dims) > 1:
            spread = ", ".join(
                f"{d}d×{n}ep" for d, n in sorted(dims.items(), key=lambda x: -x[1])
            )
            report.add(
                LintIssue(
                    code=f"{kind}.inconsistent_dim",
                    severity=Severity.ERROR,
                    message=(
                        f"Episodes disagree on {kind} dimensionality ({spread}); "
                        f"a single dataset should be uniform."
                    ),
                    hint="Split mismatched embodiments into separate datasets.",
                )
            )

    @staticmethod
    def _check_camera_consistency(
        name_sets: set[frozenset[str]], report: LintReport
    ) -> None:
        if len(name_sets) > 1:
            report.add(
                LintIssue(
                    code="camera.inconsistent_keys",
                    severity=Severity.WARN,
                    message=(
                        f"Episodes expose {len(name_sets)} different camera-key "
                        f"sets; downstream loaders expect a stable schema."
                    ),
                )
            )

    @staticmethod
    def _resolve_reader(path: str | Path):
        from forge.formats.registry import FormatRegistry

        fmt = FormatRegistry.detect_format(Path(path))
        return FormatRegistry.get_reader(fmt)
