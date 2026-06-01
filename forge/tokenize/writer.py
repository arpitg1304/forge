"""Write a tokenized LeRobot v3 dataset.

``tokenize_and_write`` resolves a source dataset, fits (or loads) an action
tokenizer on its action corpus, then streams every episode through the
LeRobot v3 writer with a per-frame ``action_tokens`` column. The fitted
tokenizer is saved to ``<output>/meta/action_tokenizer.json`` so inference code
can detokenize model outputs with :func:`forge.tokenize.load_tokenizer`.

v1 is sequential (mirrors the converter's sequential loop) and per-step only:
chunk-granularity tokenizers are rejected with a clear error.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from forge.core.exceptions import ConversionError
from forge.core.models import DatasetInfo, Episode, Frame
from forge.formats.lerobot_v3.writer import LeRobotV3Writer, LeRobotV3WriterConfig
from forge.tokenize.base import TokenizerError, load_tokenizer
from forge.tokenize.registry import TokenizerRegistry

if TYPE_CHECKING:
    from forge.tokenize.base import ActionTokenizer

TOKEN_FEATURE = "action_tokens"


@dataclass
class TokenizeWriteResult:
    """Summary of a tokenize-and-write run."""

    output_path: Path
    tokenizer_path: Path
    strategy: str
    num_episodes: int = 0
    num_frames: int = 0
    errors: list[str] = field(default_factory=list)


def tokenize_and_write(
    source: str | Path,
    output: str | Path,
    strategy: str,
    *,
    tokenizer_path: str | Path | None = None,
    num_bins: int = 256,
    keep_actions: bool = False,
    reader=None,
    fps: float = 30.0,
    robot_type: str = "unknown",
    progress_callback: Callable[[int, str], None] | None = None,
) -> TokenizeWriteResult:
    """Emit a LeRobot v3 dataset with an ``action_tokens`` column.

    Args:
        source: Source dataset (path / hf:// / registry id) or any value the
            supplied ``reader`` understands.
        output: Output dataset directory.
        strategy: Registered tokenizer strategy name.
        tokenizer_path: If given, load a pre-fitted tokenizer instead of fitting.
        num_bins: Vocabulary size when fitting a fresh tokenizer.
        keep_actions: Retain the original float ``action`` column alongside tokens.
        reader: Optional reader override (must expose ``read_episodes(path)``).
        fps: Frames per second for the output dataset.
        robot_type: Robot type identifier for the output dataset.
        progress_callback: Optional ``callback(episode_idx, episode_id)``.

    Returns:
        A :class:`TokenizeWriteResult`.
    """
    output = Path(output)
    reader = reader or _resolve_reader(source)

    # 1. Resolve the tokenizer (load pre-fitted, or fit on the action corpus).
    if tokenizer_path is not None:
        tokenizer = load_tokenizer(tokenizer_path)
    else:
        corpus = _build_corpus(reader.read_episodes(source))
        tokenizer = TokenizerRegistry.create(strategy, num_bins=num_bins).fit(corpus)

    if tokenizer.granularity != "per_step":
        raise TokenizerError(
            f"Tokenizer '{strategy}' has granularity '{tokenizer.granularity}'; "
            "the v1 dataset writer supports per_step tokenizers only. "
            "Use the library/compare API for chunk tokenizers."
        )

    # 2. Stream episodes through the writer, injecting per-frame token columns.
    writer = LeRobotV3Writer(
        LeRobotV3WriterConfig(
            fps=fps,
            robot_type=robot_type,
            tokenized_action_feature=TOKEN_FEATURE,
        )
    )

    result = TokenizeWriteResult(
        output_path=output,
        tokenizer_path=output / "meta" / "action_tokenizer.json",
        strategy=tokenizer.name,
    )

    episode_idx = 0
    for episode in reader.read_episodes(source):
        if progress_callback:
            progress_callback(episode_idx, episode.episode_id)
        wrapped = _tokenize_episode(episode, tokenizer, keep_actions)
        try:
            frames = wrapped.load_frames()
            writer.write_episode(wrapped, output, episode_index=episode_idx)
            result.num_episodes += 1
            result.num_frames += len(frames)
        except Exception as e:  # pragma: no cover - defensive
            result.errors.append(f"Episode {episode.episode_id}: {e}")
            raise ConversionError("source", "lerobot-v3", str(e))
        episode_idx += 1

    info = DatasetInfo(
        path=output,
        format="lerobot-v3",
        num_episodes=result.num_episodes,
        total_frames=result.num_frames,
        inferred_fps=fps,
        inferred_robot_type=robot_type,
    )
    writer.finalize(output, info)

    # 3. Save the fitted tokenizer alongside the dataset for inference decode.
    tokenizer.save(result.tokenizer_path)
    return result


def _tokenize_episode(
    episode: Episode, tokenizer: ActionTokenizer, keep_actions: bool
) -> Episode:
    """Return an Episode whose frames carry ``extras[action_tokens]``.

    The original action is optionally dropped so the output stores tokens only.
    """

    def loader():
        for frame in episode.frames():
            if frame.action is None:
                yield frame
                continue
            tokens = tokenizer.encode(np.asarray(frame.action, dtype=np.float64))
            new_extras = dict(frame.extras)
            new_extras[TOKEN_FEATURE] = np.asarray(tokens).astype(np.int64)
            yield Frame(
                index=frame.index,
                timestamp=frame.timestamp,
                images=frame.images,
                state=frame.state,
                action=frame.action if keep_actions else None,
                reward=frame.reward,
                is_terminal=frame.is_terminal,
                is_first=frame.is_first,
                is_last=frame.is_last,
                extras=new_extras,
            )

    return Episode(
        episode_id=episode.episode_id,
        metadata=episode.metadata,
        language_instruction=episode.language_instruction,
        success=episode.success,
        robot_type=episode.robot_type,
        cameras=episode.cameras,
        fps=episode.fps,
        _frame_loader=loader,
    )


def _build_corpus(episodes: Iterable[Episode]) -> NDArray:
    rows: list[NDArray] = []
    for ep in episodes:
        for frame in ep.frames():
            if frame.action is not None:
                rows.append(np.asarray(frame.action, dtype=np.float64))
    if not rows:
        raise ValueError("no actions found in source dataset")
    return np.vstack(rows)


def _resolve_reader(source: str | Path):
    from forge.formats.registry import FormatRegistry

    fmt = FormatRegistry.detect_format(Path(source))
    return FormatRegistry.get_reader(fmt)
