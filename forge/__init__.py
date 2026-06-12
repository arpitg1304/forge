"""Forge - The normalization layer for robotics data.

Forge converts between robotics episode formats (RLDS, LeRobot v2/v3, Zarr, Rosbag)
with minimal friction. It's the "on-ramp" to the VLA data platform — whatever format
you captured, Forge makes it usable by downstream tools.

Key Characteristics:
- I/O-bound, no GPU required
- Streaming/lazy architecture for memory efficiency
- LeRobot v3 as the platform's lingua franca

Example:
    >>> import forge
    >>> info = forge.inspect("./my_dataset")
    >>> print(info.cameras)
    >>> print(info.missing_required)

    >>> forge.convert("./rlds_data", "./lerobot_data", target_format="lerobot-v3")
"""

import os

# Suppress TensorFlow verbose logging before any TF import can happen
# This must be done before tensorflow is imported anywhere
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # FATAL only
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  # Suppress oneDNN messages
os.environ.setdefault(
    "CUDA_VISIBLE_DEVICES", os.environ.get("CUDA_VISIBLE_DEVICES", "")
)  # Keep existing if set
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GLOG_minloglevel", "2")  # Google logging: ERROR+

from forge.convert.converter import (
    ConversionConfig,
    ConversionResult,
    Converter,
    convert,
)
from forge.core.exceptions import (
    ConversionError,
    ForgeError,
    FormatDetectionError,
    InspectionError,
    MissingDependencyError,
    SchemaError,
    UnsupportedFormatError,
    ValidationError,
)
from forge.core.models import (
    CameraInfo,
    DatasetInfo,
    Dtype,
    Episode,
    FieldSchema,
    Frame,
    LazyArray,
    LazyImage,
)
from forge.filter.engine import FilterConfig, FilterEngine, FilterResult
from forge.formats.registry import FormatRegistry
from forge.inspect.inspector import InspectionOptions, Inspector
from forge.segment import SegmentAnalyzer, SegmentationReport, SegmentConfig
from forge.tokenize import (
    ActionTokenizer,
    ComparisonReport,
    TokenizerComparator,
    TokenizerRegistry,
    TokenizerStats,
    load_tokenizer,
)

__version__ = "0.1.0"

__all__ = [
    # Version
    "__version__",
    # Core models
    "CameraInfo",
    "DatasetInfo",
    "Dtype",
    "Episode",
    "FieldSchema",
    "Frame",
    "LazyArray",
    "LazyImage",
    # Exceptions
    "ConversionError",
    "ForgeError",
    "FormatDetectionError",
    "InspectionError",
    "MissingDependencyError",
    "SchemaError",
    "UnsupportedFormatError",
    "ValidationError",
    # Registry
    "FormatRegistry",
    # Inspector
    "Inspector",
    "InspectionOptions",
    # Converter
    "Converter",
    "ConversionConfig",
    "ConversionResult",
    # Filter
    "FilterConfig",
    "FilterEngine",
    "FilterResult",
    # Segment
    "SegmentAnalyzer",
    "SegmentConfig",
    "SegmentationReport",
    # Tokenize
    "ActionTokenizer",
    "TokenizerRegistry",
    "TokenizerComparator",
    "ComparisonReport",
    "TokenizerStats",
    "load_tokenizer",
    # Module-level functions
    "inspect",
    "convert",
]


def inspect(
    path: str,
    format: str | None = None,
    **options,
) -> DatasetInfo:
    """Inspect a dataset and return structured information.

    Args:
        path: Path to dataset.
        format: Format hint (auto-detected if None).
        **options: InspectionOptions fields.

    Returns:
        DatasetInfo with schema, stats, and conversion readiness.

    Example:
        >>> info = forge.inspect("./my_dataset")
        >>> print(info.cameras)
        >>> print(info.missing_required)

        >>> info = forge.inspect("./dataset", deep_scan=True, sample_episodes=10)
    """
    opts = InspectionOptions(**options)
    inspector = Inspector(opts)
    return inspector.inspect(path, format)
