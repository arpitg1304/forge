"""Download MCAP test fixtures from the Rerun project.

Sourced from:
  - https://github.com/rerun-io/rerun/tree/main/tests/assets/mcap
  - https://github.com/rerun-io/rerun/tree/main/crates/store/re_importer/tests/assets

These files are gitignored under sample_data/mcap/ — too large to vendor in
the repo. Run this script after a fresh clone if you want to exercise the
MCAP integration tests against real recordings.

Usage:
    python scripts/download_mcap_fixtures.py
    python scripts/download_mcap_fixtures.py --force  # re-download even if present
"""

from __future__ import annotations

import argparse
import sys
import urllib.request
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEST = REPO_ROOT / "sample_data" / "mcap"

# Pinned to a specific rerun commit so upstream changes to the recordings
# can't silently break our test assertions (which encode file-specific facts
# like message counts). To bump: update the SHA, re-run with --force, and
# refresh tests/fixtures/mcap/INVENTORY.md plus any affected assertions.
RERUN_SHA = "e8386c5f5434c4c0d3f1c7dac7dbfe1a9b1a7bb4"

FIXTURES: dict[str, str] = {
    "r2b_galileo.mcap": (
        f"https://github.com/rerun-io/rerun/raw/{RERUN_SHA}/"
        "tests/assets/mcap/r2b_galileo.mcap"
    ),
    "trossen_transfer_cube.mcap": (
        f"https://github.com/rerun-io/rerun/raw/{RERUN_SHA}/"
        "tests/assets/mcap/trossen_transfer_cube.mcap"
    ),
    "supported_ros2_messages.mcap": (
        f"https://github.com/rerun-io/rerun/raw/{RERUN_SHA}/"
        "crates/store/re_importer/tests/assets/supported_ros2_messages.mcap"
    ),
}


def _download(url: str, target: Path) -> int:
    with urllib.request.urlopen(url) as resp:
        data = resp.read()
    target.write_bytes(data)
    return len(data)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force", action="store_true", help="re-download even if files already exist"
    )
    args = parser.parse_args()

    DEST.mkdir(parents=True, exist_ok=True)

    for name, url in FIXTURES.items():
        target = DEST / name
        if target.exists() and not args.force:
            print(f"  [skip] {name} already exists ({target.stat().st_size:,} bytes)")
            continue
        print(f"  [get ] {name} <- {url}")
        try:
            size = _download(url, target)
            print(f"         wrote {size:,} bytes")
        except Exception as e:
            print(f"         FAILED: {e}", file=sys.stderr)
            return 1

    print(f"\nFixtures available at {DEST.relative_to(REPO_ROOT)}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
