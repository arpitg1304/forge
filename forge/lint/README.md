# Forge Lint

`forge lint` checks a dataset against Hugging Face's published [LeRobot
recording guidelines](https://huggingface.co/blog/lerobot-datasets) and flags
the hygiene defects maintainers document as common on the Hub — *before* you
spend GPU-hours training on a broken dataset.

Where `forge quality` scores the *content* of trajectories (smoothness, dead
actions, chatter…), `forge lint` checks *hygiene*: is the dataset annotated,
named, and shaped the way downstream tooling and VLA training expect?

## Usage

```bash
forge lint ./my_dataset
forge lint hf://lerobot/pusht --export lint.json
forge lint ./my_dataset --strict        # fail on warnings too, not just errors
```

Exit code is non-zero when any ERROR is found (or any WARN under `--strict`),
so it drops straight into CI.

## What it checks

Checks run against the reader's inspected `DatasetInfo` (cheap — no video
decode, no full episode scan), so they are correct even for formats that leave
per-episode fields empty.

| Code | Severity | Meaning |
|---|---|---|
| `dataset.empty` | error | No episodes. |
| `task.missing` | warn | No language/task instructions at all. |
| `task.partial_coverage` | warn | Only some episodes are annotated. |
| `task.generic` | warn | Placeholder task (`"task desc"`, `"Hold"`, …). |
| `task.too_short` | warn | Task below the 25-char guideline. |
| `camera.none` | warn | No cameras declared. |
| `camera.ambiguous_name` | warn | Camera name doesn't say where it is (`image`, `laptop`, …). |
| `camera.low_resolution` | info | Below 640×480. |
| `camera.too_few_views` | info | Fewer than two views. |
| `action.missing` | warn | No action field — can't train a policy. |

Thresholds live in `LintConfig` and are all overridable.

## Python API

```python
from forge.lint import DatasetLinter

report = DatasetLinter().lint_dataset("./my_dataset")
print(report.passed, len(report.errors), len(report.warnings))
for issue in report.issues:
    print(issue.severity.value, issue.code, issue.message)
```

`DatasetLinter.lint_episodes(episodes)` is also available for readers that
populate per-episode `Episode.cameras` / `language_instruction`, or for
in-memory episodes; it adds per-episode broken-length and cross-episode
dimension-consistency checks that the metadata path can't see.

## Scope (v1)

- Metadata-level checks via `inspect` (the CLI default).
- The per-episode path (`lint_episodes`) covers broken-episode and
  dimension-consistency checks but requires a reader that fills `Episode`
  fields — not all do.
- Not yet: full-scan per-episode frame-count auditing for formats whose readers
  don't expose per-episode counts in metadata.
