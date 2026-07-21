# Adding Datasets to the Forge Registry

## Steps

1. Edit `forge/registry/datasets.json`
2. Add a new entry under the `"datasets"` key using the template below
3. Run `forge registry validate` to check your entry
4. Open a PR — maintainers will verify paths and set `forge_verified: true`

## Template

```json
"your_dataset_id": {
  "id": "your_dataset_id",
  "name": "Human Readable Name",
  "description": "1-2 sentence description of the dataset.",
  "paper_url": "https://arxiv.org/abs/XXXX.XXXXX",
  "license": "MIT",
  "format": "rlds",
  "embodiment": ["franka"],
  "task_types": ["pick_place", "push"],
  "scale": {
    "episodes": 1000,
    "hours": 5.0,
    "approximate": true
  },
  "sources": [
    {
      "type": "hf_hub",
      "uri": "org/dataset_name",
      "split": null,
      "notes": null
    }
  ],
  "demo_suitable": false,
  "demo_episodes": null,
  "demo_source_index": null,
  "tags": ["manipulation", "real_world"],
  "forge_verified": false,
  "added_at": "2026-03-08",
  "notes": null
}
```

## Field Reference

| Field | Required | Description |
|-------|----------|-------------|
| `id` | Yes | Snake_case identifier (must match the JSON key) |
| `name` | Yes | Human-readable name |
| `description` | Yes | 1-2 sentence description |
| `format` | Yes | Native format: `rlds`, `lerobot`, `lerobot-v2`, `lerobot-v3`, `hdf5`, `zarr`, `mcap`, `rosbag`, `other` |
| `embodiment` | Yes | List of robot types (e.g., `["franka"]`) |
| `sources` | Yes | At least one download source (see below) |
| `paper_url` | Recommended | ArXiv or project page URL |
| `license` | Recommended | License string (e.g., `MIT`, `CC-BY-4.0`, `Apache-2.0`) |
| `task_types` | Optional | List of task categories |
| `scale` | Optional | Episode count, hours, and whether approximate |
| `demo_suitable` | Optional | `true` if a small subset (<=100 episodes) is available |
| `demo_episodes` | Optional | Exact episode count of the demo subset |
| `demo_source_index` | Optional | Index into `sources[]` pointing to the demo subset |
| `tags` | Optional | Searchable labels from the controlled vocabulary |
| `forge_verified` | No | Set to `false` — maintainers will verify |
| `notes` | Optional | Caveats, access requirements, etc. |

## Source Types

| Type | URI Format | Example |
|------|-----------|---------|
| `hf_hub` | HuggingFace repo ID | `lerobot/pusht` |
| `gcs` | GCS bucket path | `gs://gresearch/robotics/bridge` |
| `http` | Direct download URL | `https://example.com/dataset.zip` |
| `rsync` | Rsync path | `rsync://server/path` |

## Valid Tags

Use from this controlled vocabulary:

`manipulation`, `bi_manual`, `mobile_manipulation`, `humanoid`, `language_conditioned`, `contact_rich`, `simulation`, `real_world`, `multi_task`, `single_task`, `large_scale`

**Access-restriction tags** (surfaced as a dedicated "Access" filter and a badge
in the registry HTML view — add when a dataset is not freely/openly downloadable):

- `registration_required` — download is gated behind sign-up / an access agreement
  (e.g. a HuggingFace gated repo or a platform login).
- `non_commercial` — license forbids commercial use (e.g. CC-BY-NC-SA-4.0).

Always also record the specifics in `notes`.

## Validation

```bash
# Check your entry
forge registry validate

# Also check that source URIs are reachable
forge registry validate --probe
```
