# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- **Cloud storage support (`s3://`, `gs://`).** Every command that accepts a
  dataset path now also accepts Amazon S3 and Google Cloud Storage URIs, in
  addition to local paths and `hf://` URLs. For example:

  ```bash
  forge inspect s3://my-bucket/datasets/run_0413
  forge convert gs://lab-data/rosbags ./out --format lerobot-v3
  forge quality s3://my-bucket/datasets/droid --report report.html
  ```

  - Backed by [fsspec](https://filesystem-spec.readthedocs.io/) (now a core
    dependency), with `s3fs` / `gcsfs` as optional extras:
    `pip install "forge-robotics[s3]"` or `[gcs]`. Passing an `s3://` URI
    without `s3fs` installed fails with the exact `pip install` command to run.
  - All filesystem access is routed through a single utility,
    [`forge.io.paths`](forge/io/paths.py). Remote datasets are downloaded to a
    temporary directory on first access and cleaned up automatically at process
    exit, so every format (including video, HDF5, and rosbag, which need random
    file access) behaves identically to local paths.
  - Authentication uses each provider's default credential chain (AWS env
    vars / profiles / IAM roles; GCP Application Default Credentials). Forge
    never handles credentials itself.

### Changed

- `fsspec` is now a core dependency.

### Known limitations

- **Writing** outputs directly to cloud URIs (e.g. `forge convert … s3://bucket/out`)
  is not supported yet; commands fail fast with a clear message. Write to a
  local directory and upload it afterwards.
- Remote datasets are downloaded in full before processing. True range-read
  streaming (reading parquet/zarr remotely without a full download) is a
  planned follow-up; the fsspec plumbing in `forge.io.paths` is in place for it.
