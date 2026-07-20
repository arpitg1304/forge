# `forge.io`

The single place Forge turns a user-supplied source string into something the
format readers can consume. It lets **every** command that accepts a dataset
path also accept cloud object-store URIs (`s3://`, `gs://`) alongside local
paths and `hf://` URLs.

## Why this module exists

Forge's format readers are built around `pathlib.Path` and local file access —
`h5py.File`, `av.open`, `pq.read_table`, `Path.glob`, `open()` — and format
detection probes the directory tree with local `Path` operations. Rather than
teach every reader to speak fsspec, all path resolution is funneled through one
utility: [`paths.py`](paths.py). Remote datasets are **downloaded to a
temporary directory** on first access and cleaned up automatically, so every
format behaves identically whether the data is local or in a bucket.

```
source string ──▶ localize() ──▶ local Path ──▶ existing readers (unchanged)
   s3://…                          (temp dir, auto-cleaned at exit)
   gs://…
   /local/path                     (returned as-is)
   hf://org/repo                    (handled by forge.hub, not here)
```

> **Design note:** most operations download the remote source in full before
> processing — this keeps readers that seek randomly (video keyframes, HDF5
> chunks, rosbag indexes) working without modification. The exception is
> **`forge inspect` on LeRobot-v3 / Zarr**, which reads metadata over the
> network via **range requests** ([`DataSource`](source.py)) — inspecting a
> multi-GB cloud dataset fetches only kilobytes. Streaming the ingest/scoring
> paths (parquet column reads) is the next step.

## Install

```bash
pip install "forge-robotics[s3]"     # Amazon S3  (s3://)  → pulls in s3fs
pip install "forge-robotics[gcs]"    # Google Cloud (gs://) → pulls in gcsfs
```

`fsspec` itself is a core dependency; only the provider backend is optional. If
you pass an `s3://` URI without `s3fs` installed, Forge fails with the exact
`pip install` command to run.

## Usage

### CLI

Any command that takes a dataset path works with a cloud URI:

```bash
forge inspect s3://my-bucket/datasets/run_0413
forge convert gs://lab-data/rosbags ./out --format lerobot-v3
forge quality s3://my-bucket/datasets/droid --report report.html
forge stats  gs://lab-data/pusht.zarr
```

Writing outputs to a cloud URI (`forge convert … s3://bucket/out`) is **not**
supported yet — commands fail fast with a clear message. Write locally and
upload afterwards.

### Python API

```python
from forge.io import localize, is_remote_uri, get_filesystem

# Resolve any source to a local Path (downloads if remote, passthrough if local)
local_path = localize("s3://my-bucket/datasets/run_0413")
# ... hand local_path to any forge reader ...

is_remote_uri("s3://bucket/key")   # True
is_remote_uri("/data/run")         # False
is_remote_uri("hf://org/repo")     # False  (handled by forge.hub)

# Drop down to the underlying fsspec filesystem if you need raw access
fs, path = get_filesystem("s3://my-bucket/datasets/run_0413")
fs.ls(path)
```

`localize()` registers each temp directory for removal at process exit; call
`forge.io.cleanup_localized()` to remove them eagerly (e.g. in a long-running
service that resolves many datasets).

## Authentication

Forge **never handles credentials itself** — it relies entirely on each
provider's default credential chain.

| Provider | How credentials are found (in order) |
|---|---|
| **S3** (`s3://`) | `AWS_ACCESS_KEY_ID`/`AWS_SECRET_ACCESS_KEY` env vars → `AWS_PROFILE` in `~/.aws/credentials` & `~/.aws/config` → EC2/ECS/EKS instance IAM role |
| **GCS** (`gs://`) | `GOOGLE_APPLICATION_CREDENTIALS` (service-account key) → `gcloud auth application-default login` → attached service account on GCP |

Docs: [AWS credentials](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html)
· [GCP Application Default Credentials](https://cloud.google.com/docs/authentication/application-default-credentials).

## Diagnosing cloud connectivity issues

Work top-to-bottom — the checks are ordered from "wrong install" to "wrong
permissions". The fastest way to isolate a problem is to reproduce it with the
underlying tool (`aws`/`gcloud` or `s3fs`/`gcsfs`) directly, since Forge just
delegates to them.

### 1. `MissingDependencyError: ... pip install forge-robotics[s3]`

The backend isn't installed. Run the exact command in the message
(`[s3]` for `s3://`, `[gcs]` for `gs://`).

### 2. Confirm the URI resolves at the fsspec layer

This bypasses Forge entirely and tests just the filesystem + credentials:

```python
from forge.io import get_filesystem
fs, path = get_filesystem("s3://my-bucket/datasets/run_0413")
print(fs.exists(path))     # False → wrong path or no read permission
print(fs.ls(path)[:5])     # lists objects if you can see the prefix
```

Or with the provider CLI (which uses the same credential chain):

```bash
aws s3 ls s3://my-bucket/datasets/run_0413/
gcloud storage ls gs://lab-data/rosbags/
```

If these fail, the problem is credentials/permissions/path — not Forge.

### 3. Common failures and fixes

| Symptom | Likely cause | Fix |
|---|---|---|
| `NoCredentialsError` / `Unable to locate credentials` | No credentials on the machine | `aws configure` / `export AWS_PROFILE=…` / `gcloud auth application-default login` |
| `AccessDenied` / `403 Forbidden` | Credentials work but lack `s3:GetObject`/`s3:ListBucket` (or `storage.objects.get/list`) on that prefix | Grant read on the bucket/prefix, or use a profile/role that has it |
| `FileNotFoundError: Remote source not found (or empty)` | Path typo, wrong bucket, or an empty prefix | Verify with `aws s3 ls` / `gcloud storage ls`; check for a trailing-slash or region mismatch |
| `NoSuchBucket` / `The specified bucket does not exist` | Wrong bucket name or wrong region | Check the name; set `AWS_DEFAULT_REGION` / `AWS_REGION` |
| `EndpointConnectionError` / connection timeout | No network, VPC egress blocked, or a custom endpoint needed | Check connectivity; for S3-compatible stores (MinIO, R2) set `AWS_ENDPOINT_URL` |
| `ExpiredToken` / `InvalidToken` | Temporary/STS credentials expired | Re-authenticate (`aws sso login`, refresh the assumed role) |
| Wrong/expired GCP creds despite `gcloud` working | `GOOGLE_APPLICATION_CREDENTIALS` points at a stale key that overrides ADC | Unset it to fall back to ADC, or point it at the right key |
| Download is slow / appears to hang | Whole dataset is being downloaded to a temp dir (expected) | Use a smaller sample where supported (`--sample`), or a machine closer to the bucket region |

### 4. S3-compatible stores (MinIO, Cloudflare R2, Ceph)

Point botocore/s3fs at the custom endpoint via the standard env var — no Forge
config needed:

```bash
export AWS_ENDPOINT_URL=https://<your-endpoint>
export AWS_ACCESS_KEY_ID=…  AWS_SECRET_ACCESS_KEY=…
forge inspect s3://my-bucket/my-dataset
```

### 5. Turn on backend logging

fsspec/botocore/gcsfs emit useful debug logs. Enable them and resolve a path
directly:

```python
import logging
logging.basicConfig(level=logging.DEBUG)   # botocore, s3fs, gcsfs all log here
from forge.io import localize
localize("s3://my-bucket/dataset")         # watch the request/response trace
```

For a quick sign of life you can also check where a resolved dataset landed —
Forge prints `Downloaded to: /tmp/forge-cloud-…` before it starts reading.

## Testing

Cloud access is tested without any real provider — see
[`tests/test_io_paths.py`](../../tests/test_io_paths.py):

- fsspec's in-process `memory://` filesystem (needs only core `fsspec`; always
  runs) for scheme detection, download-to-temp, cleanup, and end-to-end
  lerobot-v3 / zarr resolution.
- a `moto` mock S3 server for the real `s3://` code path (skipped if
  `moto[server]` / `s3fs` aren't installed).

```bash
pip install -e ".[dev]"
pytest tests/test_io_paths.py -v
```

To drive the real `forge` CLI against a real S3-compatible server (no AWS
account needed), see [TESTING_MINIO.md](TESTING_MINIO.md) — a step-by-step guide
to running a local MinIO and testing cloud support against it.

## Public API

| Name | Purpose |
|---|---|
| `localize(uri, *, console=None)` | Resolve `uri` to a local `Path`, downloading if remote. Local paths & `hf://` returned as-is. |
| `is_remote_uri(uri)` | `True` for `s3://`/`gs://`/other fsspec schemes; `False` for local paths and `hf://`. |
| `get_scheme(uri)` | The lowercased URL scheme (`""` for local paths). |
| `get_filesystem(uri)` | `(fsspec_filesystem, path)` for raw access; raises `MissingDependencyError` with the pip hint if the backend is missing. |
| `cleanup_localized()` | Remove all temp dirs created for downloaded datasets (also runs at exit). |
| `RemoteWriteNotSupportedError` | Raised when a cloud URI is given as an output destination. |
