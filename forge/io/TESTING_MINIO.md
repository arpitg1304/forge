# Testing cloud support against a local MinIO

[MinIO](https://min.io) is a self-hosted, S3-compatible object store. Running it
locally is the most faithful way to test Forge's `s3://` support end-to-end
(real `s3://` scheme, real `s3fs`, real video/parquet download) without an AWS
account or any cloud cost.

> For automated, zero-setup tests (fsspec `memory://` + a `moto` mock S3
> server) see [`tests/test_io_paths.py`](../../tests/test_io_paths.py). Use MinIO
> when you want to drive the actual `forge` CLI against a real server, or to
> exercise large real datasets (videos included).

All commands below assume a scratch directory you can delete afterwards:

```bash
export MINIO_DIR=/tmp/forge-minio      # anywhere you like
mkdir -p "$MINIO_DIR" && cd "$MINIO_DIR"
```

## 1. Install & start MinIO

Pick one option.

**a) Standalone binary (no Docker):**

```bash
# macOS arm64 (Apple Silicon). For other platforms swap the URL:
#   macOS Intel : https://dl.min.io/server/minio/release/darwin-amd64/minio
#   Linux amd64 : https://dl.min.io/server/minio/release/linux-amd64/minio
curl -fsSL -o minio https://dl.min.io/server/minio/release/darwin-arm64/minio
chmod +x minio

MINIO_ROOT_USER=minioadmin MINIO_ROOT_PASSWORD=minioadmin \
  ./minio server ./minio-data \
  --address 127.0.0.1:9000 --console-address 127.0.0.1:9001 &
echo $! > minio.pid            # save PID for teardown
```

**b) Homebrew (macOS):**

```bash
brew install minio/stable/minio
MINIO_ROOT_USER=minioadmin MINIO_ROOT_PASSWORD=minioadmin \
  minio server "$MINIO_DIR/minio-data" --address 127.0.0.1:9000 &
```

**c) Docker:**

```bash
docker run -d --name forge-minio -p 9000:9000 -p 9001:9001 \
  -e MINIO_ROOT_USER=minioadmin -e MINIO_ROOT_PASSWORD=minioadmin \
  minio/minio server /data --console-address ":9001"
```

Wait until it's healthy:

```bash
until curl -fsS http://127.0.0.1:9000/minio/health/live >/dev/null; do sleep 1; done
echo "MinIO ready — console http://127.0.0.1:9001 (minioadmin/minioadmin)"
```

## 2. Point Forge at MinIO

Forge uses the standard AWS credential chain, so a custom endpoint just needs
env vars — no Forge-specific config:

```bash
export AWS_ACCESS_KEY_ID=minioadmin
export AWS_SECRET_ACCESS_KEY=minioadmin
export AWS_DEFAULT_REGION=us-east-1
export AWS_ENDPOINT_URL=http://127.0.0.1:9000
export AWS_ENDPOINT_URL_S3=http://127.0.0.1:9000
```

Make sure the S3 backend is installed:

```bash
pip install "forge-robotics[s3]"     # brings in s3fs
```

## 3. Upload a dataset

This example uses `lerobot/droid_100` (a lerobot-v3 dataset, ~460 MB with
videos). Grab it from the Hub if you don't already have it:

```bash
python -c "from huggingface_hub import snapshot_download; \
print(snapshot_download('lerobot/droid_100', repo_type='dataset'))"
```

Upload the snapshot directory to a bucket with `s3fs` (already installed with
`[s3]`), resolving HF's symlinks as you go:

```bash
python - <<'PY'
from pathlib import Path
import s3fs
from huggingface_hub import snapshot_download

snap = Path(snapshot_download("lerobot/droid_100", repo_type="dataset"))

fs = s3fs.S3FileSystem(
    key="minioadmin", secret="minioadmin",
    client_kwargs={"endpoint_url": "http://127.0.0.1:9000"},
)
if not fs.exists("forge-datasets"):
    fs.mkdir("forge-datasets")

for p in snap.rglob("*"):
    if p.is_file():
        rel = p.relative_to(snap).as_posix()
        fs.put_file(str(p.resolve()), f"forge-datasets/droid_100/{rel}")

print("Uploaded. Bucket contents:")
for k in fs.find("forge-datasets/droid_100"):
    print("  ", k)
PY
```

Any local dataset works the same way — just point `snap` at its directory.

## 4. Run Forge against `s3://`

```bash
forge inspect s3://forge-datasets/droid_100
forge quality s3://forge-datasets/droid_100 --sample 5
forge convert s3://forge-datasets/droid_100 ./out --format lerobot-v3 --dry-run
```

Expected: each command prints `Fetching from cloud storage: …` / `Downloaded to:
/tmp/forge-cloud-…`, then behaves exactly as it would for a local path. The
temp copy is deleted when the command exits.

Writing **to** a cloud URI is intentionally rejected:

```bash
forge convert s3://forge-datasets/droid_100 s3://forge-datasets/out --format lerobot-v3
# Error: Writing outputs to a cloud URI is not supported yet: s3://forge-datasets/out
```

## 5. Troubleshooting

| Symptom | Fix |
|---|---|
| `MissingDependencyError … pip install forge-robotics[s3]` | Install the `[s3]` extra. |
| `NoCredentialsError` | Export the `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` above. |
| `EndpointConnectionError` / connection refused | MinIO isn't running or `AWS_ENDPOINT_URL` is wrong/missing. Re-check the health probe in step 1. |
| `NoSuchBucket` | Create the bucket (the upload script does this) or fix the bucket name. |
| Reads succeed but seem to re-download every run | Expected — remote datasets are copied to a fresh temp dir each run and cleaned up at exit. |

See [`README.md`](README.md) for the general (real S3/GCS) connectivity guide.

## 6. Teardown

```bash
kill "$(cat "$MINIO_DIR/minio.pid")"      # binary/brew;  or: docker rm -f forge-minio
rm -rf "$MINIO_DIR"                        # removes the server binary + all uploaded data
```
