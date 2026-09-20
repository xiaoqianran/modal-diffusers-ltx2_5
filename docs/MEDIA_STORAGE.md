# Media storage

LTX-2.5 separates the control plane from the media data plane.

```text
Control plane
Browser -> FastAPI -> Modal Dict / Modal RPC

Media data plane (Volume fallback)
Browser -> FastAPI proxy -> Modal Volume -> GPU worker

Media data plane (routed S3-compatible)
Browser <--------------------> primary store (R2)
                                 |
                                 | failover on transfer/write failure
                                 v
                              fallback (MinIO)

GPU input fast path: primary R2 -> CloudBucketMount -> job-local namespace
GPU fallback input:  MinIO -> S3 GET -> job-local namespace
GPU output: local scratch -> MediaStorage -> primary/fallback
```

Job records store stable object keys such as `outputs/<job>.mp4`. They never
store presigned URLs, because those URLs expire.

## Recommended hybrid layout

Keep model/runtime state close to the GPU and move only user media to object
storage:

```text
Modal Volume
├─ models
├─ kernel cache
└─ runtime state / LoRA metadata

S3-compatible object storage
├─ inputs
└─ outputs
```

This avoids routing large media through the local FastAPI process while keeping
model loading and CUDA/NATTEN caches on Modal-managed storage.

## Backends

### Modal Volume

```text
LTX25_MEDIA_BACKEND=volume
```

This preserves the zero-configuration development path. The browser sends a
raw binary PUT to the local router and the router writes the asset to
`ltx25-state`.

### S3-compatible storage

Set `LTX25_MEDIA_BACKEND=s3` for MinIO, Backblaze B2, Tigris, AWS S3, or
another compatible service. Provider-specific backend aliases are intentionally
not supported; the media layer depends only on the S3 API contract.

The browser uploads directly to object storage:

- files below the multipart threshold use one presigned PUT;
- larger files use parallel multipart PUTs;
- Retake / Extend use server-side `CopyObject`;
- `/outputs/<name>` returns a short-lived redirect to a signed GET/HEAD;
- the GPU worker reads input objects through Modal `CloudBucketMount`.

The worker does **not** mux MP4 directly on the bucket mount. MP4 finalization
and faststart require seek-like file operations, so the worker encodes to local
`/tmp/ltx25-outputs` and then uploads the finalized object through the S3 API.
`CloudBucketMount` is read-only and is used only for GPU access to source media.

## Generic S3 configuration

The local router needs S3-compatible credentials:

```text
LTX25_MEDIA_BACKEND=s3
LTX25_S3_BUCKET=ltx25-media
LTX25_S3_ENDPOINT_URL=https://s3.example.com
LTX25_S3_REGION=us-east-1
AWS_ACCESS_KEY_ID=<access key>
AWS_SECRET_ACCESS_KEY=<secret key>

LTX25_MODAL_MEDIA_SECRET=s3-media
LTX25_MEDIA_PRESIGN_SECONDS=3600
LTX25_MEDIA_MULTIPART_THRESHOLD_MB=96
LTX25_MEDIA_PART_SIZE_MB=16
```

The Modal worker needs the same credentials in a Modal Secret:

```bash
modal secret create s3-media \
  AWS_ACCESS_KEY_ID=<access key> \
  AWS_SECRET_ACCESS_KEY=<secret key>
```

The environment used for `modal deploy modal_app.py` must contain the media
backend, bucket, endpoint, region, and secret name. The local router and
deployed worker must select the same backend and bucket.

### Tigris

Tigris uses the same provider-neutral S3 path; no Tigris-specific application
code is required:

```text
LTX25_MEDIA_BACKEND=s3
LTX25_S3_BUCKET=ltx25-media
LTX25_S3_ENDPOINT_URL=https://fly.storage.tigris.dev
LTX25_S3_REGION=auto
LTX25_MODAL_MEDIA_SECRET=tigris-media
```

Keep the scoped `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY` out of git and
place the same pair in the local `.env` and the Modal `tigris-media` Secret.

### Cloudflare R2

Cloudflare R2 is also consumed through the provider-neutral S3 path:

```text
LTX25_MEDIA_BACKEND=s3
LTX25_S3_BUCKET=ltx25-media-r2
LTX25_S3_ENDPOINT_URL=https://<ACCOUNT_ID>.r2.cloudflarestorage.com
LTX25_S3_REGION=auto
LTX25_MODAL_MEDIA_SECRET=r2-media
```

Use an R2 S3 access key pair as `AWS_ACCESS_KEY_ID` / `AWS_SECRET_ACCESS_KEY`.
No `r2` application backend alias is required; switching providers is purely
configuration.

## Current SG-JP MinIO deployment

The current hybrid deployment uses:

```text
LTX25_MEDIA_BACKEND=s3
LTX25_S3_BUCKET=ltx25-media
LTX25_S3_ENDPOINT_URL=https://s3-sg-jp.202820.xyz
LTX25_S3_REGION=us-east-1
LTX25_MODAL_MEDIA_SECRET=s3-media
```

Long-lived credentials are intentionally kept out of git. The local copy lives
in the ignored `.env` file; the GPU worker receives them from the Modal
`s3-media` Secret.


## Routed primary/fallback storage

Production can configure multiple physical stores while keeping each adapter single-backend.
`media_store.py` owns one Volume/S3 backend; `media_storage.py` owns stable routing.

```text
MediaRef(store_id, key)
        |
        v
MediaStorage
   |          |
 primary   fallback
   |          |
  R2        MinIO
```

`store_id` is persisted with every uploaded asset and generated output. It is an opaque,
stable identifier such as `r2` or `minio`, not the mutable role name `primary`/`fallback`.
This means the deployment can swap roles later without invalidating historical metadata.

Current production policy:

```text
primary  = Cloudflare R2
fallback = SG-JP MinIO
```

Failover is deliberately **not replication**. A successful R2 write is not mirrored to MinIO.
If historical-object read HA is required later, add replication as a separate subsystem rather
than coupling it to the write-routing path.

### Upload invariant

One upload intent is pinned to exactly one store from prepare through complete. Browser PUTs
retry the selected provider first. If byte transfer still fails, the browser requests a fresh
fallback plan; the old multipart session is aborted best-effort and the file is retransmitted
from the beginning. Multipart ETags are never reused across stores.

### GPU input invariant

`runtime.py` only consumes ordinary filesystem paths. The Modal deployment shell constructs a
job-local input namespace under `/tmp/ltx25-jobs/<job>/inputs`: primary R2 objects may use a
read-only CloudBucketMount fast path, while fallback objects are materialized locally through
`MediaStorage.download_to()`. The generation core does not import storage or Modal APIs.

### Routed configuration

```text
LTX25_MEDIA_PRIMARY_ID=r2
LTX25_MEDIA_PRIMARY_BACKEND=s3
LTX25_MEDIA_PRIMARY_S3_BUCKET=ltx25-media-r2
LTX25_MEDIA_PRIMARY_S3_ENDPOINT_URL=https://<ACCOUNT_ID>.r2.cloudflarestorage.com
LTX25_MEDIA_PRIMARY_S3_REGION=auto
LTX25_MEDIA_PRIMARY_S3_ACCESS_KEY_ID=<key>
LTX25_MEDIA_PRIMARY_S3_SECRET_ACCESS_KEY=<secret>

LTX25_MEDIA_FALLBACK_ID=minio
LTX25_MEDIA_FALLBACK_BACKEND=s3
LTX25_MEDIA_FALLBACK_S3_BUCKET=ltx25-media
LTX25_MEDIA_FALLBACK_S3_ENDPOINT_URL=https://minio.example.com
LTX25_MEDIA_FALLBACK_S3_REGION=us-east-1
LTX25_MEDIA_FALLBACK_S3_ACCESS_KEY_ID=<key>
LTX25_MEDIA_FALLBACK_S3_SECRET_ACCESS_KEY=<secret>
```

The Modal worker secret also exposes the primary credentials as `AWS_ACCESS_KEY_ID` /
`AWS_SECRET_ACCESS_KEY` for CloudBucketMount, plus both prefixed credential pairs for
`MediaStorage`.

## Browser CORS

Direct browser PUTs require CORS on the object store. At minimum the frontend
origin must be allowed to use PUT and JavaScript must be able to read the
multipart response `ETag`.

Example policy shape:

```json
[
  {
    "AllowedOrigins": [
      "http://127.0.0.1:5187",
      "http://localhost:5187"
    ],
    "AllowedMethods": ["GET", "HEAD", "PUT"],
    "AllowedHeaders": ["Content-Type", "Range"],
    "ExposeHeaders": ["ETag", "Content-Length", "Content-Range", "Accept-Ranges"],
    "MaxAgeSeconds": 3600
  }
]
```

MinIO also supports its own API CORS configuration. Regardless of provider,
multipart completion requires `ETag` to be browser-readable.

## Upload protocol

```text
POST /api/assets/prepare
  -> asset_id + proxy/single/multipart upload plan

PUT bytes
  -> local proxy URL (Volume), or
  -> presigned S3 object/part URLs

POST /api/assets/{asset_id}/complete
  -> completes multipart when needed
  -> verifies final object size
  -> records the durable asset key

DELETE /api/assets/{asset_id}/upload
  -> best-effort abort after a failed multipart upload
```

Multipart uploads default to four concurrent browser PUTs.

## Output delivery

Application state continues to expose stable URLs:

```text
/outputs/<filename>
```

With Volume storage this is a local `FileResponse`. With S3-compatible
storage it is a 307 redirect to a freshly signed GET or HEAD URL.

## Metrics

`GET /api/health` includes:

```text
media.backend
media.direct_upload
media.direct_download
transfer_metrics.upload
transfer_metrics.download
transfer_metrics.copy
```

For direct uploads the browser reports `client_upload_seconds` at finalize
time. This is kept separately from server-side S3 finalize time because the
router is no longer in the media byte path.

## Switching an existing deployment

Do not switch a deployment with active jobs in flight. Existing media objects
in the old Modal Volume are not automatically copied to object storage.

A safe rollout is:

1. stop new jobs and let active jobs finish;
2. copy any historical inputs/outputs that must remain available;
3. configure S3 credentials and browser CORS;
4. create/update the Modal media secret;
5. deploy the worker with the S3 environment;
6. start the local router with the same environment;
7. verify `/api/health` reports `media.backend = s3`;
8. submit a small image job and then a multipart-sized upload.
