# Media storage

LTX-2.5 separates the control plane from the media data plane.

```text
Control plane
Browser -> FastAPI -> Modal Dict / Modal RPC

Media data plane (default)
Browser -> FastAPI proxy -> Modal Volume -> GPU worker

Media data plane (R2/S3)
Browser <--------------------> R2/S3
                                ^
                                |
                       CloudBucketMount
                                |
                           GPU worker
```

Job records store stable object keys such as `outputs/<job>.mp4`. They never
store presigned URLs, because those URLs expire.

## Backends

### Volume (default)

```bash
LTX25_MEDIA_BACKEND=volume
```

This preserves the zero-configuration development path. The browser sends a
raw binary PUT to the local router and the router writes the asset to
`ltx25-state`.

### Cloudflare R2

The browser uploads directly to R2:

- files below the multipart threshold use one presigned PUT;
- larger files use parallel S3 multipart PUT;
- Retake / Extend use server-side `CopyObject`;
- `/outputs/<name>` returns a short-lived redirect to a signed R2 GET/HEAD;
- the GPU worker reads input objects through Modal `CloudBucketMount`.

The worker does **not** mux MP4 directly on the bucket mount. MP4 finalization
and faststart require seek-like file operations, while S3 mounts are optimized
for sequential object writes. The worker therefore encodes to local
`/tmp/ltx25-outputs` and then copies the finalized file sequentially to
`/media/outputs`.

## R2 configuration

Create an R2 bucket, for example `ltx25-media`, and create an R2 API token
with object read/write permission for that bucket.

The local router needs S3-compatible credentials:

```text
LTX25_MEDIA_BACKEND=r2
LTX25_R2_BUCKET=ltx25-media
LTX25_R2_ENDPOINT_URL=https://<ACCOUNT_ID>.r2.cloudflarestorage.com
LTX25_R2_ACCESS_KEY_ID=<R2 access key id>
LTX25_R2_SECRET_ACCESS_KEY=<R2 secret access key>

LTX25_MEDIA_PRESIGN_SECONDS=3600
LTX25_MEDIA_MULTIPART_THRESHOLD_MB=96
LTX25_MEDIA_PART_SIZE_MB=16
```

The Modal worker needs the same R2 credentials in a Modal Secret using the
AWS-compatible key names expected by `CloudBucketMount`:

```bash
modal secret create r2-media \
  AWS_ACCESS_KEY_ID=<R2 access key id> \
  AWS_SECRET_ACCESS_KEY=<R2 secret access key>
```

Then set:

```text
LTX25_MODAL_MEDIA_SECRET=r2-media
```

The environment used for `modal deploy modal_app.py` must also contain
`LTX25_MEDIA_BACKEND`, the bucket name, and endpoint URL. The local router
and deployed worker must select the same media backend and bucket.

## R2 CORS

Browser-to-R2 PUTs require bucket CORS. A development configuration for the
default frontend origins is:

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

Add the real production frontend origin before deploying publicly. Multipart
completion requires the browser to be able to read each part response's
`ETag`, so `ETag` must remain exposed.

## Upload protocol

The browser uses a two-phase control protocol around direct object transfer:

```text
POST /api/assets/prepare
  -> asset_id + proxy/single/multipart upload plan

PUT bytes
  -> local proxy URL (Volume), or
  -> presigned R2 object/part URLs

POST /api/assets/{asset_id}/complete
  -> verifies final object size
  -> records the durable asset key

DELETE /api/assets/{asset_id}/upload
  -> best-effort abort after a failed upload
```

Multipart uploads default to four concurrent browser PUTs. The finalization
request returns part ETags to the router, which completes the S3 multipart
upload and checks the final object size.

## Output delivery

Application state continues to expose stable URLs:

```text
/outputs/<filename>
```

With Volume storage this is a local `FileResponse`. With R2/S3 it is a 307
redirect to a freshly signed GET or HEAD URL. This keeps expiring credentials
out of job history while allowing the browser to talk directly to object
storage for the actual media bytes.

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

Do not switch a deployment with active jobs in flight. Existing objects in the
old Modal Volume are not automatically copied into R2. Historical job metadata
is retained, but media objects must be migrated if old outputs must remain
available after the backend switch.

A safe rollout is:

1. stop submitting new jobs;
2. wait for active jobs to finish;
3. copy any required `inputs/` and `outputs/` objects to the R2 bucket;
4. configure R2 CORS and the Modal media secret;
5. deploy the worker with the R2 environment;
6. start the local router with the same R2 environment;
7. verify `/api/health` reports `media.backend = r2`;
8. submit a small image job, then a multipart-sized video upload.
