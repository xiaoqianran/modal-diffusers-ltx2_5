#!/usr/bin/env python3
"""Zero-dependency mock of the ltx25 FastAPI backend, for frontend development.

Implements the same HTTP contract as ltx25/api.py so the Vite dev server can
proxy to it, without requiring Modal credentials, a GPU, fastapi, or torch.

    python3 tools/mock_backend.py --port 8000

Then point the frontend at it (vite.config.js already defaults to :8000):

    cd frontend && npm run dev

Differences from the real backend are deliberate and limited to execution:
jobs are simulated by a background thread that advances through the same
stages (text_encode -> denoise -> upsample -> refine -> decode -> encode)
and renders a real, playable mp4/png via ffmpeg. Everything the frontend
observes - status transitions, progress values, queue ordering, error
strings - follows the real contract.

Useful flags:
    --delay-scale X        multiply every stage duration (default 0.05; use 1.0
                           to replay RTX PRO 6000 timings, 0 for instant jobs)
    --fail-rate 0.0-1.0    fraction of jobs that fail at a random stage
    --stall-rate 0.0-1.0   fraction of "long" jobs that stall near 90%
    --mock-loras N         how many fake LoRA entries /api/loras returns
"""
from __future__ import annotations

import argparse
import json
import math
import mimetypes
import os
import random
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

# ---------------------------------------------------------------- config

STILL_IMAGE_MODES = {"t2i", "refine_image", "ref2i"}
STILL_PREFIX = {"t2i": "t2i", "refine_image": "refine", "ref2i": "ref2i"}
ACTIVE_STATUSES = {"queued", "running"}
JOB_ID_RE = re.compile(r"^[0-9a-f]{32}$")
ASSET_ID_RE = re.compile(r"^[0-9a-f]{32}$")

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
VIDEO_SUFFIXES = {".mp4", ".mov", ".webm", ".mkv", ".gif"}
AUDIO_SUFFIXES = {".wav", ".mp3", ".m4a", ".flac", ".ogg", ".aac"}

# Stage walls for a non-refine 121-frame video job, in seconds. Mirrors the
# proportions measured on the RTX PRO 6000 (docs/ARCHITECTURE.md timings).
# Every value is multiplied by --delay-scale, which defaults to 0.05 so layout
# iteration is not blocked by waiting: a full video lands in ~0.6s. Pass
# --delay-scale 1 to replay the real timings when testing progress rendering.
STAGES = [
    ("text_encode", 1.2),
    ("denoise", 5.1),
    ("upsample", 0.5),
    ("refine", 3.0),
    ("decode", 2.0),
    ("encode", 1.2),
]

# Floor per stage so progress transitions stay observable even at scale 0.
MIN_STAGE_SECONDS = 0.02

OPTIONS = argparse.Namespace(
    host="127.0.0.1",
    port=8000,
    delay_scale=0.05,
    fail_rate=0.0,
    stall_rate=0.0,
    mock_loras=4,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    print(f"[mock] {message}", flush=True)


# ---------------------------------------------------------------- state

STATE_LOCK = threading.RLock()
JOBS: dict[str, dict] = {}
SESSIONS: set[int] = set()
WORK_QUEUE: list[str] = []
CANCELLED: set[str] = set()
RUNNING_JOB: str | None = None
KEEP_GPU_WARM = True
CACHE_ROOT = Path(tempfile.gettempdir()) / "ltx25-mock"
UPLOAD_DIR = CACHE_ROOT / "inputs"
OUTPUT_DIR = CACHE_ROOT / "outputs"
VIDEO_DIR = CACHE_ROOT / "videos"


# ---------------------------------------------------------------- ffmpeg

def render_video(path: Path, width: int, height: int, frames: int, fps: float,
                 seed: int, prompt: str) -> bool:
    """Render a real, playable H.264 clip with an AAC tone.

    The visual is a moving gradient plus a frame counter so scrubbing proves
    the file is a genuine video rather than a still image.
    """
    # ffmpeg requires even dimensions for yuv420p.
    width -= width % 2
    height -= height % 2
    duration = max(frames - 1, 1) / max(fps, 1.0)
    hue = (seed % 360) / 360.0

    # Build the video filter chain. gradients is a *source* filter, so it is
    # supplied as the input; the remaining filters must be chained onto it in a
    # single comma-separated list (a stray comma makes ffmpeg treat the chain as
    # two outputs and fail with "expected exactly 1 input and 1 output").
    filters = [f"hue=h={hue * 360:.1f}:s=1.3"]
    if shutil.which("fc-match"):
        filters.append(
            f"drawtext=text='%{{n}}/{frames}':fontsize={max(14, height // 12)}:"
            f"fontcolor=white@0.85:x=(w-text_w)/2:y=h-th-{max(8, height // 20)}"
        )
    filters.append("format=yuv420p")
    vf = ",".join(filters)

    command = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", (
            f"gradients=s={width}x{height}:d={duration:.3f}"
            f":c0=0x10110f:c1=0x1f2a14:speed=0.15"
        ),
        "-f", "lavfi", "-i", f"sine=frequency={180 + seed % 400}:duration={duration:.3f}",
        "-vf", vf,
        "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
        "-pix_fmt", "yuv420p", "-r", f"{fps:g}",
        "-c:a", "aac", "-b:a", "64k", "-shortest",
        "-movflags", "+faststart",
        str(path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=180)
    except Exception as exc:
        log(f"ffmpeg render failed: {exc!r}")
        return False
    if result.returncode != 0:
        log(f"ffmpeg render failed: {result.stderr.strip()[-400:]}")
        return False
    return True


def render_image(path: Path, width: int, height: int, seed: int) -> bool:
    """Render a PNG via ffmpeg (avoids a Pillow dependency)."""
    hue = (seed % 360)
    command = [
        "ffmpeg", "-y", "-v", "error",
        "-f", "lavfi", "-i", f"gradients=s={width}x{height}:d=1:c0=0x10110f:c1=0x6b7d3a",
        "-vf", f"hue=h={hue}:s=1.2,format=rgb24",
        "-frames:v", "1",
        str(path),
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=60)
    except Exception as exc:
        log(f"ffmpeg image render failed: {exc!r}")
        return False
    return result.returncode == 0


# ---------------------------------------------------------------- jobs

def public_job(record: dict) -> dict:
    return {k: v for k, v in record.items() if k != "call_id"}


def save(record: dict) -> None:
    record["updated_at"] = utc_now()


def probe_video(path: Path) -> tuple[int, int, float]:
    """Return (width, height, fps) of a video via ffprobe, or a sane default."""
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", "-select_streams", "v:0",
             "-show_entries", "stream=width,height,r_frame_rate",
             "-of", "json", str(path)],
            capture_output=True, text=True, timeout=30,
        )
        stream = json.loads(result.stdout)["streams"][0]
        rate = stream.get("r_frame_rate", "24/1")
        num, _, den = rate.partition("/")
        fps = float(num) / float(den or 1)
        return int(stream["width"]), int(stream["height"]), fps or 24.0
    except Exception:
        return 0, 0, 24.0


def stage_plan(request: dict) -> list[tuple[str, float]]:
    """Choose a stage plan for this request (still images take the t2i recipe).

    Durations are the real measured walls scaled by --delay-scale, then floored
    at MIN_STAGE_SECONDS so a scale of 0 still produces distinct progress
    transitions instead of a single jump to 100%.
    """
    stages = []
    for name, seconds in STAGES:
        # Skip stages the request disables, so the progress bar length matches
        # the work the real backend would actually perform.
        if name in {"upsample", "refine"} and not request.get("upscale") and not request.get("temporal_upscale"):
            continue
        if name in {"upsample", "refine"} and request.get("mode") in {"ref2i", "iclora", "retake", "extend"}:
            continue
        # Still-image modes are much shorter than video.
        if request.get("mode") in STILL_IMAGE_MODES:
            seconds *= 0.45
        stages.append((name, max(seconds * OPTIONS.delay_scale, MIN_STAGE_SECONDS)))
    return stages


def run_job(job_id: str) -> None:
    """Advance one job through its stages, emitting real progress updates."""
    global RUNNING_JOB
    record = JOBS.get(job_id)
    if record is None or job_id in CANCELLED:
        return
    request = record["request"]

    record["status"] = "running"
    record["progress"] = 0.0
    save(record)

    stages = stage_plan(request)
    total = sum(seconds for _, seconds in stages) or 1.0
    is_video = request.get("mode") not in STILL_IMAGE_MODES
    prefix = STILL_PREFIX.get(request.get("mode"), "")
    filename = f"{prefix}_{job_id}.png" if not is_video else f"{job_id}.mp4"
    target = OUTPUT_DIR / filename

    # Inject a failure at a random point for a fraction of jobs.
    fail_at = None
    if OPTIONS.fail_rate > 0 and random.random() < OPTIONS.fail_rate:
        fail_at = random.choice(["text_encode", "denoise", "decode", "encode"])
        log(f"job {job_id[:8]}: will fail at {fail_at}")

    # Long jobs sometimes stall near the end, reproducing the motion-stall QC
    # issue documented in README.md.
    stall = (
        OPTIONS.stall_rate > 0
        and random.random() < OPTIONS.stall_rate
        and int(request.get("num_frames") or 121) >= 200
    )

    elapsed_base = 0.0
    for name, seconds in stages:
        if job_id in CANCELLED:
            return
        if fail_at == name:
            time.sleep(seconds * 0.5)
            record = JOBS[job_id]
            record["status"] = "failed"
            record["error"] = (
                f"RuntimeError: simulated {name} failure (MockOutOfMemory "
                f"while allocating attention workspace for {name})"
            )
            record["generation_seconds"] = round(elapsed_base, 2)
            record["peak_vram_gb"] = None
            save(record)
            log(f"job {job_id[:8]}: failed at {name}")
            return

        steps = max(4, int(seconds * 8))
        for step in range(steps):
            if job_id in CANCELLED:
                return
            time.sleep(seconds / steps)
            done = elapsed_base + seconds * (step + 1) / steps
            record = JOBS.get(job_id)
            if record is None:
                return
            record["progress"] = min(done / total, 0.99)
            save(record)

        if name == "decode" and stall:
            log(f"job {job_id[:8]}: simulating decode stall")
            for _ in range(40):
                if job_id in CANCELLED:
                    return
                time.sleep(0.35 * max(OPTIONS.delay_scale, 0.05))
        elapsed_base += seconds

    if job_id in CANCELLED:
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    ok = False
    if is_video:
        width = int(request.get("width") or 768)
        height = int(request.get("height") or 512)
        if request.get("upscale"):
            width, height = width * 2, height * 2
        frames = int(request.get("num_frames") or 121)
        if request.get("temporal_upscale"):
            frames = (frames - 1) * 2 + 1
        fps = float(request.get("fps") or 24.0) * (2 if request.get("temporal_upscale") else 1)
        ok = render_video(target, width, height, frames, fps,
                          int(request.get("seed") or 0), request.get("prompt", ""))
    else:
        width = int(request.get("width") or 512) * 2
        height = int(request.get("height") or 512) * 2
        ok = render_image(target, width, height, int(request.get("seed") or 0))

    record = JOBS.get(job_id)
    if record is None:
        return
    if not ok:
        record["status"] = "failed"
        record["error"] = "RuntimeError: mock ffmpeg render failed"
        save(record)
        return

    record["status"] = "completed"
    record["progress"] = 1.0
    record["error"] = None
    record["generation_seconds"] = round(elapsed_base, 2)
    record["peak_vram_gb"] = round(
        (17.9 if request.get("mode") in STILL_IMAGE_MODES else 40.9)
        * random.uniform(0.85, 1.12), 1
    )
    if is_video:
        record["video_url"] = f"/outputs/{target.name}"
        record["image_url"] = None
    else:
        record["image_url"] = f"/outputs/{target.name}"
        record["video_url"] = None
    save(record)
    log(f"job {job_id[:8]}: completed -> {target.name} ({elapsed_base:.1f}s simulated)")

    # If a retake/extend job produced an edited video, log the merge step so the
    # console shows the same sequence as the real backend.
    if request.get("mode") in {"retake", "extend"}:
        log(f"job {job_id[:8]}: merged {request['mode']} output into {target.name}")


def worker_loop() -> None:
    """Single GPU container => strictly serial execution."""
    global RUNNING_JOB
    while True:
        job_id = None
        with STATE_LOCK:
            if WORK_QUEUE:
                job_id = WORK_QUEUE.pop(0)
                RUNNING_JOB = job_id
        if job_id is None:
            time.sleep(0.15)
            continue
        try:
            run_job(job_id)
        except Exception as exc:  # never let the worker thread die
            log(f"worker error on {job_id[:8]}: {exc!r}")
            record = JOBS.get(job_id)
            if record is not None and record.get("status") in ACTIVE_STATUSES:
                record["status"] = "failed"
                record["error"] = f"{type(exc).__name__}: {exc}"
                save(record)
        finally:
            with STATE_LOCK:
                RUNNING_JOB = None


# ---------------------------------------------------------------- routing

def create_job(request: dict) -> dict:
    job_id = uuid.uuid4().hex
    session_number = request.get("session_number") or int(time.time() * 1000)
    request["session_number"] = session_number
    now = utc_now()
    record = {
        "id": job_id,
        "session_number": session_number,
        "status": "queued",
        "progress": 0.0,
        "error": None,
        "video_url": None,
        "image_url": None,
        "request": request,
        "created_at": now,
        "updated_at": now,
        "generation_seconds": None,
        "peak_vram_gb": None,
        "call_id": None,
    }
    with STATE_LOCK:
        JOBS[job_id] = record
        WORK_QUEUE.append(job_id)
        SESSIONS.add(session_number)
    log(f"job {job_id[:8]}: queued ({request.get('mode')}, "
        f"{request.get('width')}x{request.get('height')}, "
        f"{request.get('num_frames')}f, queue depth {len(WORK_QUEUE)})")
    return public_job(record)


def list_jobs(session_number: int | None, limit: int) -> list[dict]:
    with STATE_LOCK:
        records = list(JOBS.values())
    if session_number is not None:
        records = [r for r in records if r.get("session_number") == session_number]
    records.sort(key=lambda r: r.get("created_at", ""), reverse=True)
    return [public_job(r) for r in records[: max(1, min(limit, 200))]]


def interrupt(job_id: str | None) -> dict:
    with STATE_LOCK:
        target = JOBS.get(job_id) if job_id else None
        if target is None and job_id is None:
            active = [r for r in JOBS.values() if r.get("status") in ACTIVE_STATUSES]
            running = [r for r in active if r.get("status") == "running"]
            candidates = running or [r for r in active if r.get("status") == "queued"]
            if candidates:
                target = min(candidates, key=lambda r: r.get("created_at", ""))
        if target is None or target.get("status") not in ACTIVE_STATUSES:
            return {"interrupted": False, "current_job_id": None, "requested_job_id": job_id}
        target_id = target["id"]
        CANCELLED.add(target_id)
        if target_id in WORK_QUEUE:
            WORK_QUEUE.remove(target_id)
        target["status"] = "interrupted"
        target["error"] = "Interrupted by user"
        save(target)
    log(f"job {target_id[:8]}: interrupted")
    return {"interrupted": True, "current_job_id": target_id, "requested_job_id": job_id}


LORA_NAMES = [
    "ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors",
    "ltx-2.5-22b-ic-lora-dubit.safetensors",
    "ltx-2.5-22b-lora-cinematic-filmgrain.safetensors",
    "ltx-2.5-22b-lora-anime-style.safetensors",
    "ltx-2.5-22b-ic-lora-hdr-scene.safetensors",
    "ltx-2.5-22b-lora-drone-footage.safetensors",
]


def list_loras() -> list[dict]:
    items = []
    for name in LORA_NAMES[: OPTIONS.mock_loras]:
        lowered = name.lower()
        items.append({
            "id": name,
            "name": Path(name).stem,
            "size": random.randint(40, 900) * 1024 * 1024,
            "kind": "iclora" if ("ic-lora" in lowered or "iclora" in lowered) else "standard",
            "model_version": "LTX-2.5",
            "reference_downscale_factor": 1.0 if "ic-lora" in lowered else None,
            "generic_iclora_compatible": "pixel-spatial-upscaler" in lowered or "dubit" in lowered,
        })
    return sorted(items, key=lambda item: item["name"].lower())


# ---------------------------------------------------------------- handler

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "ltx25-mock/1.0"

    def log_message(self, fmt, *args):  # quieter access log
        return

    # -- helpers -------------------------------------------------------
    def send_json(self, payload, status=200):
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,DELETE,HEAD,OPTIONS")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(body)

    def send_error_json(self, status, detail):
        self.send_json({"detail": detail}, status=status)

    def read_body(self) -> bytes:
        length = int(self.headers.get("Content-Length") or 0)
        return self.rfile.read(length) if length else b""

    def read_json(self) -> dict:
        raw = self.read_body()
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ValueError(f"invalid JSON body: {exc}") from exc

    # -- verbs ---------------------------------------------------------
    def do_OPTIONS(self):
        self.send_response(204)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Headers", "*")
        self.send_header("Access-Control-Allow-Methods", "GET,POST,DELETE,HEAD,OPTIONS")
        self.send_header("Content-Length", "0")
        self.end_headers()

    def do_HEAD(self):
        self.do_GET()

    def do_GET(self):
        parsed = urlparse(self.path)
        path, query = parsed.path, parse_qs(parsed.query)

        if path == "/api/health":
            with STATE_LOCK:
                keep_gpu_warm = KEEP_GPU_WARM
            return self.send_json({
                "status": "ok",
                "model": "Lightricks/LTX-2.5-Diffusers",
                "worker": "mock-worker",
                "transport": "local-http-mock",
                "gpu_attached_to_web": False,
                "transformer_precision": "nvfp4",
                "keep_gpu_warm": keep_gpu_warm,
                "gpu_idle_seconds": 600,
                "warmup": (
                    {
                        "state": "ready",
                        "gpu": "MOCK RTX PRO 6000",
                        "allocated_gb": 19.4,
                        "engines": ["ltx", "qwen"],
                    }
                    if keep_gpu_warm else {"state": "disabled"}
                ),
            })

        if path == "/api/loras":
            return self.send_json(list_loras())

        if path == "/api/jobs":
            session = query.get("session_number")
            limit = int((query.get("limit") or ["50"])[0])
            return self.send_json(list_jobs(int(session[0]) if session else None, limit))

        match = re.fullmatch(r"/api/jobs/([0-9a-fA-F]+)", path)
        if match:
            record = JOBS.get(match.group(1))
            if record is None:
                return self.send_error_json(404, "Job not found")
            return self.send_json(public_job(record))

        if path.startswith("/outputs/"):
            return self.serve_output(path.rsplit("/", 1)[-1])

        return self.send_error_json(404, "Not Found")

    def do_POST(self):
        global KEEP_GPU_WARM
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/api/admin/warm":
            with STATE_LOCK:
                KEEP_GPU_WARM = True
            return self.send_json({"status": "warming"})

        if path == "/api/sessions":
            number = int(time.time() * 1000)
            with STATE_LOCK:
                SESSIONS.add(number)
            log(f"session {number} created")
            return self.send_json({"session_number": number}, status=201)

        if path == "/api/jobs":
            try:
                request = self.read_json()
            except ValueError as exc:
                return self.send_error_json(400, str(exc))
            error = validate(request)
            if error:
                log(f"rejected job: {error}")
                return self.send_error_json(422, error)
            return self.send_json(create_job(request), status=202)

        if path == "/api/interrupt":
            try:
                body = self.read_json()
            except ValueError:
                body = {}
            return self.send_json(interrupt(body.get("job_id")))

        if path == "/api/assets":
            return self.handle_upload()

        if path == "/api/jobs/concat":
            return self.handle_concat()

        if path == "/api/prompts/enhance":
            return self.send_error_json(503, "Prompt enhancer is not configured in the local router")

        if path == "/api/admin/unload":
            with STATE_LOCK:
                KEEP_GPU_WARM = False
            return self.send_json({"result": "mock worker stays resident"})

        return self.send_error_json(404, "Not Found")

    def do_DELETE(self):
        match = re.fullmatch(r"/api/jobs/([0-9a-fA-F]+)", urlparse(self.path).path)
        if not match:
            return self.send_error_json(404, "Not Found")
        job_id = match.group(1)
        record = JOBS.get(job_id)
        if record is None:
            return self.send_error_json(404, "Job not found")
        if record.get("status") in ACTIVE_STATUSES:
            return self.send_error_json(409, "A queued or running job cannot be deleted")
        with STATE_LOCK:
            JOBS.pop(job_id, None)
            CANCELLED.add(job_id)
        for candidate in OUTPUT_DIR.glob(f"*{job_id}*"):
            candidate.unlink(missing_ok=True)
        log(f"job {job_id[:8]}: deleted")
        self.send_response(204)
        self.send_header("Content-Length", "0")
        self.end_headers()

    # -- uploads -------------------------------------------------------
    def handle_upload(self):
        content_type = self.headers.get("Content-Type", "")
        if "multipart/form-data" not in content_type:
            return self.send_error_json(400, "Expected multipart/form-data")

        try:
            filename, data = parse_multipart(self.read_body(), content_type)
        except ValueError as exc:
            return self.send_error_json(400, str(exc))

        if not filename:
            return self.send_error_json(400, "Uploaded file is empty")

        suffix = Path(filename).suffix.lower()
        if suffix in IMAGE_SUFFIXES:
            kind = "image"
        elif suffix in VIDEO_SUFFIXES:
            kind = "video"
        elif suffix in AUDIO_SUFFIXES:
            kind = "audio"
        else:
            return self.send_error_json(415, "Unsupported media type")

        if not data:
            return self.send_error_json(400, "Uploaded file is empty")
        if len(data) > 1024 * 1024 * 1024:
            return self.send_error_json(413, "Upload exceeds 1 GiB")

        # Validate with ffprobe the way the real API does, so bad files fail
        # here instead of silently in the worker.
        if kind in {"video", "audio"} and suffix != ".gif":
            UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
            probe_path = UPLOAD_DIR / f"probe-{uuid.uuid4().hex}{suffix}"
            probe_path.write_bytes(data)
            try:
                selector = "v:0" if kind == "video" else "a:0"
                probe = subprocess.run(
                    ["ffprobe", "-v", "error", "-select_streams", selector,
                     "-show_entries", "stream=codec_name", "-of", "csv=p=0", str(probe_path)],
                    capture_output=True, text=True, timeout=30,
                )
                if probe.returncode != 0 or not probe.stdout.strip():
                    return self.send_error_json(400, f"Uploaded {kind} could not be decoded")
            finally:
                probe_path.unlink(missing_ok=True)

        asset_id = uuid.uuid4().hex
        UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
        (UPLOAD_DIR / f"{asset_id}{suffix}").write_bytes(data)
        log(f"asset {asset_id[:8]}: {kind} {filename} ({len(data)/1024/1024:.2f} MB)")
        return self.send_json({
            "id": asset_id, "kind": kind, "filename": filename, "size": len(data),
        }, status=201)

    # -- outputs -------------------------------------------------------
    def serve_output(self, filename: str):
        if Path(filename).name != filename:
            return self.send_error_json(400, "Invalid output filename")
        path = OUTPUT_DIR / filename
        if not path.is_file():
            return self.send_error_json(404, "Output not found")
        media_type = mimetypes.guess_type(filename)[0] or "application/octet-stream"
        data = path.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", media_type)
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Accept-Ranges", "bytes")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        if self.command != "HEAD":
            self.wfile.write(data)

    def handle_concat(self):
        try:
            body = self.read_json()
        except ValueError as exc:
            return self.send_error_json(400, str(exc))
        job_ids = body.get("job_ids") or []
        if len(job_ids) < 2:
            return self.send_error_json(422, "At least two job ids are required")
        sources = []
        for job_id in job_ids:
            record = JOBS.get(job_id)
            if record is None or record.get("status") != "completed":
                return self.send_error_json(404, f"Completed video not found: {job_id}")
            candidate = OUTPUT_DIR / f"{job_id}.mp4"
            if not candidate.is_file():
                return self.send_error_json(404, f"Completed video not found: {job_id}")
            sources.append(candidate)

        concat_id = uuid.uuid4().hex
        target = OUTPUT_DIR / f"combined-{concat_id}.mp4"
        listing = Path(tempfile.mkstemp(suffix=".txt", text=True)[1])
        listing.write_text("".join(f"file '{s.as_posix()}'\n" for s in sources), encoding="utf-8")
        try:
            result = subprocess.run(
                ["ffmpeg", "-y", "-v", "error", "-f", "concat", "-safe", "0",
                 "-i", str(listing), "-c", "copy", "-movflags", "+faststart", str(target)],
                capture_output=True, text=True, timeout=180,
            )
        finally:
            listing.unlink(missing_ok=True)
        if result.returncode != 0:
            target.unlink(missing_ok=True)
            return self.send_error_json(400, f"Video concatenation failed: {result.stderr[-300:]}")
        log(f"concat: {len(sources)} jobs -> {target.name}")
        return self.send_json({"video_url": f"/outputs/{target.name}", "filename": target.name})


# ---------------------------------------------------------------- validation

def validate(request: dict) -> str | None:
    """Subset of GenerateRequest validation - enough to surface real UI bugs."""
    mode = request.get("mode") or "t2av"
    conditions = request.get("conditions") or []
    prompt = (request.get("prompt") or "").strip()
    if not prompt:
        return "prompt must not be empty"

    width, height = int(request.get("width") or 0), int(request.get("height") or 0)
    if width % 32 or height % 32:
        return "width and height must be multiples of 32"
    if not (256 <= width <= 1920 and 256 <= height <= 1920):
        return "width and height must be between 256 and 1920"

    num_frames = request.get("num_frames")
    if num_frames is not None and (int(num_frames) - 1) % 8 != 0:
        return "num_frames must be 8n+1 (for example 9, 121, 241 or 481)"

    for condition in conditions:
        if not ASSET_ID_RE.match(str(condition.get("asset_id", ""))):
            return "condition asset_id must be a 32-character lowercase hex string"
        if condition.get("kind") not in {"image", "video"}:
            return "condition kind must be image or video"

    if request.get("upscale") and width * height > 960 * 544:
        return "2x upscale base resolution cannot exceed 960x544 pixels"

    upscale = bool(request.get("upscale"))
    if request.get("upscale_method") == "pixel" and not upscale:
        return "pixel upscale method requires upscale=true"

    if mode == "t2av" and conditions:
        return "t2av mode does not accept visual conditions"
    if mode == "i2v":
        if len(conditions) != 1 or conditions[0].get("kind") != "image" or conditions[0].get("index") != 0:
            return "i2v mode requires one image condition at index 0"
    if mode == "flf2v":
        pairs = {(c.get("kind"), c.get("index")) for c in conditions}
        if len(conditions) != 2 or pairs != {("image", 0), ("image", -1)}:
            return "flf2v mode requires first and last image conditions at indices 0 and -1"
    if mode == "condition" and not conditions:
        return "condition mode requires at least one image or video condition"
    if mode == "iclora":
        if len(conditions) != 1 or conditions[0].get("index") != 1:
            return "iclora mode requires one image or video reference at index 1"
        if len(request.get("loras") or []) != 1:
            return "iclora mode requires exactly one IC-LoRA"
    if mode == "retake":
        if len(conditions) != 1 or conditions[0].get("kind") != "video":
            return "retake mode requires one source video"
        start, end = request.get("retake_start"), request.get("retake_end")
        if start is None or end is None or float(start) >= float(end):
            return "retake_start must be less than retake_end"
    if mode == "extend":
        videos = [c for c in conditions if c.get("kind") == "video"]
        if len(videos) != 1:
            return "extend mode requires exactly one source video condition"
    if mode == "a2v" and not request.get("audio_asset_id"):
        return "a2v mode requires an audio asset"
    if mode in {"t2i", "refine_image", "ref2i"} and not upscale and mode != "ref2i":
        return f"{mode} mode always upsamples"
    return None


def parse_multipart(body: bytes, content_type: str) -> tuple[str, bytes]:
    """Minimal multipart/form-data parser for a single file field."""
    match = re.search(r"boundary=([^;]+)", content_type)
    if not match:
        raise ValueError("missing multipart boundary")
    boundary = match.group(1).strip().strip('"').encode()
    marker = b"--" + boundary

    for part in body.split(marker):
        if b"Content-Disposition" not in part:
            continue
        header_blob, _, content = part.partition(b"\r\n\r\n")
        if not content:
            continue
        filename = "upload.bin"
        disposition = header_blob.decode("utf-8", "replace")
        name_match = re.search(r'filename="([^"]*)"', disposition)
        if name_match:
            filename = name_match.group(1)
        return filename, content.rstrip(b"\r\n").rstrip(b"--").rstrip(b"\r\n")
    raise ValueError("no file part found in multipart body")


# ---------------------------------------------------------------- entrypoint

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--delay-scale", type=float, default=0.05,
                        help="multiply real stage durations (default 0.05 => ~0.6s per "
                             "video; 1.0 replays measured RTX PRO 6000 timings; 0 for instant)")
    parser.add_argument("--fail-rate", type=float, default=0.0)
    parser.add_argument("--stall-rate", type=float, default=0.0)
    parser.add_argument("--mock-loras", type=int, default=4)
    parser.add_argument("--clear", action="store_true", help="wipe cached outputs on start")
    args = parser.parse_args()

    if not shutil.which("ffmpeg"):
        print("ffmpeg is required to render mock outputs", file=sys.stderr)
        return 1

    OPTIONS.host, OPTIONS.port = args.host, args.port
    OPTIONS.delay_scale = max(args.delay_scale, 0.0)
    OPTIONS.fail_rate, OPTIONS.stall_rate = args.fail_rate, args.stall_rate
    OPTIONS.mock_loras = args.mock_loras

    if args.clear:
        shutil.rmtree(CACHE_ROOT, ignore_errors=True)
    for directory in (UPLOAD_DIR, OUTPUT_DIR, VIDEO_DIR):
        directory.mkdir(parents=True, exist_ok=True)

    threading.Thread(target=worker_loop, daemon=True, name="mock-worker").start()

    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"[mock] LTX-2.5 mock backend on http://{args.host}:{args.port}", flush=True)
    print(f"[mock] delay_scale={OPTIONS.delay_scale} fail_rate={args.fail_rate} "
          f"stall_rate={args.stall_rate} loras={args.mock_loras}", flush=True)
    print(f"[mock] artifacts: {CACHE_ROOT}", flush=True)
    print("[mock] endpoints: /api/health /api/sessions /api/jobs /api/assets "
          "/api/loras /api/interrupt /outputs/{file}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[mock] shutting down", flush=True)
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
