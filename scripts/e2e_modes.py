from __future__ import annotations

import argparse
import json
import subprocess
import time
from pathlib import Path

import httpx
from PIL import Image


BASE_URL = "http://127.0.0.1:48125"
FIXTURE_DIR = Path("api-e2e-assets/multi-mode")
ASSET_MAP = FIXTURE_DIR / "assets.json"
RESULT_DIR = Path("api-e2e-results/multi-mode")
NEGATIVE = "blurry, jittery, distorted anatomy, duplicated objects, unreadable text"


def upload(client: httpx.Client, path: Path) -> dict:
    with path.open("rb") as handle:
        response = client.post("/api/assets", files={"file": (path.name, handle)})
    response.raise_for_status()
    return response.json()


def prepare() -> None:
    fixtures = {
        "first": FIXTURE_DIR / "first.png",
        "last": FIXTURE_DIR / "last.png",
        "source": FIXTURE_DIR / "source.mp4",
        "audio": FIXTURE_DIR / "audio.wav",
    }
    missing = [str(path) for path in fixtures.values() if not path.is_file()]
    if missing:
        raise SystemExit(f"missing fixtures: {missing}")

    with httpx.Client(base_url=BASE_URL, timeout=120.0) as client:
        health = client.get("/api/health")
        health.raise_for_status()
        assets = {name: upload(client, path) for name, path in fixtures.items()}

    ASSET_MAP.write_text(json.dumps(assets, indent=2), encoding="utf-8")
    print(json.dumps(assets, indent=2), flush=True)


def cases(assets: dict) -> list[dict]:
    first = assets["first"]["id"]
    last = assets["last"]["id"]
    source = assets["source"]["id"]
    audio = assets["audio"]["id"]
    common = {
        "negative_prompt": NEGATIVE,
        "guidance_scale": 1.0,
        "steps": 4,
        "seed": 2200,
    }
    return [
        {
            "name": "i2v_diffusion_1216x832",
            "expected": {"kind": "video", "width": 1216, "height": 832},
            "request": {
                **common,
                "seed": 2201,
                "mode": "i2v",
                "prompt": (
                    "A woman under a transparent umbrella walks forward on a rainy neon city street. "
                    "Preserve the person and street layout from the first frame. Smooth cinematic motion, "
                    "natural rain, realistic reflections, subtle handheld tracking. Audio: rain, footsteps, "
                    "distant traffic, no music."
                ),
                "width": 608,
                "height": 416,
                "num_frames": 121,
                "fps": 24,
                "upscale": True,
                "upscale_method": "latent",
                "decoder": "diffusion",
                "conditions": [{"asset_id": first, "kind": "image", "index": 0, "strength": 1.0}],
            },
        },
        {
            "name": "flf2v",
            "expected": {"kind": "video", "width": 384, "height": 288},
            "request": {
                **common,
                "seed": 2202,
                "mode": "flf2v",
                "prompt": (
                    "Rainy Tokyo street at night. Move naturally from the supplied first frame to the "
                    "supplied last frame with coherent pedestrian motion and stable architecture. "
                    "Audio: steady rain, footsteps, distant traffic."
                ),
                "width": 384,
                "height": 288,
                "num_frames": 81,
                "fps": 24,
                "upscale": False,
                "decoder": "vae",
                "conditions": [
                    {"asset_id": first, "kind": "image", "index": 0, "strength": 1.0},
                    {"asset_id": last, "kind": "image", "index": -1, "strength": 1.0},
                ],
            },
        },
        {
            "name": "a2v",
            "expected": {"kind": "video", "width": 384, "height": 288},
            "request": {
                **common,
                "seed": 2203,
                "mode": "a2v",
                "prompt": (
                    "A cinematic rainy city street whose visible motion follows the supplied environmental "
                    "audio. Wet pavement, umbrella pedestrian, realistic traffic reflections, stable camera."
                ),
                "width": 384,
                "height": 288,
                "num_frames": 81,
                "fps": 24,
                "upscale": False,
                "decoder": "vae",
                "audio_asset_id": audio,
                "audio_start": 0.0,
                "audio_duration": 3.0,
            },
        },
        {
            "name": "retake",
            "expected": {"kind": "video", "width": 384, "height": 288},
            "request": {
                **common,
                "seed": 2204,
                "mode": "retake",
                "prompt": (
                    "Keep the original rainy street before and after the selected interval. Inside the "
                    "retaken interval, make the umbrella pedestrian turn slightly toward the shop lights "
                    "while preserving scene continuity and realistic rain."
                ),
                "width": 384,
                "height": 288,
                "num_frames": 81,
                "fps": 24,
                "upscale": False,
                "decoder": "vae",
                "retake_start": 0.75,
                "retake_end": 2.25,
                "regenerate_video": True,
                "regenerate_audio": True,
                "conditions": [{"asset_id": source, "kind": "video", "index": 0, "strength": 1.0}],
            },
        },
        {
            "name": "extend",
            "expected": {"kind": "video", "width": 384, "height": 288},
            "request": {
                **common,
                "seed": 2205,
                "mode": "extend",
                "prompt": (
                    "Continue the rainy Tokyo street naturally beyond the source clip. The pedestrian keeps "
                    "walking forward, neon reflections continue consistently, camera motion remains smooth. "
                    "Audio continues with rain, footsteps and distant traffic."
                ),
                "width": 384,
                "height": 288,
                "num_frames": 81,
                "fps": 24,
                "upscale": False,
                "decoder": "vae",
                "extend_direction": "end",
                "extend_seconds": 1.0,
                "extend_context_seconds": 1.0,
                "conditions": [{"asset_id": source, "kind": "video", "index": 0, "strength": 1.0}],
            },
        },
        {
            "name": "t2i",
            "expected": {"kind": "image", "width": 768, "height": 576},
            "request": {
                **common,
                "seed": 2206,
                "mode": "t2i",
                "prompt": (
                    "A cinematic rainy Tokyo side street at night, one transparent umbrella beside a warm "
                    "ramen shop, wet asphalt reflecting cyan and amber neon, realistic photography, crisp "
                    "fine detail, shallow depth of field."
                ),
                "width": 384,
                "height": 288,
                "fps": 24,
                "decoder": "diffusion",
                "upscale": True,
                "upscale_method": "latent",
            },
        },
    ]


def probe_video(path: Path) -> dict:
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "stream=codec_type,codec_name,width,height,r_frame_rate:format=duration,size",
        "-of", "json", str(path),
    ]
    return json.loads(subprocess.check_output(cmd, text=True, encoding="utf-8"))


def validate_video(path: Path, expected: dict) -> tuple[dict, list[str]]:
    probe = probe_video(path)
    streams = probe.get("streams", [])
    videos = [item for item in streams if item.get("codec_type") == "video"]
    audios = [item for item in streams if item.get("codec_type") == "audio"]
    errors: list[str] = []
    if not videos:
        errors.append("missing video stream")
    else:
        video = videos[0]
        if (video.get("width"), video.get("height")) != (expected["width"], expected["height"]):
            errors.append(f"resolution={video.get('width')}x{video.get('height')}")
    if not audios:
        errors.append("missing audio stream")
    if path.stat().st_size < 10_000:
        errors.append(f"file too small: {path.stat().st_size}")
    return probe, errors


def validate_image(path: Path, expected: dict) -> tuple[dict, list[str]]:
    errors: list[str] = []
    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        info = {"format": image.format, "width": image.width, "height": image.height}
    if (info["width"], info["height"]) != (expected["width"], expected["height"]):
        errors.append(f"resolution={info['width']}x{info['height']}")
    return info, errors


def run() -> None:
    if not ASSET_MAP.is_file():
        raise SystemExit(f"run prepare first: {ASSET_MAP}")
    assets = json.loads(ASSET_MAP.read_text(encoding="utf-8"))
    RESULT_DIR.mkdir(parents=True, exist_ok=True)

    plan = cases(assets)
    summary = {"started_epoch": time.time(), "cases": []}
    failures = 0

    with httpx.Client(base_url=BASE_URL, timeout=180.0) as client:
        health = client.get("/api/health")
        health.raise_for_status()
        summary["health_before"] = health.json()

        for case in plan:
            response = client.post("/api/jobs", json=case["request"])
            response.raise_for_status()
            job = response.json()
            item = {
                "name": case["name"],
                "expected": case["expected"],
                "job_id": job["id"],
                "created_epoch": time.time(),
                "status": job["status"],
            }
            summary["cases"].append(item)
            print(f"SUBMIT {case['name']} {job['id']} {job['status']}", flush=True)

        pending = {item["job_id"]: item for item in summary["cases"]}
        deadline = time.time() + 30 * 60
        last_status: dict[str, str] = {}
        while pending and time.time() < deadline:
            for job_id in list(pending):
                item = pending[job_id]
                response = client.get(f"/api/jobs/{job_id}")
                response.raise_for_status()
                job = response.json()
                status = job["status"]
                if last_status.get(job_id) != status:
                    print(
                        f"STATUS {item['name']} {status} progress={job.get('progress', 0):.2f}",
                        flush=True,
                    )
                    last_status[job_id] = status
                item.update(
                    status=status,
                    progress=job.get("progress"),
                    generation_seconds=job.get("generation_seconds"),
                    peak_vram_gb=job.get("peak_vram_gb"),
                    video_url=job.get("video_url"),
                    image_url=job.get("image_url"),
                    error=job.get("error"),
                )
                if status in {"completed", "failed", "interrupted"}:
                    item["terminal_epoch"] = time.time()
                    item["wall_seconds"] = item["terminal_epoch"] - item["created_epoch"]
                    pending.pop(job_id)
            if pending:
                time.sleep(2)

        for item in pending.values():
            item["status"] = "timeout"
            item["error"] = "polling timeout"

        for item in summary["cases"]:
            if item["status"] != "completed":
                failures += 1
                print(f"FAIL {item['name']} {item['status']} {item.get('error')}", flush=True)
                continue

            expected = item["expected"]
            url = item["image_url"] if expected["kind"] == "image" else item["video_url"]
            suffix = ".png" if expected["kind"] == "image" else ".mp4"
            target = RESULT_DIR / f"{item['name']}-{item['job_id']}{suffix}"
            response = client.get(url)
            response.raise_for_status()
            target.write_bytes(response.content)

            if expected["kind"] == "image":
                probe, errors = validate_image(target, expected)
            else:
                probe, errors = validate_video(target, expected)
            item["output"] = str(target)
            item["file_size_bytes"] = target.stat().st_size
            item["probe"] = probe
            item["validation_errors"] = errors
            if errors:
                failures += 1
                print(f"INVALID {item['name']} {errors}", flush=True)
            else:
                print(
                    f"VALID {item['name']} gen={item.get('generation_seconds')}s "
                    f"vram={item.get('peak_vram_gb')}GiB",
                    flush=True,
                )

        try:
            response = client.post("/api/admin/unload")
            response.raise_for_status()
            summary["unload"] = response.json()
        except Exception as exc:
            summary["unload_error"] = f"{type(exc).__name__}: {exc}"

    summary["finished_epoch"] = time.time()
    summary["failure_count"] = failures
    output = Path("api-e2e-results/multi-mode-summary.json")
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"SUMMARY {output} failures={failures}", flush=True)
    raise SystemExit(1 if failures else 0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["prepare", "run"])
    args = parser.parse_args()
    if args.command == "prepare":
        prepare()
    else:
        run()


if __name__ == "__main__":
    main()
