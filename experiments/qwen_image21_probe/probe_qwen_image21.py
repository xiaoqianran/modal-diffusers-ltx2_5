from __future__ import annotations

import argparse
import gc
import json
import platform
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from PIL import Image

MODEL_ID = "Qwen/Qwen-Image-2.1"
MODEL_REVISION = "b3179ad355be050328e483a9dfdd9e60cd62adfa"


def _gib(value: int | float) -> float:
    return round(float(value) / 1024**3, 3)


def cuda_memory() -> dict[str, float | str]:
    free, total = torch.cuda.mem_get_info()
    return {
        "device": torch.cuda.get_device_name(0),
        "allocated_gib": _gib(torch.cuda.memory_allocated()),
        "reserved_gib": _gib(torch.cuda.memory_reserved()),
        "peak_allocated_gib": _gib(torch.cuda.max_memory_allocated()),
        "peak_reserved_gib": _gib(torch.cuda.max_memory_reserved()),
        "free_gib": _gib(free),
        "total_gib": _gib(total),
    }


def reset_peak() -> None:
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()


@dataclass
class Result:
    name: str
    elapsed_seconds: float
    width: int
    height: int
    condition_count: int
    memory: dict[str, Any]
    output: str


def load_pipeline(model_id: str = MODEL_ID, revision: str = MODEL_REVISION):
    from diffusers import QwenImage21Pipeline

    start = time.perf_counter()
    pipe = QwenImage21Pipeline.from_pretrained(
        model_id,
        revision=revision,
        dtype=torch.bfloat16,
    ).to("cuda")
    torch.cuda.synchronize()
    return pipe, round(time.perf_counter() - start, 3), cuda_memory()


def run_case(
    pipe,
    outdir: Path,
    *,
    name: str,
    prompt: str,
    width: int,
    height: int,
    steps: int = 40,
    seed: int = 42,
    images: list[Image.Image] | Image.Image | None = None,
) -> Result:
    reset_peak()
    generator = torch.Generator(device="cuda").manual_seed(seed)
    start = time.perf_counter()
    output = pipe(
        prompt=prompt,
        image=images,
        width=width,
        height=height,
        num_inference_steps=steps,
        true_cfg_scale=1.0,
        negative_prompt=None,
        use_kv_cache=True,
        generator=generator,
    ).images[0]
    torch.cuda.synchronize()
    elapsed = round(time.perf_counter() - start, 3)

    path = outdir / f"{name}.png"
    output.save(path)
    condition_count = len(images) if isinstance(images, list) else int(images is not None)
    return Result(
        name=name,
        elapsed_seconds=elapsed,
        width=width,
        height=height,
        condition_count=condition_count,
        memory=cuda_memory(),
        output=str(path),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-id", default=MODEL_ID)
    parser.add_argument("--revision", default=MODEL_REVISION)
    parser.add_argument("--outdir", default="experiments/qwen_image21_probe/runs/local")
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--sizes", default="1024x1024,1536x1024,2048x2048")
    parser.add_argument("--edit-images", default="")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    pipe, load_seconds, resident = load_pipeline(args.model_id, args.revision)

    report: dict[str, Any] = {
        "environment": {
            "python": platform.python_version(),
            "torch": torch.__version__,
            "model_id": args.model_id,
            "revision": args.revision,
        },
        "load_seconds": load_seconds,
        "resident_memory": resident,
        "cases": [],
    }

    prompt = (
        "Cinematic wide shot of a rain-soaked futuristic Tokyo alley at night, "
        "a lone detective under warm practical lighting, realistic materials, "
        "subtle fog, anamorphic composition, high detail."
    )
    for token in args.sizes.split(","):
        width, height = (int(x) for x in token.lower().split("x", 1))
        result = run_case(
            pipe,
            outdir,
            name=f"t2i_{width}x{height}",
            prompt=prompt,
            width=width,
            height=height,
            steps=args.steps,
            seed=args.seed,
        )
        report["cases"].append(asdict(result))
        (outdir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    image_paths = [Path(x.strip()) for x in args.edit_images.split(",") if x.strip()]
    images = [Image.open(path).convert("RGB") for path in image_paths]
    if images:
        for count in (1, 2, 4):
            if len(images) < count:
                continue
            result = run_case(
                pipe,
                outdir,
                name=f"edit_{count}img",
                prompt=(
                    "Preserve the main subject identity. Recompose the shot as a cinematic "
                    "snowy mountain sunrise scene while keeping coherent lighting and materials."
                ),
                width=1024,
                height=1024,
                steps=args.steps,
                seed=args.seed,
                images=images[0] if count == 1 else images[:count],
            )
            report["cases"].append(asdict(result))
            (outdir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")

    gc.collect()
    torch.cuda.empty_cache()
    report["final_resident_memory"] = cuda_memory()
    (outdir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
