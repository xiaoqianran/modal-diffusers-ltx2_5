from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path

import modal

APP_NAME = "qwen-image21-probe"
MODEL_ID = "Qwen/Qwen-Image-2.1"
MODEL_REVISION = "b3179ad355be050328e483a9dfdd9e60cd62adfa"
DIFFUSERS_COMMIT = "80c7ed262aeffbeb43ef13ae04baeb9b84515a69"
HF_SECRET_NAME = os.environ.get("LTX25_MODAL_HF_SECRET", "huggingface")
CACHE_VOLUME_NAME = os.environ.get("QWEN_IMAGE21_CACHE_VOLUME", "qwen-image21-cache")

app = modal.App(APP_NAME)
cache_volume = modal.Volume.from_name(CACHE_VOLUME_NAME, create_if_missing=True)
hf_secret = modal.Secret.from_name(HF_SECRET_NAME)

image = (
    modal.Image.debian_slim(python_version="3.12")
    .run_commands(
        "apt-get update && "
        "apt-get install -y --no-install-recommends git && "
        "rm -rf /var/lib/apt/lists/*"
    )
    .pip_install(
        "torch==2.13.0+cu130",
        "torchvision==0.28.0+cu130",
        index_url="https://download.pytorch.org/whl/cu130",
    )
    .pip_install(
        f"git+https://github.com/huggingface/diffusers.git@{DIFFUSERS_COMMIT}",
        "transformers==5.17.0",
        "accelerate==1.15.0",
        "safetensors==0.8.0",
        "huggingface-hub==1.32.0",
        "Pillow>=10",
    )
)


def _gib(value: int | float) -> float:
    return round(float(value) / 1024**3, 3)


@app.cls(
    image=image,
    gpu="RTX-PRO-6000",
    memory=65536,
    timeout=30 * 60,
    startup_timeout=30 * 60,
    max_containers=1,
    scaledown_window=2,
    volumes={"/cache": cache_volume},
    secrets=[hf_secret],
    env={
        "HF_HOME": "/cache/huggingface",
        "HF_XET_HIGH_PERFORMANCE": "1",
    },
)
@modal.concurrent(max_inputs=1)
class QwenImage21Probe:
    @modal.enter()
    def load(self):
        import torch
        from diffusers import QwenImage21Pipeline

        started = time.perf_counter()
        try:
            self.pipe = QwenImage21Pipeline.from_pretrained(
                MODEL_ID,
                revision=MODEL_REVISION,
                dtype=torch.bfloat16,
            ).to("cuda")
        finally:
            cache_volume.commit()
        torch.cuda.synchronize()
        self.load_seconds = time.perf_counter() - started

    def _memory(self):
        import torch

        free, total = torch.cuda.mem_get_info()
        return {
            "allocated_gib": _gib(torch.cuda.memory_allocated()),
            "reserved_gib": _gib(torch.cuda.memory_reserved()),
            "peak_allocated_gib": _gib(torch.cuda.max_memory_allocated()),
            "peak_reserved_gib": _gib(torch.cuda.max_memory_reserved()),
            "free_gib": _gib(free),
            "total_gib": _gib(total),
        }

    @modal.method()
    def benchmark(self, sizes: list[tuple[int, int]] | None = None) -> dict:
        import torch

        sizes = sizes or [(1024, 1024)]
        report = {
            "model_id": MODEL_ID,
            "device": torch.cuda.get_device_name(0),
            "load_seconds": round(self.load_seconds, 3),
            "resident_idle": self._memory(),
            "cases": [],
        }

        prompt = (
            "Cinematic wide shot of a rain-soaked futuristic Tokyo alley at night, "
            "a lone detective under warm practical lighting, realistic materials, "
            "subtle fog, anamorphic composition, high detail."
        )

        for width, height in sizes:
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()

            started = time.perf_counter()
            output = self.pipe(
                prompt=prompt,
                width=width,
                height=height,
                num_inference_steps=40,
                true_cfg_scale=1.0,
                negative_prompt=None,
                use_kv_cache=True,
                generator=torch.Generator(device="cuda").manual_seed(42),
            ).images[0]
            torch.cuda.synchronize()

            target = Path(f"/tmp/qwen_image21_{width}x{height}.png")
            output.save(target)
            report["cases"].append(
                {
                    "width": width,
                    "height": height,
                    "elapsed_seconds": round(time.perf_counter() - started, 3),
                    "memory": self._memory(),
                    "output_bytes": target.stat().st_size,
                }
            )
            del output

        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        report["final_idle"] = self._memory()
        return report


@app.local_entrypoint()
def main(full: bool = False):
    sizes = (
        [(1024, 1024), (1536, 1024), (2048, 2048)]
        if full
        else [(1024, 1024)]
    )
    report = QwenImage21Probe().benchmark.remote(sizes)
    print(json.dumps(report, indent=2))
