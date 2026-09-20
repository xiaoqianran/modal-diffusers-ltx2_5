from __future__ import annotations

import gc
import json
import os
import time

import modal

APP_NAME = "ltx25-qwen-image21-dual-probe"
HF_SECRET_NAME = os.environ.get("LTX25_MODAL_HF_SECRET", "huggingface")
LTX_MODEL_VOLUME_NAME = os.environ.get("LTX25_MODAL_MODEL_VOLUME", "ltx25-models")
QWEN_CACHE_VOLUME_NAME = os.environ.get("QWEN_IMAGE21_CACHE_VOLUME", "qwen-image21-cache")
QWEN_MODEL_ID = "Qwen/Qwen-Image-2.1"
QWEN_MODEL_REVISION = "b3179ad355be050328e483a9dfdd9e60cd62adfa"
DIFFUSERS_COMMIT = "80c7ed262aeffbeb43ef13ae04baeb9b84515a69"

MODEL_ROOT = "/models/ltx25"
PIPELINE_DIR = f"{MODEL_ROOT}/pipeline"
NVFP4_CKPT = f"{MODEL_ROOT}/checkpoints/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors"

app = modal.App(APP_NAME)
ltx_models = modal.Volume.from_name(LTX_MODEL_VOLUME_NAME)
qwen_cache = modal.Volume.from_name(QWEN_CACHE_VOLUME_NAME, create_if_missing=True)
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
        index_url="https://download.pytorch.org/whl/cu130",
    )
    .pip_install(
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
    .pip_install(
        "pydantic-settings>=2.7,<3",
        "bitsandbytes==0.50.2",
        "peft==0.21.0",
        "kernels==0.17.1",
    )
    .add_local_dir(
        "ltx25",
        "/app/ltx25",
        copy=False,
        ignore=["**/__pycache__/**", "**/*.pyc"],
    )
)


def _gib(value: int | float) -> float:
    return round(float(value) / 1024**3, 3)


def _memory(torch) -> dict[str, float]:
    free, total = torch.cuda.mem_get_info()
    return {
        "allocated_gib": _gib(torch.cuda.memory_allocated()),
        "reserved_gib": _gib(torch.cuda.memory_reserved()),
        "peak_allocated_gib": _gib(torch.cuda.max_memory_allocated()),
        "peak_reserved_gib": _gib(torch.cuda.max_memory_reserved()),
        "free_gib": _gib(free),
        "total_gib": _gib(total),
    }


@app.cls(
    image=image,
    gpu="RTX-PRO-6000",
    memory=65536,
    timeout=30 * 60,
    startup_timeout=30 * 60,
    max_containers=1,
    scaledown_window=2,
    secrets=[hf_secret],
    volumes={
        "/models": ltx_models.with_mount_options(read_only=True),
        "/cache": qwen_cache,
    },
    env={
        "PYTHONPATH": "/app",
        "HF_HOME": "/cache/huggingface",
        "HF_XET_HIGH_PERFORMANCE": "1",
        "QUANTIZED_MODEL_DIR": PIPELINE_DIR,
        "LTX25_TEXT_ENCODER_DIR": f"{PIPELINE_DIR}/text_encoder",
        "LTX25_TRANSFORMER_CONFIG_DIR": f"{PIPELINE_DIR}/transformer",
        "LTX25_NVFP4_CKPT": NVFP4_CKPT,
        "LTX25_REQUIRE_LOCAL_ASSETS": "1",
        "LTX25_TRANSFORMER_PRECISION": "nvfp4",
        "LTX25_PARALLEL_COLD_LOAD": "1",
        "OFFLOAD_MODE": "none",
        "LTX25_CUDA_GRAPH": "0",
        "LTX25_COMPILE_BLOCKS": "off",
        "OUTPUT_DIR": "/tmp/outputs",
        "INPUT_DIR": "/tmp/inputs",
        "LORA_DIR": "/tmp/loras",
        "HISTORY_DB": "/tmp/history.sqlite3",
    },
)
@modal.concurrent(max_inputs=1)
class DualResidentProbe:
    @modal.method()
    def measure(self, generate_qwen: bool = False) -> dict:
        import torch
        from diffusers import QwenImage21Pipeline
        from ltx25.config import settings
        from ltx25.runtime import LTXGenerator

        torch.cuda.reset_peak_memory_stats()
        report = {
            "device": torch.cuda.get_device_name(0),
            "initial": _memory(torch),
        }

        ltx_started = time.perf_counter()
        self.ltx = LTXGenerator(settings)
        self.ltx.load()
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        report["ltx"] = {
            "load_seconds": round(time.perf_counter() - ltx_started, 3),
            "idle": _memory(torch),
        }

        qwen_started = time.perf_counter()
        try:
            self.qwen = QwenImage21Pipeline.from_pretrained(
                QWEN_MODEL_ID,
                revision=QWEN_MODEL_REVISION,
                dtype=torch.bfloat16,
            ).to("cuda")
        finally:
            qwen_cache.commit()
        torch.cuda.synchronize()
        gc.collect()
        torch.cuda.empty_cache()
        report["qwen_after_ltx"] = {
            "load_seconds": round(time.perf_counter() - qwen_started, 3),
            "dual_resident_idle": _memory(torch),
        }

        if generate_qwen:
            torch.cuda.reset_peak_memory_stats()
            started = time.perf_counter()
            image = self.qwen(
                prompt=(
                    "Cinematic wide shot of a rain-soaked futuristic Tokyo alley at night, "
                    "a lone detective under warm practical lighting, realistic materials, "
                    "subtle fog, anamorphic composition, high detail."
                ),
                width=1024,
                height=1024,
                num_inference_steps=40,
                true_cfg_scale=1.0,
                negative_prompt=None,
                use_kv_cache=True,
                generator=torch.Generator(device="cuda").manual_seed(42),
            ).images[0]
            torch.cuda.synchronize()
            report["qwen_1024_with_ltx_resident"] = {
                "elapsed_seconds": round(time.perf_counter() - started, 3),
                "memory": _memory(torch),
            }
            del image
            gc.collect()
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
            report["after_qwen_generation_idle"] = _memory(torch)

        return report


@app.local_entrypoint()
def main(generate_qwen: bool = False):
    report = DualResidentProbe().measure.remote(generate_qwen)
    print(json.dumps(report, indent=2))
