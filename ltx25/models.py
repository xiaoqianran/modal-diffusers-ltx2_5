from __future__ import annotations

import gc
import threading
import time
from pathlib import Path

from .config import Settings

FP8_SKIP_MODULES_PATTERN = (
    "norm",
    "^proj_in$",
    "^proj_out$",
    "^audio_proj_in$",
    "^audio_proj_out$",
    "time_embed",
    "audio_time_embed",
    "av_cross_attn_video_scale_shift",
    "av_cross_attn_audio_scale_shift",
    "av_cross_attn_video_a2v_gate",
    "av_cross_attn_audio_v2a_gate",
    "prompt_adaln",
    "audio_prompt_adaln",
    "to_gate_logits",  # small per-block gate projections, keep precise
)


def _patch_flex_for_real_kernels() -> None:
    """diffusers' flex processor calls torch flex_attention eagerly, which falls back to the
    math kernel (materializes the full NxN score matrix -> OOM at production grids), and
    create_block_mask without _compile=True materializes a full bool mask. Compile both so
    the actual block-sparse flex kernel runs."""
    import torch
    import torch.nn.attention.flex_attention as fa_mod

    if getattr(fa_mod, "_ltx25_flex_patched", False):
        return
    import torch._dynamo as dynamo

    # Many distinct grids/head-counts across decoder stages: the default recompile limit (8)
    # silently drops back to the eager math fallback (full NxN scores -> OOM).
    dynamo.config.recompile_limit = 256
    dynamo.config.cache_size_limit = 256
    orig_flex = fa_mod.flex_attention
    orig_cbm = fa_mod.create_block_mask
    compiled_flex = torch.compile(orig_flex)

    def cbm(*args, **kwargs):
        # `_compile=True` internally calls `torch.compile(create_block_mask)(...)`, which
        # resolves the module-global again -> restore the original around the call to
        # avoid infinite recursion through this wrapper.
        kwargs.setdefault("_compile", True)
        fa_mod.create_block_mask = orig_cbm
        try:
            return orig_cbm(*args, **kwargs)
        finally:
            fa_mod.create_block_mask = cbm

    fa_mod.flex_attention = compiled_flex
    fa_mod.create_block_mask = cbm
    fa_mod._ltx25_flex_patched = True


def _create_natten_processor():
    """Create the LTX2 NATTEN processor with a repo-scoped kernels allowlist.

    `kernels>=0.17` no longer implicitly trusts the publisher metadata that the
    current Diffusers processor assumes. Trust only the exact NATTEN repository
    instead of enabling arbitrary remote kernel code globally.
    """
    from diffusers.models.autoencoders.ltx2_diffusion_decoder import (
        LTX2VideoVaeNeighborhoodNattenProcessor,
    )
    from kernels import get_kernel

    processor = LTX2VideoVaeNeighborhoodNattenProcessor.__new__(
        LTX2VideoVaeNeighborhoodNattenProcessor
    )
    processor._na3d = get_kernel(
        "shi-labs/natten",
        version=1,
        trust_remote_code=["shi-labs/natten"],
    ).na3d
    processor.backend = None
    return processor


class ModelLifecycle:
    """Owns model loading, unloading, decoder setup, and acceleration installation."""

    def __init__(self, config: Settings):
        self.config = config
        self._pipe = None
        self._upsample_pipe = None
        self._temporal_upsample_pipe = None
        self._diffusion_decode_pipe = None
        self._graph_runner = None
        self._load_lock = threading.Lock()

    def load(self):
        if self._pipe is not None:
            return self._pipe
        with self._load_lock:
            if self._pipe is not None:
                return self._pipe
            import torch
            from diffusers import LTX2ConditionPipeline, LTX2LatentUpsamplePipeline, LTX2VideoTransformer3DModel
            from diffusers.pipelines.ltx2.latent_upsampler import LTX2LatentUpsamplerModel
            from transformers import Gemma4UnifiedForConditionalGeneration

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA GPU is required for LTX-2.5 inference")
            model_dir = self.config.quantized_model_dir.resolve()
            text_encoder_dir = (
                self.config.ltx25_text_encoder_dir.resolve()
                if self.config.ltx25_text_encoder_dir is not None
                else model_dir / "text_encoder_bnb_4bit"
            )
            transformer_dir = (
                self.config.ltx25_transformer_config_dir.resolve()
                if self.config.ltx25_transformer_config_dir is not None
                else model_dir / "transformer_bnb_4bit"
            )
            if not text_encoder_dir.is_dir() or not transformer_dir.is_dir():
                raise RuntimeError(
                    f"Quantized LTX-2.5 components are missing under {model_dir}. "
                    "Run scripts/download_quantize_ltx25.py first."
                )
            upsampler_dir = model_dir / "latent_upsampler"
            if not upsampler_dir.is_dir():
                raise RuntimeError(
                    f"LTX-2.5 latent upsampler is missing under {model_dir}. "
                    "Run scripts/download_quantize_ltx25.py --component quality first."
                )
            temporal_upsampler_dir = model_dir / "temporal_latent_upsampler"
            # Gemma and the NVFP4 transformer are independent large reads. Modal's
            # cold-start guidance recommends loading independent model files
            # concurrently; overlap the CPU-side Gemma load with the GPU-side
            # NVFP4 load on the resident-worker path.
            text_encoder_future = None
            text_encoder_pool = None

            def load_text_encoder():
                t0 = time.time()
                model = Gemma4UnifiedForConditionalGeneration.from_pretrained(
                    text_encoder_dir, dtype=torch.bfloat16
                )
                return model, time.time() - t0

            parallel_cold_load = (
                self.config.ltx25_parallel_cold_load
                and self.config.ltx25_transformer_precision == "nvfp4"
            )
            if parallel_cold_load:
                from concurrent.futures import ThreadPoolExecutor

                text_encoder_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ltx25-load")
                text_encoder_future = text_encoder_pool.submit(load_text_encoder)
            else:
                text_encoder, _ = load_text_encoder()
            if self.config.ltx25_transformer_precision == "bf16":
                # Release bf16 weights (~38GB, 96GB-class GPUs). text_encoder stays NF4.
                transformer = LTX2VideoTransformer3DModel.from_pretrained(
                    self.config.model_id,
                    subfolder="transformer",
                    revision=self.config.model_revision,
                    torch_dtype=torch.bfloat16,
                )
            elif self.config.ltx25_transformer_precision == "fp8":
                # Verified recipe from scratch_fp8_probe/ (F1-F5): load the bf16 Hub
                # shards on CPU, then apply diffusers layerwise casting
                # (storage=fp8_e4m3fn / compute=bf16) *while still on CPU*. Casting
                # on GPU instead needs a transient ~43GB bf16-on-GPU peak that OOMs
                # a real 48GB card (probe F4b established the CPU-cast path as the
                # 48GB-class recipe). Afterwards the module joins the normal
                # offload_mode handling below (model_cpu_offload verified: F2/F4).
                from diffusers.hooks import apply_layerwise_casting

                fp8_t0 = time.time()
                transformer = LTX2VideoTransformer3DModel.from_pretrained(
                    self.config.model_id,
                    subfolder="transformer",
                    revision=self.config.model_revision,
                    torch_dtype=torch.bfloat16,
                )
                apply_layerwise_casting(
                    transformer,
                    storage_dtype=torch.float8_e4m3fn,
                    compute_dtype=torch.bfloat16,
                    skip_modules_pattern=FP8_SKIP_MODULES_PATTERN,
                    non_blocking=False,
                )
                resident_gb = sum(
                    p.numel() * p.element_size() for p in transformer.parameters()
                ) / 1024**3
                print(
                    f"[ltx25] transformer precision=fp8 (layerwise cast applied on CPU "
                    f"in {time.time() - fp8_t0:.1f}s, resident weights {resident_gb:.1f}GB)",
                    flush=True,
                )
            elif self.config.ltx25_transformer_precision == "nvfp4":
                # Official Blackwell-native FP4 distilled transformer (single-file
                # ComfyUI format). Loaded straight onto the GPU by ltx25/acceleration/nvfp4.py:
                # quantized Linears become NVFP4Linear (torch._scaled_mm FP4 GEMM,
                # ~3.4x raw / ~1.8x per-layer vs bf16 incl. activation-quant cost).
                # The bnb transformer_dir is only used for its config.json (same
                # architecture); its weights are not read.
                import json as _json

                from .acceleration.nvfp4 import load_nvfp4_transformer

                nvfp4_ckpt = self.config.ltx25_nvfp4_ckpt
                if not nvfp4_ckpt and self.config.ltx25_require_local_assets:
                    raise RuntimeError(
                        "LTX25_NVFP4_CKPT is required when LTX25_REQUIRE_LOCAL_ASSETS=1. "
                        "Stage the NVFP4 checkpoint before starting the GPU worker."
                    )
                if not nvfp4_ckpt:
                    from huggingface_hub import hf_hub_download

                    nvfp4_ckpt = hf_hub_download(
                        "Lightricks/LTX-2.5",
                        "diffusion_models/ltx-2.5-22b-distilled-transformer-nvfp4.safetensors",
                    )
                nvfp4_path = Path(nvfp4_ckpt).resolve()
                if not nvfp4_path.is_file():
                    raise RuntimeError(f"NVFP4 checkpoint not found: {nvfp4_path}")
                with open(transformer_dir / "config.json") as fh:
                    nvfp4_cfg = _json.load(fh)
                nvfp4_t0 = time.time()
                transformer = load_nvfp4_transformer(
                    str(nvfp4_path), nvfp4_cfg, torch.device("cuda")
                )
                print(
                    f"[ltx25] transformer precision=nvfp4 loaded in "
                    f"{time.time() - nvfp4_t0:.1f}s (FP4 GEMM, resident ~19GB)",
                    flush=True,
                )
            else:
                transformer = LTX2VideoTransformer3DModel.from_pretrained(
                    transformer_dir, dtype=torch.bfloat16
                )
            if text_encoder_future is not None:
                text_encoder, text_encoder_seconds = text_encoder_future.result()
                text_encoder_pool.shutdown(wait=True)
                print(
                    f"[ltx25] parallel cold load: Gemma {text_encoder_seconds:.1f}s "
                    "overlapped with NVFP4 transformer",
                    flush=True,
                )
            pipe = LTX2ConditionPipeline.from_pretrained(
                model_dir,
                text_encoder=text_encoder,
                transformer=transformer,
                dtype=torch.bfloat16,
                local_files_only=True,
            )
            pipe.vae.enable_tiling()
            latent_upsampler = LTX2LatentUpsamplerModel.from_pretrained(
                model_dir,
                subfolder="latent_upsampler",
                dtype=torch.bfloat16,
                local_files_only=True,
            )
            upsample_pipe = LTX2LatentUpsamplePipeline(vae=pipe.vae, latent_upsampler=latent_upsampler)
            temporal_upsample_pipe = None
            if temporal_upsampler_dir.is_dir():
                temporal_upsampler = LTX2LatentUpsamplerModel.from_pretrained(
                    model_dir,
                    subfolder="temporal_latent_upsampler",
                    dtype=torch.bfloat16,
                    local_files_only=True,
                )
                temporal_upsample_pipe = LTX2LatentUpsamplePipeline(
                    vae=pipe.vae, latent_upsampler=temporal_upsampler
                )
            if self.config.offload_mode == "none":
                pipe.to("cuda")
                upsample_pipe.to("cuda")
                if temporal_upsample_pipe is not None:
                    temporal_upsample_pipe.to("cuda")
            elif self.config.offload_mode == "sequential":
                pipe.enable_sequential_cpu_offload()
                upsample_pipe.enable_sequential_cpu_offload()
                if temporal_upsample_pipe is not None:
                    temporal_upsample_pipe.enable_sequential_cpu_offload()
            else:
                pipe.enable_model_cpu_offload()
                upsample_pipe.enable_model_cpu_offload()
                if temporal_upsample_pipe is not None:
                    temporal_upsample_pipe.enable_model_cpu_offload()
            if self.config.ltx25_compile_blocks != "off":
                # per-block torch.compile(ltx25/acceleration/compile.py)。CUDA Graph の
                # install より前に適用する(graph は compile 済みブロックの呼び出しを
                # capture する必要がある)。compiled eager 単体は素の eager より遅い
                # ため、graph 無効時・nvfp4 以外・offload 有効時は適用しない。
                _cb = self.config.ltx25_compile_blocks
                if (
                    self.config.ltx25_transformer_precision == "nvfp4"
                    and self.config.ltx25_cuda_graph
                    and self.config.offload_mode == "none"
                ):
                    from .acceleration.compile import apply_block_compile

                    apply_block_compile(pipe.transformer, _cb)
                else:
                    print(
                        f"[ltx25] LTX25_COMPILE_BLOCKS={_cb} ignored: requires "
                        "precision=nvfp4 + LTX25_CUDA_GRAPH=1 + OFFLOAD_MODE=none "
                        f"(got precision={self.config.ltx25_transformer_precision!r}, "
                        f"cuda_graph={self.config.ltx25_cuda_graph}, "
                        f"offload={self.config.offload_mode!r})",
                        flush=True,
                    )
            if self.config.ltx25_cuda_graph:
                # transformer.forward 全体の CUDA Graph 化(ltx25/acceleration/cuda_graph.py 参照)。
                # OFFLOAD_MODE=none 限定: model/sequential offload は重みのデバイスが
                # リクエスト間で動き、capture 済み graph が焼き込んだアドレスと
                # 食い違って黙って壊れるため適用しない。
                if self.config.offload_mode == "none":
                    from .acceleration.cuda_graph import ForwardGraphRunner

                    self._graph_runner = ForwardGraphRunner(
                        pipe.transformer,
                        max_captures=self.config.ltx25_cuda_graph_max_captures,
                    )
                    self._graph_runner.install()
                    print(
                        "[ltx25] CUDA graph capture enabled for transformer.forward "
                        f"(max_captures={self.config.ltx25_cuda_graph_max_captures})",
                        flush=True,
                    )
                else:
                    print(
                        "[ltx25] LTX25_CUDA_GRAPH=1 ignored: requires OFFLOAD_MODE=none "
                        f"(got {self.config.offload_mode!r})",
                        flush=True,
                    )
            self._pipe = pipe
            self._upsample_pipe = upsample_pipe
            self._temporal_upsample_pipe = temporal_upsample_pipe
        return self._pipe

    def unload(self) -> dict:
        """Release every pipeline/model reference so VRAM returns to (near) zero while
        the process stays alive (Phase 5a resident switching). The next generate()
        simply goes through load() again -- load() only checks `self._pipe is None`,
        so dropping the references restores the exact lazy-load entry state.
        Callers must ensure no generation is running before unloading."""
        freed = []
        with self._load_lock:
            if self._graph_runner is not None:
                # capture 済み graph は transformer の重み・mempool を参照し続けるため、
                # パイプライン参照を落とす前に破棄しないと VRAM が返らない。
                self._graph_runner.reset()
                self._graph_runner.uninstall()
                self._graph_runner = None
                freed.append("cuda_graphs")
            for attr in ("_pipe", "_upsample_pipe", "_temporal_upsample_pipe",
                         "_diffusion_decode_pipe"):
                if getattr(self, attr) is not None:
                    setattr(self, attr, None)
                    freed.append(attr.lstrip("_"))
            gc.collect()
            allocated_gb = None
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    allocated_gb = round(torch.cuda.memory_allocated() / 1024**3, 3)
            except Exception:  # torch never imported / no CUDA: nothing to free
                pass
        return {"freed": freed, "allocated_gb": allocated_gb}

    def _configure_decode_tiling(self, num_frames: int, height: int, width: int) -> None:
        """decode 直前にタイル構成を決める(ジョブごと、decoder は常駐共有のため毎回設定)。

        既定タイル(768^2x80f / stride 704^2x56f)は 24GB 級を想定した保守値で、
        大出力ではタイル数と重複(オーバーラップ)計算が膨らむ(1536x896x121f で
        12 タイル・重複 約1.4x)。ここでは **空き VRAM の範囲で最大のタイル**
        (最小の分割数)を選ぶ: 分割は幅→高さ→フレームの順に増やし、
        タイル体積 <= 予算(空きVRAM / 0.34GB/Mpx / 1.25 マージン、
        experiments/probes/probe_decode_tiling.py の実測係数)を満たす最初の構成を採る。
        1024x576x121f では単一タイルになり decode 9.6s -> 7.9s(継ぎ目も消える)。
        LTX25_DECODE_SINGLE_TILE: auto(既定)/ on(常に単一タイル、VRAM検査なし)/
        off(常に既定タイル)。"""
        import torch

        decoder = self._diffusion_decode_pipe.diffusion_decoder
        mode = self.config.ltx25_decode_single_tile
        if mode == "off":
            decoder.enable_tiling()  # 既定値へ戻す
            return

        OVERLAP_PX = 64   # 既定タイルと同じ空間オーバーラップ(768-704)
        OVERLAP_F = 8     # 時間方向(既定24は保守的すぎるため最小の8n)
        free_gb = torch.cuda.mem_get_info()[0] / 1024**3
        budget_mpx = free_gb / (0.34 * 1.25) * 1e6

        def tile_dims(nw: int, nh: int, nf: int) -> tuple[int, int, int]:
            tw = -(-width // nw) + (OVERLAP_PX if nw > 1 else 0)
            th = -(-height // nh) + (OVERLAP_PX if nh > 1 else 0)
            tf = -(-num_frames // nf) + (OVERLAP_F if nf > 1 else 0)
            return tw, th, tf

        # 幅→高さ→フレームの順で分割を増やし、予算に収まる最初の構成を採用
        candidates = [(1, 1, 1), (2, 1, 1), (2, 2, 1), (3, 2, 1), (2, 2, 2),
                      (3, 2, 2), (3, 3, 2), (4, 3, 2)]
        chosen = None
        for nw, nh, nf in candidates:
            tw, th, tf = tile_dims(nw, nh, nf)
            if mode == "on" or tw * th * tf <= budget_mpx:
                chosen = (nw, nh, nf, tw, th, tf)
                break
        if chosen is None:
            decoder.enable_tiling()
            print(
                f"[ltx25] decode tiling: default tiles (budget {budget_mpx/1e6:.0f}Mpx "
                f"too small for {width}x{height}x{num_frames}f)",
                flush=True,
            )
            return
        nw, nh, nf, tw, th, tf = chosen
        # stride は「各次元の刻み = ceil(dim/n)」。オーバーラップは tile_min 側にだけ
        # 足してあるため、分割数1の次元では引かない(引くと2タイル化してしまう)。
        decoder.enable_tiling(
            tile_sample_min_height=th,
            tile_sample_min_width=tw,
            tile_sample_min_num_frames=tf,
            tile_sample_stride_height=th - (OVERLAP_PX if nh > 1 else 0),
            tile_sample_stride_width=tw - (OVERLAP_PX if nw > 1 else 0),
            tile_sample_stride_num_frames=tf - (OVERLAP_F if nf > 1 else 0),
        )
        print(
            f"[ltx25] decode tiling: {nw}x{nh}x{nf} tiles of {tw}x{th}x{tf}f "
            f"(free {free_gb:.1f}GB, budget {budget_mpx/1e6:.0f}Mpx)",
            flush=True,
        )

    def load_diffusion_decoder(self):
        """Lazily load the LTX-2.5 diffusion decoder pipeline (~0.83GB, kept resident)."""
        if self._diffusion_decode_pipe is not None:
            return self._diffusion_decode_pipe
        with self._load_lock:
            if self._diffusion_decode_pipe is not None:
                return self._diffusion_decode_pipe
            import torch
            from diffusers import FlowMatchEulerDiscreteScheduler
            from diffusers.models.autoencoders import LTX2VideoDiffusionDecoderModel
            from diffusers.pipelines.ltx2 import LTX2VideoDiffusionDecodePipeline

            local_decoder_dir = self.config.quantized_model_dir.resolve() / "diffusion_decoder"
            if self.config.ltx25_require_local_assets and not local_decoder_dir.is_dir():
                raise RuntimeError(
                    f"Pre-staged diffusion decoder is missing: {local_decoder_dir}. "
                    "Run the CPU model preparation step before starting the GPU worker."
                )
            if local_decoder_dir.is_dir():
                decoder = LTX2VideoDiffusionDecoderModel.from_pretrained(
                    local_decoder_dir,
                    torch_dtype=torch.bfloat16,
                    local_files_only=True,
                )
            else:
                decoder = LTX2VideoDiffusionDecoderModel.from_pretrained(
                    self.config.model_id,
                    subfolder="diffusion_decoder",
                    revision=self.config.model_revision,
                    torch_dtype=torch.bfloat16,
                )
            decoder.to("cuda")
            # Bound per-tile attention grids (121f x 1024^2 untiled OOMs even at 96GB).
            decoder.enable_tiling()
            scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
                str(self.config.quantized_model_dir.resolve()), subfolder="scheduler"
            )
            # Prefer NATTEN's fused na3d kernel (fetched from the Hub via `kernels`,
            # needs torch>=2.11 prebuilt variants); fall back to the compiled
            # flex-attention patch when unavailable.
            try:
                decoder.set_attn_processor(_create_natten_processor())
                print("[ltx25] diffusion decoder attention: NATTEN na3d (shi-labs/natten)", flush=True)
            except Exception as exc:
                # Compile flex-attention so the block-sparse kernel actually runs (the eager
                # fallback materializes full NxN scores and OOMs at production grids).
                print(
                    f"[ltx25] NATTEN unavailable ({type(exc).__name__}: {exc}); "
                    "diffusion decoder attention: compiled flex-attention fallback",
                    flush=True,
                )
                _patch_flex_for_real_kernels()
            self._diffusion_decode_pipe = LTX2VideoDiffusionDecodePipeline(
                diffusion_decoder=decoder, scheduler=scheduler, vae=None
            )
        return self._diffusion_decode_pipe
