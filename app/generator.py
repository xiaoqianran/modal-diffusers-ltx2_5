from __future__ import annotations

import gc
import copy
import math
import subprocess
import threading
import time
import types
from pathlib import Path
from typing import Callable

from .config import Settings
from .encoding import encode_video_crf
from .schemas import GenerateRequest


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


class _ProgressRamp:
    """Smoothly advances progress from `lo` toward `hi` while a long opaque step
    (diffusion decode has no step callback) is running."""

    def __init__(self, progress: Callable[[float], None], lo: float, hi: float, tau_s: float):
        self._progress = progress
        self._lo, self._hi, self._tau = lo, hi, tau_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def _run(self):
        start = time.monotonic()
        while not self._stop.wait(2.0):
            elapsed = time.monotonic() - start
            self._progress(self._lo + (self._hi - self._lo) * (1 - math.exp(-elapsed / self._tau)))

    def __enter__(self):
        self._thread.start()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join(timeout=5)
        self._progress(self._hi)


class LTXGenerator:
    """Lazily loads the gated model so health checks remain cheap."""

    def __init__(self, config: Settings):
        self.config = config
        self._pipe = None
        self._upsample_pipe = None
        self._temporal_upsample_pipe = None
        self._diffusion_decode_pipe = None
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
            text_encoder_dir = model_dir / "text_encoder_bnb_4bit"
            transformer_dir = model_dir / "transformer_bnb_4bit"
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
            # Keep non-quantized norms/embeddings and activations in bf16 too.
            # The NF4 config only controls Linear4bit compute dtype on reload.
            text_encoder = Gemma4UnifiedForConditionalGeneration.from_pretrained(
                text_encoder_dir, dtype=torch.bfloat16
            )
            if self.config.ltx25_transformer_precision == "bf16":
                # Release bf16 weights (~38GB, 96GB-class GPUs). text_encoder stays NF4.
                transformer = LTX2VideoTransformer3DModel.from_pretrained(
                    self.config.model_id,
                    subfolder="transformer",
                    revision=self.config.model_revision,
                    torch_dtype=torch.bfloat16,
                )
            else:
                transformer = LTX2VideoTransformer3DModel.from_pretrained(
                    transformer_dir, dtype=torch.bfloat16
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
            self._pipe = pipe
            self._upsample_pipe = upsample_pipe
            self._temporal_upsample_pipe = temporal_upsample_pipe
        return self._pipe

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
                from diffusers.models.autoencoders.ltx2_diffusion_decoder import (
                    LTX2VideoVaeNeighborhoodNattenProcessor,
                )

                decoder.set_attn_processor(LTX2VideoVaeNeighborhoodNattenProcessor())
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

    def _decode_audio(self, pipe, audio_latents):
        """audio latents -> mel (audio_vae) -> waveform (vocoder).

        Needed when the main pipeline runs with output_type="latent" (diffusion decoder
        path). audio_vae.decode() bypasses the cpu-offload forward hook, so briefly move
        the (small) audio_vae to CUDA if it is offloaded.
        """
        import torch

        audio_vae = pipe.audio_vae
        original_device = next(audio_vae.parameters()).device
        with torch.no_grad():
            if original_device.type != "cuda" and torch.cuda.is_available():
                audio_vae.to("cuda")
            try:
                latents = audio_latents.to(next(audio_vae.parameters()).device, audio_vae.dtype)
                mel = audio_vae.decode(latents, return_dict=False)[0]
            finally:
                if original_device.type != "cuda":
                    audio_vae.to(original_device)
            # vocoder forward goes through the offload hook (inputs are moved too).
            audio = pipe.vocoder(mel)
        return audio

    def _finish_retake(self, source: Path, generated: Path, target: Path, request: GenerateRequest) -> None:
        """Keep source media outside the selected interval while using generated media inside it."""
        start, end = request.retake_start, request.retake_end
        video_input = "1:v:0" if request.regenerate_video else "0:v:0"
        command = ["ffmpeg", "-y", "-v", "error", "-i", str(source), "-i", str(generated)]
        if request.regenerate_audio:
            audio_filter = (
                f"[0:a]atrim=0:{start},asetpts=PTS-STARTPTS[a0];"
                f"[1:a]atrim={start}:{end},asetpts=PTS-STARTPTS[a1];"
                f"[0:a]atrim=start={end},asetpts=PTS-STARTPTS[a2];"
                "[a0][a1][a2]concat=n=3:v=0:a=1[a]"
            )
            command += ["-filter_complex", audio_filter, "-map", video_input, "-map", "[a]"]
        else:
            command += ["-map", video_input, "-map", "0:a:0?"]
        command += ["-c:v", "copy", "-c:a", "aac", "-shortest", str(target)]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Retake media merge failed: {result.stderr.strip()[-500:]}")

    def _finish_extend(
        self, source: Path, generated: Path, target: Path, direction: str,
        context_seconds: float, extension_seconds: float,
    ) -> None:
        """Extract the newly generated region and append/prepend it to the source."""
        if direction == "end":
            generated_range = f"start={context_seconds}:end={context_seconds + extension_seconds}"
            order = "[sv][sa][gv][ga]concat=n=2:v=1:a=1[v][a]"
        else:
            generated_range = f"start=0:end={extension_seconds}"
            order = "[gv][ga][sv][sa]concat=n=2:v=1:a=1[v][a]"
        filters = (
            "[0:v]setpts=PTS-STARTPTS[sv];[0:a]asetpts=PTS-STARTPTS[sa];"
            f"[1:v]trim={generated_range},setpts=PTS-STARTPTS[gv];"
            f"[1:a]atrim={generated_range},asetpts=PTS-STARTPTS[ga];" + order
        )
        command = [
            "ffmpeg", "-y", "-v", "error", "-i", str(source), "-i", str(generated),
            "-filter_complex", filters, "-map", "[v]", "-map", "[a]", "-c:v", "libx264",
            "-crf", str(self.config.ltx25_video_crf), "-pix_fmt", "yuv420p", "-c:a", "aac", str(target),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode != 0:
            raise RuntimeError(f"Extend media merge failed: {result.stderr.strip()[-500:]}")

    @staticmethod
    def _decode_audio_file(source: Path, sample_rate: int, start: float, duration: float):
        """Decode a selected region to stereo float32 with ffmpeg."""
        import numpy as np
        command = [
            "ffmpeg", "-v", "error", "-ss", str(start), "-t", str(duration), "-i", str(source),
            "-f", "f32le", "-acodec", "pcm_f32le", "-ac", "2", "-ar", str(sample_rate), "pipe:1",
        ]
        result = subprocess.run(command, capture_output=True)
        if result.returncode != 0:
            raise ValueError("Input audio could not be decoded")
        samples = np.frombuffer(result.stdout, dtype=np.float32)
        if samples.size < 2:
            raise ValueError("Selected audio region is empty")
        samples = samples[: samples.size // 2 * 2].reshape(-1, 2).T.copy()
        wanted = round(duration * sample_rate)
        if samples.shape[1] < wanted:
            samples = np.pad(samples, ((0, 0), (0, wanted - samples.shape[1])))
        return samples[:, :wanted]

    def generate(self, request: GenerateRequest, target: Path, progress: Callable[[float], None]) -> dict[str, float]:
        pipe = self.load()
        adapter_names = []
        lora_root = self.config.lora_dir.resolve()
        try:
            for index, item in enumerate(request.loras):
                path = (lora_root / item.id).resolve()
                if path.parent != lora_root or path.suffix.lower() != ".safetensors" or not path.is_file():
                    raise ValueError(f"LoRA file not found: {item.id}")
                adapter_name = f"job_lora_{index}"
                try:
                    pipe.load_lora_weights(path, adapter_name=adapter_name)
                except Exception as exc:
                    raise RuntimeError(f"LoRA could not be loaded ({item.id}): {exc}") from exc
                adapter_names.append(adapter_name)
            if adapter_names:
                pipe.set_adapters(adapter_names, adapter_weights=[item.strength for item in request.loras])
            return self._generate_impl(request, target, progress)
        finally:
            if request.loras:
                try:
                    pipe.unload_lora_weights()
                except Exception as exc:
                    print(f"[ltx25] LoRA cleanup failed: {exc}", flush=True)

    def _generate_impl(self, request: GenerateRequest, target: Path, progress: Callable[[float], None]) -> dict[str, float]:
        import torch
        from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition
        from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES
        from diffusers.utils import load_image, load_video

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        pipe = self.load()
        generator = torch.Generator(device="cpu").manual_seed(request.seed)
        effective_width, effective_height, effective_fps = request.width, request.height, request.fps
        input_audio_wave = None
        input_audio_latents = None
        restore_audio_prepare = None
        previous_audio_scheduler = None

        conditions = []
        image_suffixes = {".jpg", ".jpeg", ".png", ".webp"}
        video_suffixes = {".mp4", ".mov", ".webm", ".mkv", ".gif"}
        input_dir = self.config.input_dir.resolve()
        source_edit_path = None
        source_frames = None
        for condition in request.conditions:
            matches = list(input_dir.glob(f"{condition.asset_id}.*"))
            if len(matches) != 1:
                raise ValueError(f"Input asset not found: {condition.asset_id}")
            source = matches[0]
            if request.mode in {"retake", "extend"}:
                source_edit_path = source
                source_frames = load_video(str(source))
                continue
            if condition.kind == "image" and source.suffix.lower() in image_suffixes:
                frames = load_image(str(source))
                if request.mode == "iclora":
                    # IC-LoRA expects a reference video. Turn a reference sheet into
                    # a static full-length clip so its latent tokens span the output.
                    frames = [frames.copy() for _ in range(request.num_frames)]
            elif condition.kind == "video" and source.suffix.lower() in video_suffixes:
                frames = load_video(str(source))
            else:
                raise ValueError(f"Input asset type mismatch: {condition.asset_id}")
            conditions.append(
                LTX2VideoCondition(frames=frames, index=condition.index, strength=condition.strength)
            )

        def progress_callback(offset: float, span: float, step_count: int):
            def on_step(_pipe, step: int, _timestep, callback_kwargs):
                progress(min(offset + ((step + 1) / step_count) * span, 0.96))
                return callback_kwargs

            return on_step

        effective_num_frames = request.num_frames
        restore_prepare_latents = None
        extend_context_duration = extend_duration = None
        if request.mode == "retake":
            if not source_frames:
                raise ValueError("Retake source video has no decodable frames")
            effective_num_frames = min(len(source_frames), 481)
            effective_num_frames = ((effective_num_frames - 1) // 8) * 8 + 1
            if effective_num_frames < 9:
                raise ValueError("Retake source must contain at least 9 frames")
            source_frames = source_frames[:effective_num_frames]
            import av
            with av.open(str(source_edit_path)) as container:
                stream = container.streams.video[0]
                effective_width, effective_height = stream.width, stream.height
                effective_fps = float(stream.average_rate or request.fps)
            if effective_width % 32 or effective_height % 32:
                raise ValueError("Retake source resolution must be divisible by 32")
            duration = (effective_num_frames - 1) / effective_fps
            if request.retake_end > duration:
                raise ValueError(f"Retake end exceeds source duration ({duration:.2f}s)")

            pixels = pipe.video_processor.preprocess_video(
                source_frames, height=effective_height, width=effective_width
            ).to(device=pipe._execution_device, dtype=pipe.vae.dtype)
            from diffusers.pipelines.ltx2.pipeline_ltx2_condition import retrieve_latents
            with torch.no_grad():
                source_latents = retrieve_latents(pipe.vae.encode(pixels), sample_mode="argmax")

            original_prepare_latents = pipe.prepare_latents

            def prepare_retake(this, *args, **kwargs):
                latents, _mask, _clean, coords = original_prepare_latents(*args, **kwargs)
                clean_5d = this._normalize_latents(
                    source_latents, this.vae.latents_mean, this.vae.latents_std, this.vae.config.scaling_factor
                ).to(device=latents.device, dtype=latents.dtype)
                clean = this._pack_latents(
                    clean_5d, this.transformer_spatial_patch_size, this.transformer_temporal_patch_size
                )
                latent_frames = clean_5d.shape[2]
                keep = clean_5d.new_ones((clean_5d.shape[0], 1, latent_frames, clean_5d.shape[3], clean_5d.shape[4]))
                if request.regenerate_video:
                    start_latent = max(0, math.floor(request.retake_start * effective_fps / 8))
                    end_latent = min(latent_frames, math.ceil(request.retake_end * effective_fps / 8) + 1)
                    keep[:, :, start_latent:end_latent] = 0
                mask = this._pack_latents(
                    keep, this.transformer_spatial_patch_size, this.transformer_temporal_patch_size
                )
                latents = latents * (1 - mask) + clean * mask
                return latents, mask, clean, coords

            restore_prepare_latents = original_prepare_latents
            pipe.prepare_latents = types.MethodType(prepare_retake, pipe)

        elif request.mode == "extend":
            if not source_frames:
                raise ValueError("Extend source video has no decodable frames")
            import av
            with av.open(str(source_edit_path)) as container:
                stream = container.streams.video[0]
                effective_width, effective_height = stream.width, stream.height
                effective_fps = float(stream.average_rate or request.fps)
            if effective_width % 32 or effective_height % 32:
                raise ValueError("Extend source resolution must be divisible by 32")

            extension_intervals = max(8, round(request.extend_seconds * effective_fps / 8) * 8)
            available_intervals = max(8, ((len(source_frames) - 1) // 8) * 8)
            context_intervals = max(8, round(request.extend_context_seconds * effective_fps / 8) * 8)
            context_intervals = min(context_intervals, available_intervals, 480 - extension_intervals)
            if context_intervals < 8 or extension_intervals + context_intervals > 480:
                raise ValueError("Extend context plus extension exceeds the 481-frame generation limit")
            context_frames_count = context_intervals + 1
            context_frames = (
                source_frames[-context_frames_count:]
                if request.extend_direction == "end" else source_frames[:context_frames_count]
            )
            effective_num_frames = context_intervals + extension_intervals + 1
            extend_context_duration = context_intervals / effective_fps
            extend_duration = extension_intervals / effective_fps

            pixels = pipe.video_processor.preprocess_video(
                context_frames, height=effective_height, width=effective_width
            ).to(device=pipe._execution_device, dtype=pipe.vae.dtype)
            from diffusers.pipelines.ltx2.pipeline_ltx2_condition import retrieve_latents
            with torch.no_grad():
                context_latents = retrieve_latents(pipe.vae.encode(pixels), sample_mode="argmax")
            original_prepare_latents = pipe.prepare_latents

            def prepare_extend(this, *args, **kwargs):
                latents, _mask, _clean, coords = original_prepare_latents(*args, **kwargs)
                normalized_context = this._normalize_latents(
                    context_latents, this.vae.latents_mean, this.vae.latents_std, this.vae.config.scaling_factor
                ).to(device=latents.device, dtype=latents.dtype)
                total_latent_frames = (effective_num_frames - 1) // 8 + 1
                clean_5d = normalized_context.new_zeros(
                    (1, normalized_context.shape[1], total_latent_frames,
                     normalized_context.shape[3], normalized_context.shape[4])
                )
                keep = normalized_context.new_zeros(
                    (1, 1, total_latent_frames, normalized_context.shape[3], normalized_context.shape[4])
                )
                count = normalized_context.shape[2]
                region = slice(0, count) if request.extend_direction == "end" else slice(-count, None)
                clean_5d[:, :, region] = normalized_context
                keep[:, :, region] = 1
                clean = this._pack_latents(
                    clean_5d, this.transformer_spatial_patch_size, this.transformer_temporal_patch_size
                )
                mask = this._pack_latents(
                    keep, this.transformer_spatial_patch_size, this.transformer_temporal_patch_size
                )
                latents = latents * (1 - mask) + clean * mask
                return latents, mask, clean, coords

            restore_prepare_latents = original_prepare_latents
            pipe.prepare_latents = types.MethodType(prepare_extend, pipe)

        if request.mode == "a2v":
            matches = list(input_dir.glob(f"{request.audio_asset_id}.*"))
            if len(matches) != 1:
                raise ValueError("Audio input asset not found")
            audio_source = matches[0]
            probe = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(audio_source)],
                capture_output=True, text=True,
            )
            if probe.returncode != 0:
                raise ValueError("Input audio duration could not be read")
            remaining = float(probe.stdout.strip()) - request.audio_start
            requested_duration = request.audio_duration or min(remaining, 20.0)
            if remaining < 1 or requested_duration > remaining + 0.05:
                raise ValueError("Selected audio range exceeds the input audio")
            intervals = min(480, max(8, round(requested_duration * effective_fps / 8) * 8))
            effective_num_frames = intervals + 1
            actual_duration = intervals / effective_fps

            audio_16k = self._decode_audio_file(audio_source, 16000, request.audio_start, actual_duration)
            input_audio_wave = torch.from_numpy(
                self._decode_audio_file(audio_source, pipe.vocoder.config.output_sampling_rate,
                                        request.audio_start, actual_duration)
            )
            waveform = torch.from_numpy(audio_16k).unsqueeze(0).to(pipe._execution_device)
            import torchaudio
            mel_transform = torchaudio.transforms.MelSpectrogram(
                sample_rate=16000, n_fft=1024, win_length=1024, hop_length=160,
                f_min=0.0, f_max=8000.0, n_mels=64, center=True, pad_mode="reflect",
                power=1.0, mel_scale="slaney", norm="slaney",
            ).to(waveform.device)
            mel = torch.log(torch.clamp(mel_transform(waveform), min=1e-5)).permute(0, 1, 3, 2)
            with torch.no_grad():
                posterior = pipe.audio_vae.encode(mel.to(pipe.audio_vae.dtype), return_dict=False)[0]
                input_audio_latents = posterior.mode()

            original_audio_prepare = pipe.prepare_audio_latents

            def prepare_frozen_audio(this, *args, **kwargs):
                source_latents = kwargs.get("latents")
                if source_latents is None:
                    source_latents = input_audio_latents
                if source_latents.ndim == 4:
                    source_latents = this._pack_audio_latents(source_latents)
                source_latents = this._normalize_audio_latents(
                    source_latents, this.audio_vae.latents_mean, this.audio_vae.latents_std
                )
                return source_latents.to(device=kwargs.get("device"), dtype=kwargs.get("dtype"))

            frozen_scheduler = copy.deepcopy(pipe.scheduler)
            original_set_timesteps = frozen_scheduler.set_timesteps

            def set_frozen_timesteps(
                _self, num_inference_steps=None, device=None, sigmas=None, mu=None, timesteps=None
            ):
                result = original_set_timesteps(
                    num_inference_steps=num_inference_steps, device=device, sigmas=sigmas, mu=mu, timesteps=timesteps
                )
                _self.timesteps = torch.zeros_like(_self.timesteps)
                return result

            def frozen_step(_self, _model_output, _timestep, sample, return_dict=True, **_kwargs):
                return (sample,) if not return_dict else type("FrozenOutput", (), {"prev_sample": sample})()

            frozen_scheduler.set_timesteps = types.MethodType(set_frozen_timesteps, frozen_scheduler)
            frozen_scheduler.step = types.MethodType(frozen_step, frozen_scheduler)
            restore_audio_prepare = original_audio_prepare
            previous_audio_scheduler = pipe.audio_scheduler
            pipe.prepare_audio_latents = types.MethodType(prepare_frozen_audio, pipe)
            pipe.audio_scheduler = frozen_scheduler

        decoder_kind = request.decoder or self.config.ltx25_decoder
        use_refine = request.upscale or request.temporal_upscale
        use_diffusion_decoder = use_refine and decoder_kind == "diffusion"
        final_fps = effective_fps * (2 if request.temporal_upscale else 1)

        args = {
            "prompt": request.prompt,
            "conditions": conditions or None,
            "negative_prompt": request.negative_prompt,
            "width": effective_width,
            "height": effective_height,
            "num_frames": effective_num_frames,
            "min_seconds": request.min_seconds,
            "max_seconds": request.max_seconds,
            "frame_rate": effective_fps,
            "sigmas": DISTILLED_SIGMA_VALUES,
            "guidance_scale": 1.0,
            "audio_guidance_scale": 1.0,
            "stg_scale": 0.0,
            "audio_stg_scale": 0.0,
            "modality_scale": 1.0,
            "audio_modality_scale": 1.0,
            "enable_prompt_enhancement": request.enhance_prompt,
            "generator": generator,
            "output_type": "latent" if use_refine else "np",
            "return_dict": False,
            "callback_on_step_end": progress_callback(
                0.0, 0.55 if use_refine else 0.96, len(DISTILLED_SIGMA_VALUES)
            ),
        }
        if request.mode == "a2v":
            args["audio_latents"] = input_audio_latents
        try:
            video, audio = pipe(**args)
        except Exception:
            if restore_audio_prepare is not None:
                pipe.prepare_audio_latents = restore_audio_prepare
                pipe.audio_scheduler = previous_audio_scheduler
            raise
        finally:
            if restore_prepare_latents is not None:
                pipe.prepare_latents = restore_prepare_latents
        generated_num_frames = effective_num_frames
        if generated_num_frames is None:
            # Auto-duration returns unpacked video latents [B, C, latent_F, H, W].
            generated_num_frames = (video.shape[2] - 1) * pipe.vae_temporal_compression_ratio + 1
        if use_refine:
            progress(0.58)
            if request.upscale:
                video = self._upsample_pipe(
                    latents=video,
                    height=request.height,
                    width=request.width,
                    num_frames=generated_num_frames,
                    output_type="latent",
                    return_dict=False,
                )[0]
            if request.temporal_upscale:
                if self._temporal_upsample_pipe is None:
                    raise RuntimeError(
                        "Temporal latent upsampler is missing. Run "
                        "scripts/download_quantize_ltx25.py --component temporal first."
                    )
                video = self._temporal_upsample_pipe(
                    latents=video,
                    height=request.height * (2 if request.upscale else 1),
                    width=request.width * (2 if request.upscale else 1),
                    num_frames=generated_num_frames,
                    output_type="latent",
                    return_dict=False,
                )[0]
                generated_num_frames = (generated_num_frames - 1) * 2 + 1
            progress(0.64)
            stage2_span = 0.14 if use_diffusion_decoder else 0.32
            try:
                video, audio = pipe(
                    prompt=request.prompt,
                negative_prompt=request.negative_prompt,
                latents=video,
                audio_latents=audio,
                width=effective_width * (2 if request.upscale else 1),
                height=effective_height * (2 if request.upscale else 1),
                num_frames=generated_num_frames,
                min_seconds=request.min_seconds,
                max_seconds=request.max_seconds,
                frame_rate=final_fps,
                sigmas=STAGE_2_DISTILLED_SIGMA_VALUES,
                noise_scale=STAGE_2_DISTILLED_SIGMA_VALUES[0],
                guidance_scale=1.0,
                audio_guidance_scale=1.0,
                stg_scale=0.0,
                audio_stg_scale=0.0,
                modality_scale=1.0,
                audio_modality_scale=1.0,
                generator=generator,
                output_type="latent" if use_diffusion_decoder else "np",
                return_dict=False,
                    callback_on_step_end=progress_callback(
                        0.64, stage2_span, len(STAGE_2_DISTILLED_SIGMA_VALUES)
                    ),
                )
            except Exception:
                if restore_audio_prepare is not None:
                    pipe.prepare_audio_latents = restore_audio_prepare
                    pipe.audio_scheduler = previous_audio_scheduler
                raise
        if restore_audio_prepare is not None:
            pipe.prepare_audio_latents = restore_audio_prepare
            pipe.audio_scheduler = previous_audio_scheduler
        sample_rate = pipe.vocoder.config.output_sampling_rate
        if use_diffusion_decoder:
            # `output_type="latent"` returned de-normalized video latents and audio latents.
            audio_wave = self._decode_audio(pipe, audio)[0].float().cpu()
            progress(0.80)
            decode_pipe = self.load_diffusion_decoder()
            decode_generator = torch.Generator(device="cpu").manual_seed(request.seed)
            decode_start = time.time()
            with _ProgressRamp(progress, 0.80, 0.955, tau_s=90.0), torch.no_grad():
                video = decode_pipe(
                    latents=video.to("cuda"),
                    generator=decode_generator,
                    output_type="np",
                    return_dict=False,
                    denormalize=False,  # latent-path outputs are already denormalized
                )[0]
            print(f"[ltx25] diffusion decode {time.time() - decode_start:.1f}s", flush=True)
        else:
            audio_wave = audio[0].float().cpu()
        if request.mode == "a2v":
            audio_wave = input_audio_wave
        target.parent.mkdir(parents=True, exist_ok=True)
        encode_target = target.with_suffix(".edit-generated.mp4") if request.mode in {"retake", "extend"} else target
        encode_video_crf(
            video[0],
            fps=final_fps,
            audio=audio_wave,
            audio_sample_rate=sample_rate,
            output_path=encode_target,
            crf=self.config.ltx25_video_crf,
        )
        if request.mode == "retake":
            self._finish_retake(source_edit_path, encode_target, target, request)
            encode_target.unlink(missing_ok=True)
        elif request.mode == "extend":
            self._finish_extend(
                source_edit_path, encode_target, target, request.extend_direction,
                extend_context_duration, extend_duration,
            )
            encode_target.unlink(missing_ok=True)
        progress(1.0)
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
        gc.collect()
        torch.cuda.empty_cache()
        return {"peak_vram_gb": peak_vram_gb}
