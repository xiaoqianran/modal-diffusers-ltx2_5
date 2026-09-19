from __future__ import annotations

import gc
import copy
import math
import os
import subprocess
import threading
import time
import types
from pathlib import Path
from typing import Callable

from .encoding import encode_video_crf
from .models import ModelLifecycle
from .schemas import STILL_IMAGE_MODES, GenerateRequest


PIXEL_UPSCALER_FILENAME = "ltx-2.5-22b-ic-lora-pixel-spatial-upscaler-x2-1.0.safetensors"
# Official DFR uses the detailing IC-LoRA at 0.5. The standalone V2V/IC-LoRA
# workflow recommends 1.0, but this runtime uses the DFR-style two-stage path:
# half-resolution stage 1 -> latent x2 -> full-resolution detailing stage 2.
PIXEL_DETAILING_LORA_STRENGTH = 0.5


class _PatchRegistry:
    """Track temporary attribute replacements and restore them exactly once.

    The resident pipeline is reused across jobs, so any per-request monkey patch
    must be exception-safe across the *entire* generation workflow, not only the
    immediate pipeline call where it is installed.
    """

    def __init__(self) -> None:
        self._restores: list[Callable[[], None]] = []

    def replace(self, obj, attr: str, value) -> tuple[object, Callable[[], None]]:
        original = getattr(obj, attr)
        active = True

        def restore() -> None:
            nonlocal active
            if not active:
                return
            setattr(obj, attr, original)
            active = False

        setattr(obj, attr, value)
        self._restores.append(restore)
        return original, restore

    def restore_all(self) -> None:
        for restore in reversed(self._restores):
            restore()
        self._restores.clear()

# fp8 layerwise-casting skip list, verified by module enumeration against
# Lightricks/LTX-2.5-Diffusers subfolder=transformer at the pinned revision
# (faithful copy of scratch_fp8_probe/fp8_common.py::FP8_SKIP_MODULES_PATTERN;
# protects embeddings / patchify / AdaLN / scale-shift / gate projections and
# casts only the big attention/FF Linear layers inside transformer_blocks).




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


def _subsample_distilled_sigmas(sigmas, steps):
    """蒸留σ列の間引き(steps<len のときのみ。先頭・末尾を含む等間隔選択)。"""
    try:
        steps = int(steps or 0)
    except (TypeError, ValueError):
        return sigmas
    n = len(sigmas)
    if steps <= 0 or steps >= n:
        return sigmas
    if steps == 1:
        picked = [sigmas[0]]
    else:
        idx = [round(i * (n - 1) / (steps - 1)) for i in range(steps)]
        picked = [sigmas[i] for i in idx]
    print(f"[ltx25] distilled sigma subsample: {steps}/{n} steps -> {picked}", flush=True)
    return picked


class LTXGenerator(ModelLifecycle):
    """Lazily loads the gated model so health checks remain cheap."""






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

    def _get_mel_transform_16k(self, device):
        """Return a cached pure-PyTorch Slaney MelSpectrogram transform for A2V.

        This intentionally avoids torchaudio: the production image currently uses a
        newer torch CUDA wheel than the matching torchaudio wheels published for
        cu130. Keeping the transform in torch avoids binary-version coupling while
        preserving the previous 16 kHz / 64-bin / Slaney configuration.
        """
        cache = getattr(self, "_mel_transform_cache", None)
        if cache is None:
            cache = {}
            self._mel_transform_cache = cache
        key = str(device)
        if key not in cache:
            import math
            import torch

            n_fft = 1024
            win_length = 1024
            hop_length = 160
            sample_rate = 16000
            n_mels = 64
            n_freqs = n_fft // 2 + 1

            def hz_to_mel(freq):
                f_sp = 200.0 / 3.0
                min_log_hz = 1000.0
                min_log_mel = min_log_hz / f_sp
                logstep = math.log(6.4) / 27.0
                return torch.where(
                    freq >= min_log_hz,
                    min_log_mel + torch.log(torch.clamp(freq, min=min_log_hz) / min_log_hz) / logstep,
                    freq / f_sp,
                )

            def mel_to_hz(mel):
                f_sp = 200.0 / 3.0
                min_log_hz = 1000.0
                min_log_mel = min_log_hz / f_sp
                logstep = math.log(6.4) / 27.0
                return torch.where(
                    mel >= min_log_mel,
                    min_log_hz * torch.exp(logstep * (mel - min_log_mel)),
                    f_sp * mel,
                )

            freqs = torch.linspace(0.0, sample_rate / 2.0, n_freqs, device=device)
            mel_min = hz_to_mel(torch.tensor(0.0, device=device))
            mel_max = hz_to_mel(torch.tensor(sample_rate / 2.0, device=device))
            mel_points = torch.linspace(mel_min, mel_max, n_mels + 2, device=device)
            hz_points = mel_to_hz(mel_points)
            lower = hz_points[:-2, None]
            center = hz_points[1:-1, None]
            upper = hz_points[2:, None]
            up_slope = (freqs[None, :] - lower) / torch.clamp(center - lower, min=1e-12)
            down_slope = (upper - freqs[None, :]) / torch.clamp(upper - center, min=1e-12)
            filterbank = torch.clamp(torch.minimum(up_slope, down_slope), min=0.0)
            filterbank = filterbank * (2.0 / torch.clamp(hz_points[2:] - hz_points[:-2], min=1e-12))[:, None]
            window = torch.hann_window(win_length, periodic=True, device=device)

            def mel_transform(waveform):
                shape = waveform.shape
                flat = waveform.reshape(-1, shape[-1])
                spectrum = torch.stft(
                    flat,
                    n_fft=n_fft,
                    hop_length=hop_length,
                    win_length=win_length,
                    window=window,
                    center=True,
                    pad_mode="reflect",
                    return_complex=True,
                ).abs()
                mel = torch.matmul(filterbank.to(spectrum.dtype), spectrum)
                return mel.reshape(*shape[:-1], n_mels, mel.shape[-1])

            cache[key] = mel_transform
        return cache[key]

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

    def _cast_lora_layers_to_bf16(self, pipe) -> None:
        """fp8 LoRA compatibility workaround (verified probe F5): lora_A/lora_B
        Linear layers created by load_lora_weights() *after* layerwise casting are
        materialized at the base layer's current storage dtype (fp8_e4m3fn), and the
        forward pass then fails with NotImplementedError '"addmm_cuda" not
        implemented for Float8_e4m3fn'. Cast the (tiny) LoRA modules back to bf16.
        No-op for nf4/bf16 precisions."""
        if self.config.ltx25_transformer_precision != "fp8":
            return
        import torch

        n_cast = 0
        for name, module in pipe.transformer.named_modules():
            if "lora_A" in name or "lora_B" in name:
                module.to(torch.bfloat16)
                n_cast += 1
        if n_cast:
            print(f"[ltx25] fp8 LoRA compat: cast {n_cast} lora_A/lora_B modules to bf16", flush=True)

    def generate(self, request: GenerateRequest, target: Path, progress: Callable[[float], None]) -> dict[str, float]:
        pipe = self.load()
        patches = _PatchRegistry()
        adapter_names = []
        lora_root = self.config.lora_dir.resolve()
        nvfp4_lora = self.config.ltx25_transformer_precision == "nvfp4"
        # LoRA を載せるジョブ(job loras / pixel upscale の IC-LoRA)は CUDA graph 不可:
        # capture は重みテンソルのアドレスを焼き込むため、adapter の付け外しをまたぐ
        # replay は stale な重みを黙って使う。eager に落とし、ジョブ後に capture を捨てる。
        _graph_lora_guard = self._graph_runner is not None and (
            bool(request.loras) or request.upscale_method == "pixel"
        )
        if _graph_lora_guard:
            self._graph_runner.enabled = False
        try:
            for index, item in enumerate(request.loras):
                path = (lora_root / item.id).resolve()
                if path.parent != lora_root or path.suffix.lower() != ".safetensors" or not path.is_file():
                    raise ValueError(f"LoRA file not found: {item.id}")
                adapter_name = f"job_lora_{index}"
                try:
                    if nvfp4_lora:
                        from .acceleration.nvfp4 import install_nvfp4_lora

                        count = install_nvfp4_lora(
                            pipe.transformer, path, adapter_name, item.strength
                        )
                        print(
                            f"[ltx25] NVFP4 LoRA {adapter_name}: attached to {count} layers",
                            flush=True,
                        )
                    else:
                        pipe.load_lora_weights(path, adapter_name=adapter_name)
                except Exception as exc:
                    raise RuntimeError(f"LoRA could not be loaded ({item.id}): {exc}") from exc
                adapter_names.append(adapter_name)
            if adapter_names and not nvfp4_lora:
                pipe.set_adapters(adapter_names, adapter_weights=[item.strength for item in request.loras])
                self._cast_lora_layers_to_bf16(pipe)
            if request.mode in STILL_IMAGE_MODES:
                return self._generate_still_impl(request, target, progress)
            return self._generate_impl(request, target, progress, patches)
        finally:
            patches.restore_all()
            if request.loras or request.upscale_method == "pixel":
                try:
                    if nvfp4_lora:
                        from .acceleration.nvfp4 import remove_nvfp4_loras

                        remove_nvfp4_loras(pipe.transformer)
                    else:
                        pipe.unload_lora_weights()
                except Exception as exc:
                    print(f"[ltx25] LoRA cleanup failed: {exc}", flush=True)
            if _graph_lora_guard:
                # unload_lora_weights 後の構造復元を信用せず、防御的に capture を捨てる
                # (次の非 LoRA ジョブが ~1s で再 capture する)。
                self._graph_runner.reset()
                self._graph_runner.enabled = True

    def _generate_still_impl(
        self, request: GenerateRequest, target: Path, progress: Callable[[float], None]
    ) -> dict[str, float]:
        """Still-image modes (t2i / refine_image / ref2i), kept independent from the
        video path. Recipes are faithful ports of the verified probes under
        scratch_t2i_probe/ (P3 for t2i, P4_nf17 for t2i+diffusion decoder, P6/P6b for
        refine_image, P7a/P7b for ref2i)."""
        import numpy as np
        import torch
        from PIL import Image
        from diffusers.pipelines.ltx2.pipeline_ltx2_condition import LTX2VideoCondition
        from diffusers.pipelines.ltx2.utils import DISTILLED_SIGMA_VALUES, STAGE_2_DISTILLED_SIGMA_VALUES
        from diffusers.utils import load_image

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
        pipe = self.load()
        generator = torch.Generator(device="cpu").manual_seed(request.seed)

        image_suffixes = {".jpg", ".jpeg", ".png", ".webp"}
        input_dir = self.config.input_dir.resolve()
        conditions = []
        for condition in request.conditions:
            matches = list(input_dir.glob(f"{condition.asset_id}.*"))
            if len(matches) != 1:
                raise ValueError(f"Input asset not found: {condition.asset_id}")
            source = matches[0]
            if condition.kind != "image" or source.suffix.lower() not in image_suffixes:
                raise ValueError(f"Still-image modes accept image assets only: {condition.asset_id}")
            strength = request.strength if request.mode == "refine_image" else condition.strength
            conditions.append(
                LTX2VideoCondition(frames=load_image(str(source)), index=condition.index, strength=strength)
            )

        # t2i (no image conditioning) follows the configured default decoder — the
        # diffusion decoder is dramatically sharper for pure text-to-still (probe P4).
        # Image-conditioned stills stay on VAE (diffusion decoder blurs them, probe P8).
        if request.mode == "t2i":
            decoder_kind = request.decoder or self.config.ltx25_decoder
        else:
            decoder_kind = request.decoder or "vae"
        if request.mode in {"refine_image", "ref2i"} and decoder_kind == "diffusion":
            # Probe result: image-conditioned latents through the diffusion decoder blur.
            print(
                f"[ltx25] {request.mode}: diffusion decoder is not supported for "
                "image-conditioned still output (probe: blurred results); falling back to the VAE decoder",
                flush=True,
            )
            decoder_kind = "vae"

        # Sharpness A/B (Q5): for pure text-to-still, a clarity suffix measurably
        # improves fine detail at zero cost. Image-conditioned stills keep the
        # user's prompt untouched (untested there; fidelity to the input matters more).
        still_prompt = request.prompt
        still_negative = request.negative_prompt
        if request.mode == "t2i":
            still_prompt = f"{still_prompt.rstrip()} sharp focus, crisp fine detail, high clarity, minimal haze."
            extra_neg = "soft focus, hazy, bloom"
            still_negative = f"{still_negative}, {extra_neg}" if still_negative else extra_neg

        def progress_callback(offset: float, span: float, step_count: int):
            def on_step(_pipe, step: int, _timestep, callback_kwargs):
                progress(min(offset + ((step + 1) / step_count) * span, 0.96))
                return callback_kwargs

            return on_step

        if request.mode == "ref2i":
            # Probe P7a/P7b: single-stage base 30-step schedule, tiled VAE decode.
            with torch.no_grad():
                video_np, _audio = pipe(
                    prompt=still_prompt,
                    negative_prompt=still_negative,
                    conditions=conditions,
                    width=request.width,
                    height=request.height,
                    num_frames=request.num_frames,
                    frame_rate=request.fps,
                    num_inference_steps=request.steps,
                    guidance_scale=request.guidance_scale,
                    stg_scale=1.0,
                    modality_scale=3.0,
                    audio_guidance_scale=7.0,
                    audio_stg_scale=1.0,
                    audio_modality_scale=3.0,
                    enable_prompt_enhancement=request.enhance_prompt,
                    generator=generator,
                    output_type="np",
                    return_dict=False,
                    callback_on_step_end=progress_callback(0.0, 0.9, request.steps),
                )
            frames = video_np[0]
            frame_index = frames.shape[0] - 1 if request.frame_position == "last" else frames.shape[0] // 2
        else:
            # t2i / refine_image: distilled two-stage
            # (8 sigmas -> 2x latent upsample -> 3-sigma refine), probes P3 / P6.
            num_frames = request.num_frames or 9
            use_diffusion_decoder = request.mode == "t2i" and decoder_kind == "diffusion"
            if use_diffusion_decoder and num_frames < 17:
                # NATTEN na3d needs >= its (11,11,11) kernel per tile: nf=9 fails, nf=17 works
                # (probes P4 vs P4_nf17). Promote internally.
                print(
                    f"[ltx25] t2i: diffusion decoder requires num_frames>=17 "
                    f"(NATTEN kernel size); promoting num_frames {num_frames} -> 17",
                    flush=True,
                )
                num_frames = 17
            with torch.no_grad():
                video, audio = pipe(
                    prompt=still_prompt,
                    negative_prompt=still_negative,
                    conditions=conditions or None,
                    width=request.width,
                    height=request.height,
                    num_frames=num_frames,
                    frame_rate=request.fps,
                    sigmas=DISTILLED_SIGMA_VALUES,
                    guidance_scale=1.0,
                    audio_guidance_scale=1.0,
                    stg_scale=0.0,
                    audio_stg_scale=0.0,
                    modality_scale=1.0,
                    audio_modality_scale=1.0,
                    enable_prompt_enhancement=request.enhance_prompt,
                    generator=generator,
                    output_type="latent",
                    return_dict=False,
                    callback_on_step_end=progress_callback(0.0, 0.45, len(DISTILLED_SIGMA_VALUES)),
                )
                progress(0.5)
                video = self._upsample_pipe(
                    latents=video,
                    height=request.height,
                    width=request.width,
                    num_frames=num_frames,
                    output_type="latent",
                    return_dict=False,
                )[0]
                video, audio = pipe(
                    prompt=still_prompt,
                    negative_prompt=still_negative,
                    latents=video,
                    audio_latents=audio,
                    width=request.width * 2,
                    height=request.height * 2,
                    num_frames=num_frames,
                    frame_rate=request.fps,
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
                        0.52, 0.3, len(STAGE_2_DISTILLED_SIGMA_VALUES)
                    ),
                )
                if use_diffusion_decoder:
                    progress(0.85)
                    decode_pipe = self.load_diffusion_decoder()
                    self._configure_decode_tiling(
                        num_frames, request.height * 2, request.width * 2
                    )
                    decode_generator = torch.Generator(device="cpu").manual_seed(request.seed)
                    video = decode_pipe(
                        latents=video.to("cuda"),
                        generator=decode_generator,
                        output_type="np",
                        return_dict=False,
                        denormalize=False,  # latent-path outputs are already denormalized
                    )[0]
            frames = video[0]
            frame_index = frames.shape[0] // 2

        frame = (np.clip(frames[frame_index], 0, 1) * 255).round().astype("uint8")
        target.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(frame).save(target)
        print(
            f"[ltx25] {request.mode}: saved frame {frame_index}/{frames.shape[0]} "
            f"({frame.shape[1]}x{frame.shape[0]}) -> {target.name}",
            flush=True,
        )
        progress(1.0)
        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
        gc.collect()
        torch.cuda.empty_cache()
        return {"peak_vram_gb": peak_vram_gb}

    def _generate_impl(
        self,
        request: GenerateRequest,
        target: Path,
        progress: Callable[[float], None],
        patches: _PatchRegistry,
    ) -> dict[str, float]:
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
        restore_audio_scheduler = None

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
            if request.mode in {"retake", "extend"} and condition.kind == "video":
                source_edit_path = source
                source_frames = load_video(str(source))
                continue
            # extend の画像条件はキーフレームアンカー(schemas.py の extend 分岐参照)。
            # 通常の LTX2VideoCondition として下の共通経路へ流し、prepare_extend 側で
            # context ロックとマージする(index は生成窓の latent インデックス)。
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

            _, restore_prepare_latents = patches.replace(
                pipe, "prepare_latents", types.MethodType(prepare_retake, pipe)
            )

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

            # 画像キーフレームアンカーの位置検証(2026-09-05): index は生成窓の
            # latent インデックス。context 区間(latent 0..context_end)への指定は
            # ロック済み領域と衝突するため拒否し、窓外も黙殺(pipeline 側は warning
            # でスキップする)ではなく明示エラーにする。
            last_latent = (effective_num_frames - 1) // 8
            context_end_latent = context_intervals // 8 if request.extend_direction == "end" else -1
            for _c in request.conditions:
                if _c.kind != "image":
                    continue
                resolved_idx = _c.index if _c.index >= 0 else last_latent
                if resolved_idx > last_latent:
                    raise ValueError(
                        f"extend image keyframe latent index {_c.index} exceeds the "
                        f"generation window (last latent {last_latent})"
                    )
                if request.extend_direction == "end" and resolved_idx <= context_end_latent:
                    raise ValueError(
                        f"extend image keyframe latent index {_c.index} falls inside the "
                        f"locked context region (latents 0..{context_end_latent}); use a "
                        f"larger index or -1"
                    )

            pixels = pipe.video_processor.preprocess_video(
                context_frames, height=effective_height, width=effective_width
            ).to(device=pipe._execution_device, dtype=pipe.vae.dtype)
            from diffusers.pipelines.ltx2.pipeline_ltx2_condition import retrieve_latents
            with torch.no_grad():
                context_latents = retrieve_latents(pipe.vae.encode(pixels), sample_mode="argmax")
            original_prepare_latents = pipe.prepare_latents

            def prepare_extend(this, *args, **kwargs):
                # 元の prepare_latents は画像キーフレーム条件を処理済みの
                # (latents, mask, clean, coords) を返す(キーフレームは base 系列の
                # 後ろに追加トークンとして連結される)。旧実装は mask/clean を
                # 丸ごと自前のものに差し替えていたため条件が無効化されていた。
                # 2026-09-05: base 区間(先頭 base_len トークン)だけ context ロックを
                # 適用し、キーフレーム由来の mask/clean/coords は温存するマージ方式へ
                # 変更(キーフレーム無しなら従来と同値)。
                latents, orig_mask, orig_clean, coords = original_prepare_latents(*args, **kwargs)
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
                base_len = mask.shape[1]
                latents[:, :base_len] = latents[:, :base_len] * (1 - mask) + clean * mask
                merged_mask = orig_mask.clone()
                merged_mask[:, :base_len] = torch.maximum(orig_mask[:, :base_len], mask)
                merged_clean = orig_clean.clone()
                merged_clean[:, :base_len] = orig_clean[:, :base_len] * (1 - mask) + clean * mask
                return latents, merged_mask, merged_clean, coords

            _, restore_prepare_latents = patches.replace(
                pipe, "prepare_latents", types.MethodType(prepare_extend, pipe)
            )

        # LTX25_STAGE_DEBUG=1: リアルタイム経路の未計測区間を分解する一時計測
        # (2026-09-03 調査、既定OFFで挙動不変)。a2v 前処理は ffprobe + ffmpeg×2 の
        # subprocess 3回 + MelSpectrogram 毎回構築 + audio_vae.encode を含む。
        _stage_debug = os.getenv("LTX25_STAGE_DEBUG", "0").strip() == "1"
        _a2v_prep_t0 = time.time()
        if request.mode == "a2v":
            matches = list(input_dir.glob(f"{request.audio_asset_id}.*"))
            if len(matches) != 1:
                raise ValueError("Audio input asset not found")
            audio_source = matches[0]
            # 高速化②-1: 尺の取得は torchaudio.info(メタデータのみ・subprocess なし)を
            # 優先し、読めないコンテナだけ従来の ffprobe(プロセス起動 ~40ms)へ落とす。
            # デコード本体は従来どおり ffmpeg のまま(リサンプラを替えると a2v の
            # 条件付け mel が数値的に変わるため、意図的に触らない)。
            # 注意: torchaudio.info はこの venv の torchaudio には存在しない
            # (AttributeError、2026-09-03 実機確認)。PCM wav なら標準 wave モジュールで
            # ヘッダから正確な尺が取れる(依存ゼロ・subprocess ゼロ)。
            total_duration = None
            if audio_source.suffix.lower() == ".wav":
                try:
                    import wave as _wave
                    with _wave.open(str(audio_source), "rb") as _w:
                        _n, _sr = _w.getnframes(), _w.getframerate()
                    if _n > 0 and _sr > 0:
                        total_duration = _n / float(_sr)
                except Exception:
                    total_duration = None
            if total_duration is None:
                probe = subprocess.run(
                    ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", str(audio_source)],
                    capture_output=True, text=True,
                )
                if probe.returncode != 0:
                    raise ValueError("Input audio duration could not be read")
                total_duration = float(probe.stdout.strip())
            remaining = total_duration - request.audio_start
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
            # 高速化②-2: MelSpectrogram はパラメータ固定なのでプロセス内で1回だけ
            # 構築してデバイス別にキャッシュする(毎リクエストの module 構築+
            # フィルタバンク計算+.to(device) を省く。数値は同一 module の再利用なので不変)。
            mel_transform = self._get_mel_transform_16k(waveform.device)
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
            _, restore_audio_prepare = patches.replace(
                pipe, "prepare_audio_latents", types.MethodType(prepare_frozen_audio, pipe)
            )
            _, restore_audio_scheduler = patches.replace(pipe, "audio_scheduler", frozen_scheduler)
            if _stage_debug:
                print(f"[ltx25] STAGE_DEBUG a2v_prep {time.time() - _a2v_prep_t0:.3f}s", flush=True)

        decoder_kind = request.decoder or self.config.ltx25_decoder
        use_refine = request.upscale or request.temporal_upscale
        use_diffusion_decoder = use_refine and decoder_kind == "diffusion"
        final_fps = effective_fps * (2 if request.temporal_upscale else 1)

        stage1_sigmas = _subsample_distilled_sigmas(DISTILLED_SIGMA_VALUES, request.steps)
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
            # 蒸留8σスケジュール。steps<8 を明示されたときだけ間引く(2026-09-04、
            # リアルタイム高速化)。従来 request.steps はこの経路で黙殺されており
            # 「4steps 指定」は効いていなかった。間引きは先頭・末尾を必ず含む
            # 等間隔インデックス(例 4steps: idx 0,2,5,7)。蒸留の学習分布外に
            # なるため品質は A/B 前提。steps>=8・未指定は従来どおり8σ固定。
            "sigmas": stage1_sigmas,
            "guidance_scale": 1.0,
            # リップシンク調整ノブ(schemas.GenerateRequest の同名フィールド参照)。
            # None(既定)なら従来どおり 1.0 固定 = 完全互換。
            "audio_guidance_scale": (
                request.audio_guidance_scale if request.audio_guidance_scale is not None else 1.0
            ),
            "stg_scale": 0.0,
            "audio_stg_scale": 0.0,
            # 映像側 modality_scale がリップシンクの実効ノブ(schemas の訂正コメント参照)
            "modality_scale": (
                request.modality_scale if request.modality_scale is not None else 1.0
            ),
            "audio_modality_scale": (
                request.audio_modality_scale if request.audio_modality_scale is not None else 1.0
            ),
            "enable_prompt_enhancement": request.enhance_prompt,
            "generator": generator,
            # 非refine(リアルタイム)経路は "pt" で受ける(2026-09-03 高速化①):
            # postprocess 済み (B,F,C,H,W)・GPU 上の float [0,1] が返るので、
            # uint8 化を GPU で行ってから CPU へ下ろす(従来の "np" は float32
            # ~190MB を CPU へ転送して numpy で clip/mul/round していた。実測 0.20s)。
            "output_type": "latent" if use_refine else "pt",
            "return_dict": False,
            "callback_on_step_end": progress_callback(
                0.0, 0.55 if use_refine else 0.96, len(stage1_sigmas)
            ),
        }
        if request.mode == "a2v":
            args["audio_latents"] = input_audio_latents
        # LTX25_STAGE_DEBUG=1: pipe() 内部のコンポーネント別時間を計測する一時フック
        # (text_encoder / video VAE decode / audio VAE decode / vocoder)。
        # instance 属性で forward/decode を差し替え、finally で必ず原状復帰する。
        _dbg_acc: dict = {}
        _dbg_restore: list = []
        if _stage_debug:
            def _dbg_wrap(obj, attr, name):
                orig = getattr(obj, attr)
                def timed(*a, **k):
                    t0 = time.time()
                    try:
                        return orig(*a, **k)
                    finally:
                        _dbg_acc[name] = _dbg_acc.get(name, 0.0) + (time.time() - t0)
                setattr(obj, attr, timed)
                _dbg_restore.append((obj, attr, orig))
            _dbg_wrap(pipe.text_encoder, "forward", "text_encode")
            _dbg_wrap(pipe.vae, "decode", "video_vae_decode")
            _dbg_wrap(pipe.audio_vae, "decode", "audio_vae_decode")
            _dbg_wrap(pipe.vocoder, "forward", "vocoder")
        stage_t0 = time.time()
        try:
            video, audio = pipe(**args)
        finally:
            if restore_prepare_latents is not None:
                restore_prepare_latents()
            for _obj, _attr, _orig in _dbg_restore:
                setattr(_obj, _attr, _orig)
        if _stage_debug:
            _parts = " ".join(f"{k}={v:.3f}s" for k, v in _dbg_acc.items())
            print(
                f"[ltx25] STAGE_DEBUG pipe_total {time.time() - stage_t0:.3f}s ({_parts})",
                flush=True,
            )
        if not use_refine:
            # 高速化①: uint8 変換を GPU で実行(output_type="pt" とセット)。
            # (B,F,C,H,W) float [0,1] → (B,F,H,W,3) uint8 numpy。値は従来の
            # np.clip*255→round(半数偶数丸め)→astype と bit 一致する
            # (torch.round も half-to-even、float32 演算は IEEE で同一)。
            # 以降の video の使われ方(video[0] を encode へ)は従来の "np" と
            # 同じレイアウトなので下流は無変更。encode_video_crf 側の uint8
            # 変換は dtype==uint8 のため素通りになる。
            _cvt_t0 = time.time()
            with torch.no_grad():
                # .float() が必須: "pt" は VAE 出力の dtype(bf16)のまま返るが、
                # 従来の "np" 経路は numpy 変換時に float32 へキャストしてから
                # *255/round していた。bf16 のまま演算すると丸めが変わり
                # framemd5 が一致しない(実測で確認)。float32 に揃えると
                # 旧経路と同一の IEEE 演算列になる。
                video = (
                    video.permute(0, 1, 3, 4, 2)
                    .float()
                    .clamp(0.0, 1.0)
                    .mul(255.0)
                    .round()
                    .to(torch.uint8)
                    .cpu()
                    .numpy()
                )
            if _stage_debug:
                print(f"[ltx25] STAGE_DEBUG gpu_uint8_convert {time.time() - _cvt_t0:.3f}s", flush=True)
        generated_num_frames = effective_num_frames
        if generated_num_frames is None:
            # Auto-duration returns unpacked video latents [B, C, latent_F, H, W].
            generated_num_frames = (video.shape[2] - 1) * pipe.vae_temporal_compression_ratio + 1
        if use_refine:
            print(f"[ltx25] stage timing: base denoise {time.time() - stage_t0:.1f}s", flush=True)
            stage_t0 = time.time()
            pixel_reference_latents = video.detach().clone() if request.upscale_method == "pixel" else None
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
            print(f"[ltx25] stage timing: latent upsample {time.time() - stage_t0:.1f}s", flush=True)
            stage_t0 = time.time()
            stage2_span = 0.14 if use_diffusion_decoder else 0.32
            restore_stage2_prepare = None
            if request.upscale_method == "pixel":
                pixel_lora = (
                    self.config.quantized_model_dir.resolve()
                    / "pixel_spatial_upscaler"
                    / PIXEL_UPSCALER_FILENAME
                )
                if not pixel_lora.is_file():
                    raise RuntimeError(
                        "Pixel Spatial Upscaler IC-LoRA is missing. Run "
                        "scripts/download_quantize_ltx25.py --component pixel_upscaler first."
                    )
                try:
                    if self.config.ltx25_transformer_precision == "nvfp4":
                        from .acceleration.nvfp4 import install_nvfp4_lora

                        count = install_nvfp4_lora(
                            pipe.transformer,
                            pixel_lora,
                            "pixel_spatial_upscaler",
                            PIXEL_DETAILING_LORA_STRENGTH,
                        )
                        print(
                            f"[ltx25] NVFP4 Pixel IC-LoRA: attached to {count} layers "
                            f"at strength {PIXEL_DETAILING_LORA_STRENGTH}",
                            flush=True,
                        )
                    else:
                        pipe.load_lora_weights(pixel_lora, adapter_name="pixel_spatial_upscaler")
                        names = [f"job_lora_{index}" for index in range(len(request.loras))]
                        pipe.set_adapters(
                            [*names, "pixel_spatial_upscaler"],
                            adapter_weights=[
                                *[item.strength for item in request.loras],
                                PIXEL_DETAILING_LORA_STRENGTH,
                            ],
                        )
                        self._cast_lora_layers_to_bf16(pipe)
                except Exception as exc:
                    raise RuntimeError(f"Pixel Spatial Upscaler IC-LoRA could not be loaded: {exc}") from exc

                # Diffusers does not yet expose VideoConditionByReferenceLatent. Append the
                # clean half-resolution Stage-1 latent exactly as the official LTX DFR
                # pipeline does, and scale its spatial RoPE coordinates into the target grid.
                original_prepare_latents = pipe.prepare_latents

                def prepare_pixel_reference(this, *args, **kwargs):
                    latents, mask, clean, coords = original_prepare_latents(*args, **kwargs)
                    reference = this._normalize_latents(
                        pixel_reference_latents,
                        this.vae.latents_mean,
                        this.vae.latents_std,
                        this.vae.config.scaling_factor,
                    ).to(device=latents.device, dtype=latents.dtype)
                    reference_tokens = this._pack_latents(
                        reference,
                        this.transformer_spatial_patch_size,
                        this.transformer_temporal_patch_size,
                    ).expand(latents.shape[0], -1, -1)
                    reference_mask = torch.ones(
                        (*reference_tokens.shape[:2], 1), device=latents.device, dtype=mask.dtype
                    )
                    reference_coords = this.transformer.rope.prepare_video_coords(
                        latents.shape[0],
                        reference.shape[2],
                        reference.shape[3],
                        reference.shape[4],
                        latents.device,
                        fps=final_fps,
                    )
                    reference_coords[:, 1:, :, :] *= 2
                    combined_coords = (
                        reference_coords if coords is None else torch.cat([coords, reference_coords], dim=2)
                    )
                    return (
                        torch.cat([latents, torch.zeros_like(reference_tokens)], dim=1),
                        torch.cat([mask, reference_mask], dim=1),
                        torch.cat([clean, reference_tokens], dim=1),
                        combined_coords,
                    )

                _, restore_stage2_prepare = patches.replace(
                    pipe, "prepare_latents", types.MethodType(prepare_pixel_reference, pipe)
                )
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
            finally:
                if restore_stage2_prepare is not None:
                    restore_stage2_prepare()
            print(f"[ltx25] stage timing: stage2 refine {time.time() - stage_t0:.1f}s", flush=True)
        if restore_audio_prepare is not None:
            restore_audio_prepare()
        if restore_audio_scheduler is not None:
            restore_audio_scheduler()
        sample_rate = pipe.vocoder.config.output_sampling_rate
        if use_diffusion_decoder:
            # `output_type="latent"` returned de-normalized video latents and audio latents.
            audio_wave = self._decode_audio(pipe, audio)[0].float().cpu()
            progress(0.80)
            decode_pipe = self.load_diffusion_decoder()
            self._configure_decode_tiling(
                generated_num_frames,
                effective_height * (2 if request.upscale else 1),
                effective_width * (2 if request.upscale else 1),
            )
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

        def _encode_and_finalize():
            encode_t0 = time.time()
            encode_video_crf(
                video[0],
                fps=final_fps,
                audio=audio_wave,
                audio_sample_rate=sample_rate,
                output_path=encode_target,
                crf=self.config.ltx25_video_crf,
                encoder=self.config.ltx25_video_encoder,
                nvenc_preset=self.config.ltx25_nvenc_preset,
            )
            print(f"[ltx25] stage timing: mp4 encode {time.time() - encode_t0:.1f}s", flush=True)
            if request.mode == "retake":
                self._finish_retake(source_edit_path, encode_target, target, request)
                encode_target.unlink(missing_ok=True)
            elif request.mode == "extend":
                self._finish_extend(
                    source_edit_path, encode_target, target, request.extend_direction,
                    extend_context_duration, extend_duration,
                )
                encode_target.unlink(missing_ok=True)

        peak_vram_gb = torch.cuda.max_memory_allocated() / (1024**3)
        # NVENC を開く**前**に torch の予約
        # キャッシュを返す(2026-09-06)。従来はエンコード後にしか empty_cache して
        # おらず、直前の大きいジョブでキャッシュが育っていると NVENC の
        # avcodec_open2 が VRAM を確保できず失敗した(実機: 1024×576×361f の
        # i2v 連発後の extend で "avcodec_open2(h264_nvenc)" エラー)。
        gc.collect()
        torch.cuda.empty_cache()
        _encode_and_finalize()
        progress(1.0)
        gc.collect()
        torch.cuda.empty_cache()
        return {"peak_vram_gb": peak_vram_gb}
