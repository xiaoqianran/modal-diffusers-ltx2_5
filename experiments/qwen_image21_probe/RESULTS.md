# Qwen-Image 2.1 BF16 benchmark — RTX PRO 6000 96 GB

Measured on Modal using `NVIDIA RTX PRO 6000 Blackwell Server Edition`.

Configuration:

- `Qwen/Qwen-Image-2.1`
- BF16
- Diffusers main (`0.41.0.dev0`, commit `80c7ed262aeffbeb43ef13ae04baeb9b84515a69`)
- Transformers 5.17.0
- PyTorch 2.13.0 + CUDA 13.0
- Torchvision 0.28.0
- 40 steps
- `true_cfg_scale=1.0`
- `use_kv_cache=true`
- seed 42
- no GPU Memory Snapshot

## Cold / resident

| Metric | Result |
|---|---:|
| Cached model load | 10.279 s |
| Resident allocated | 30.229 GiB |
| Resident reserved | 30.406 GiB |
| Visible VRAM | 94.971 GiB |

## Generation

| Resolution | Total | Peak allocated | Peak reserved |
|---|---:|---:|---:|
| 1024×1024 | 11.038 s | 36.849 GiB | 38.770 GiB |
| 1536×1024 | 15.933 s | 40.130 GiB | 42.246 GiB |
| 2048×2048 | 53.027 s | 56.529 GiB | 63.930 GiB |

After all runs the pipeline returned to about 30.265 GiB allocated / 30.439 GiB reserved.

## Dual-resident implication

Qwen-Image 2.1 itself is a strong candidate for permanent residency beside LTX-2.5.

For 1024² and 1536×1024, Qwen activation headroom over its resident baseline is modest:

- 1024²: about +6.62 GiB allocated over resident
- 1536×1024: about +9.90 GiB allocated over resident
- 2048²: about +26.30 GiB allocated over resident

Therefore the Director Runtime should initially target 1024-class and 1536×1024-class Qwen generation while both models remain resident. 2048² should be treated as a high-memory mode until the exact current LTX resident allocation is re-measured inside the same container.
