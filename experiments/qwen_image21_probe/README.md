# Qwen-Image 2.1 Probe

Isolated feasibility benchmark for running `Qwen/Qwen-Image-2.1` beside the existing resident LTX-2.5 worker.

This experiment intentionally does **not** modify the production LTX runtime or its pinned Diffusers dependency.

## Baseline

- GPU: RTX PRO 6000 96 GB
- Qwen model revision: `b3179ad355be050328e483a9dfdd9e60cd62adfa`
- Diffusers commit: `80c7ed262aeffbeb43ef13ae04baeb9b84515a69`
- dtype: BF16
- 40 steps
- `true_cfg_scale=1.0`
- `use_kv_cache=true`
- no GPU Memory Snapshot
- one GPU workload at a time

## Modal benchmark

```powershell
.venv\Scripts\modal.exe run experiments/qwen_image21_probe/modal_probe.py
```

The benchmark reports:

- cold model load time
- resident idle allocated/reserved VRAM
- 1024x1024, 1536x1024, 2048x2048 latency
- peak allocated/reserved VRAM
- final resident idle VRAM after generations

The first run downloads model files into the dedicated persistent `qwen-image21-cache` volume.
