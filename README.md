# LTX-2.5 Diffusers Server + Web UI

> 当前仓库结构与主链路请先看 `docs/ARCHITECTURE.md`。主路径是 `frontend -> ltx25/api.py -> ltx25/modal_client.py -> modal_app.py -> ltx25/runtime.py`。媒体 Volume / S3-compatible 直传配置见 `docs/MEDIA_STORAGE.md`。

[English](README_EN.md) | 日本語

`Lightricks/LTX-2.5-Diffusers`をModal GPUで実行し、ローカルFastAPIルーターと独立したVite UIから操作する生成アプリです。T2AV、I2V、先頭／末尾フレーム指定（FLF2V）、任意の画像／動画条件に対応します。既定の高品質モードは、初段の潜在出力を2倍アップサンプルし、追加の3-stepで精細化します。

## UIサンプル

![LTX-2.5 StudioのWeb UI](docs/ui-sample.png)

## 事前準備

- NVIDIA GPU、対応するドライバー、十分なRAM/VRAM
- Python 3.11以上、またはDocker + NVIDIA Container Toolkit
- `ffmpeg`
- Hugging Faceで[モデルページ](https://huggingface.co/Lightricks/LTX-2.5-Diffusers)の利用条件に同意したアカウントのRead token

このモデルは約190億パラメータで、リポジトリ全体の保存容量も非常に大きいため、初回ダウンロードには時間とディスク空き容量が必要です。既定の`model`オフロードはVRAMを抑える代わりに推論が遅くなります。

## 順次ダウンロード・NF4量子化

巨大コンポーネントを同時に保持しないスクリプトを先に実行します。各コンポーネントは専用キャッシュへ取得され、NF4保存物または追加チェックポイントの再ロード検証に成功した場合だけ、そのキャッシュが削除されます。中断後は同じコマンドで再開できます。LTX-2.5本体に加えて、[Pixel Spatial Upscaler IC-LoRA](https://huggingface.co/Lightricks/LTX-2.5-22b-IC-LoRA-Pixel-Spatial-Upscaler)の利用条件にも事前に同意してください。

```bash
docker build -t ltx25-server .
docker run --rm --gpus 'device=0' \
  -v "$PWD:/app" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface:ro" \
  -w /app ltx25-server \
  python scripts/download_quantize_ltx25.py
```

既定では処理開始時の空きが80GiBを下回ると停止します。高品質生成用の潜在アップサンプラーと公式Pixel Spatial Upscaler IC-LoRAを取得しますが、`transformer_full`、任意のユーザーLoRA、diffusion decoderは取得しません。既存環境へPixel IC-LoRAだけ追加する場合は`python scripts/download_quantize_ltx25.py --component pixel_upscaler`を実行します。

## Modal / RTX PRO 6000 NVFP4

`modal_app.py` 将准备阶段和 GPU 推理解耦：模型下载在 CPU Function 中完成并写入
`ltx25-models` Volume；Qwen-Image 2.1 复用独立的 `qwen-image21-cache` Volume。RTX PRO 6000
只读挂载两者，在容器启动时完成 LTX-2.5 NVFP4 与 Qwen-Image 2.1 BF16 双模型装配。GPU 侧不会下载模型权重或 diffusion decoder。

```bash
pip install -r requirements-modal.txt
modal secret create huggingface HF_TOKEN=hf_...

# CPU only: 下载/校验 LTX base/text encoder/upsamplers/decoder/NVFP4；Qwen 使用已有 cache Volume
modal run modal_app.py::prepare_models

# 部署；只有 GPU 服务容器实际启动时才申请 RTX PRO 6000
modal deploy modal_app.py
```

生产 Director 预设固定为 `LTX25_TRANSFORMER_PRECISION=nvfp4`、`OFFLOAD_MODE=none`，并默认
`DIRECTOR_QWEN_ENABLED=1`。实机双模型测试后默认启用 `LTX25_CUDA_GRAPH=1`，但双常驻模式将
`LTX25_CUDA_GRAPH_MAX_CAPTURES=1`，只保留一个 LTX shape 的 graph，以限制显存常驻增长。模型 Volume 在 GPU 容器中以只读方式挂载，输入、输出、LoRA
放在 `ltx25-state` Volume；NATTEN 等硬件相关 kernel 缓存单独放在 `ltx25-kernels`
Volume，避免已加载的 `.so` 阻塞逐任务 `state_volume.reload()`。任务状态保存在
`ltx25-jobs` Modal Dict。
本地 `ltx25/modal_client.py` 通过 `.spawn()` 提交任务，GPU 每个容器串行推理，默认最多一个 GPU 容器。
生产 Director 固定 `min_containers=1`，因此部署后会持续保留一个 RTX PRO 6000 容器，
直到显式执行 `delete-modal.bat` / `modal app stop`。Studio 的 warm lease 只负责
ready/warming 状态探测，不会覆盖这一生产常驻策略。部署、Lookup、Volume/Dict/Secret
统一使用 `LTX25_MODAL_ENVIRONMENT`（默认 `main`），避免本机 active environment 变化后
出现 lookup 到错误环境。开发部署固定使用 Modal `recreate` strategy；对于当前
`max_containers=1` 的单 GPU Director，这保证 deploy 完成后新请求不会继续进入旧容器。
模型/状态/kernel Volume 与 HF Secret 名称分别可通过
`LTX25_MODAL_MODEL_VOLUME`、`LTX25_MODAL_STATE_VOLUME`、`LTX25_MODAL_KERNEL_VOLUME`、
`LTX25_MODAL_HF_SECRET` 覆盖。
导演台请求支持 `engine=auto|ltx|qwen`。`auto` 会把纯 `t2i` 路由到 Qwen-Image 2.1，
视频、参考/编辑和 LoRA 条件任务仍走 LTX；显式 `ltx/qwen` 始终优先于自动策略。每个请求会先被纯函数
编译为 `ExecutionPlan`，任务结果返回 `plan`（resolved engine / reason / resource class / acceleration）以及
LTX 的实际 `graph` capture/replay counters，因此自动决策不会成为黑盒。

### 独立前端

前端已经与 Modal/FastAPI 后端分离，位于 `frontend/`，使用 Vue 3 + Vite。
后端只提供 `/api/*` 与 `/outputs/*`，不再托管静态页面。

```powershell
cd frontend
npm install
npm run dev
```

打开 `http://127.0.0.1:5187`。单独执行 `npm run dev` 时，Vite 默认代理到
`http://127.0.0.1:8000`，可配合 `python tools/mock_backend.py --port 8000` 做纯前端开发；
`start-ltx25.bat` 会显式把 `VITE_API_TARGET` 指向真实本地 Router `http://127.0.0.1:48125`。
如需切换后端，设置 `VITE_API_TARGET` 后重新启动前端。前端可以连续提交任务，后端仍保持单 GPU
串行执行（`max_containers=1`、`max_inputs=1`），后续任务在 Modal 队列中等待。

生产部署使用 `min_containers=1`，因此 `LTX25_MODAL_GPU_IDLE_SECONDS` 不会让唯一的
Director 容器自动缩到 0；如需释放 GPU，请显式停止 App。若未来切回
`min_containers=0`，该 idle window 才重新决定空闲容器的回收时机。

`ltx25-jobs` 使用 Modal Dict 保存运行期 job/session/asset 元数据。Modal Dict 条目不是
永久历史数据库；长期归档应写入独立持久层，而 Studio 的 History 仅应视为近期运行历史。

网关在同一容器内串行化 Volume 操作（包括完整文件响应），避免打开文件期间
执行 `reload()`；健康检查和任务轮询仍可并发。大文件传输期间，同容器的其他
文件操作会等待。任务状态仍存 Modal Dict，但当前会话使用 `session:<id>:jobs`
轻量索引，常规轮询不再扫描整个 Dict；旧会话首次访问会自动回填索引。

> NATTEN 是硬件/torch/CUDA 组合相关的预编译 kernel，由 `kernels` 在 GPU 环境中选择；
> 它不是 LTX 模型权重。其首次 kernel 获取发生在 GPU 容器中，缓存持久化到
> `ltx25-kernels`；只对 `shi-labs/natten` 开启 repo-scoped remote-code allowlist。

## Python環境で起動

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements-local.txt
cp .env.example .env
# .env の HF_TOKEN を設定
python -m uvicorn ltx25.api:app --host 127.0.0.1 --port 48125
```

API仕様は <http://127.0.0.1:48125/docs> で確認できます。

## 生成モード

- `t2av`: テキストから音声付き動画
- `i2v`: 1枚の先頭画像から音声付き動画
- `flf2v`: 先頭画像と末尾画像を両方固定した補間動画
- `condition`: 最大8個の画像／動画を任意の潜在フレーム位置と強度で指定
- `t2i`: プロンプトから静止画1枚（蒸留2段生成 → 2倍解像度PNG）
- `refine_image`: 入力画像の2倍解像度再解釈／バリエーション静止画
- `ref2i`: 参照画像の同一性を保った新しい場面の静止画

Web UIでは生成方式とレンダリングを個別に指定します。横長は768×512、768×448、960×544、1280×704、1920×1088、縦長はそれらを転置した512×768、448×768、544×960、704×1280、1088×1920、正方形は512×512を選択できます。`2倍高解像度化`はT2AV、I2V、FLF2V、Reference条件で利用でき、最大960×544（縦長は544×960）の基準サイズから1920×1088（1088×1920）を出力します。方式は、従来のlatent 2倍アップスケールと3-step refineを行う`Latent Upscale`と、初段の低解像度映像を参照latentとして公式IC-LoRAで細部を再生成する`Pixel IC-LoRA`から選択します。Pixel方式は構図・動き・被写体を参照しますが、存在しない高周波ディテールを創作するためピクセル忠実な拡大ではありません。APIでは`upscale_method=latent/pixel`を指定します。1280×704以上のプリセットはVRAM消費が大きい直接生成専用で、選択すると2倍高解像度化は解除されます。OFFの場合は8-step単段生成です。I2V/FLF2Vの先頭画像を選ぶと、Web UIが画像の縦横比に合わせて標準基準サイズを自動選択します。APIの`quality=high/draft`は後方互換用に残っていますが、新規クライアントは`upscale=true/false`を使用してください。

`2倍フレームレート化`は独立して選択できます。Temporal Latent Upscale と3-step refineにより、121フレーム/24fpsを241フレーム/48fpsへ変換し、動画と音声の尺は維持します。APIでは`temporal_upscale=true`を指定します。空間・時間の両方を選んだ場合もrefineは1回です。初回セットアップ済み環境では `scripts/download_quantize_ltx25.py --component temporal` を一度実行してください。

`Retake`では元動画と開始・終了秒を指定し、選択した時間領域だけをlatent上で再生成します。元動画のFPSと解像度は自動取得され、範囲外の映像・音声は維持されます。映像のみ、音声のみ、または両方の再生成を選択できます。元動画は8n+1フレームかつ縦横32の倍数である必要があります。

`Extend`では元動画の先頭または末尾へ1〜20秒の新しい映像・音声を追加します。参照範囲をclean latentのprefix/suffixとして固定するため、境界の被写体・動作・構図を引き継ぎます。元動画のFPSと解像度は自動採用され、参照範囲と延長範囲は8フレーム単位へ調整されます。1回の生成に使う参照＋延長は最大481フレームです。

### 静止画モード（t2i / refine_image / ref2i）

動画パイプラインを流用して静止画PNGを生成する3モードです（レシピは`scratch_t2i_probe/`の実証プローブに基づく）。出力は`outputs/t2i_*.png`／`refine_*.png`／`ref2i_*.png`へ保存され、履歴DBには`image_url`として記録されます（MP4は作成しません）。LoRA合成・シード・進捗・キューは動画モードと同様に機能します。

- **`t2i`**: `num_frames=9`固定・蒸留2段（8σ → 2x latent upsample → 3σ refine）・VAEデコードで、基準解像度の2倍（既定512²→1024²）の中央フレームをPNG化します。`decoder: "diffusion"`を指定した場合のみ、NATTENカーネルの最小サイズ制約（kernel 11×11×11 > 9フレーム）を満たすため内部で`num_frames`を17へ昇格し（ログに明示）、diffusion decoderでデコードします。
- **`refine_image`**: 入力画像（`/api/assets`で登録、`conditions`の`index: 0`に指定）を条件に、t2iと同じ2段レシピで2倍解像度に再解釈します。`strength`（0.1〜1.0、既定1.0）で参照の効きを調整します。デコードはVAE固定です。strength=1.0の実測で入力とのmean abs diffは約0.099（プローブP6と一致）。
- **`ref2i`**: 参照画像1〜複数（`conditions`スキーマ流用、各latent index／strength指定可）＋場面プロンプトから、30-step基本スケジュール（guidance 3.0）で短い内部動画を生成し、`frame_position: "last"`（既定）または`"center"`のフレームをPNG化します。`num_frames`は25/41/49（既定49、参照から離れた場面ほど大きい値が有効）。デコードはVAE固定で、`decoder: "diffusion"`を指定しても警告ログ付きでVAEへフォールバックします（画像条件×diffusion decoderはブラーする実証結果のため）。

```bash
# t2i（既定: 512²基準 → 1024² PNG）
curl -X POST http://127.0.0.1:48125/api/jobs -H 'content-type: application/json' \
  -d '{"mode":"t2i","prompt":"A photorealistic portrait, golden hour light","width":512,"height":512,"seed":42}'

# refine_image（入力画像の2x再解釈）
curl -X POST http://127.0.0.1:48125/api/jobs -H 'content-type: application/json' \
  -d "{\"mode\":\"refine_image\",\"prompt\":\"...\",\"width\":512,\"height\":512,\"strength\":1.0,\"conditions\":[{\"asset_id\":\"$ASSET_ID\",\"kind\":\"image\",\"index\":0}]}"

# ref2i（参照→新場面の静止画、末尾フレーム抽出）
curl -X POST http://127.0.0.1:48125/api/jobs -H 'content-type: application/json' \
  -d "{\"mode\":\"ref2i\",\"prompt\":\"The same woman, new scene...\",\"width\":512,\"height\":512,\"num_frames\":49,\"frame_position\":\"last\",\"conditions\":[{\"asset_id\":\"$ASSET_ID\",\"kind\":\"image\",\"index\":0,\"strength\":1.0}]}"
```

実測（RTX PRO 6000 Blackwell 96GB、`OFFLOAD_MODE=model`、512²基準、seed=42）: t2i 46.5s（初回モデルロード込み。ロード後の生成本体は約6〜8s）／ピークVRAM 17.9GB、t2i+diffusion decoder 約30s（ウォーム）、refine_image 30.0s（ウォーム）、ref2i nf=49 45.5s（ウォーム、denoise約34s）。Web UIでは3モードをモード選択から使え、結果はギャラリーにPNGタイルで表示（ダウンロード可）、「この画像を I2V/FLF2V の先頭画像に使う」ボタンで生成PNGをそのままI2Vの先頭画像欄へセットできます。

`Audio → Video`ではWAV/MP3/M4A/FLAC/OGG/AACをAudio VAEでlatent化し、音声モダリティを固定したまま映像だけを生成します。音声開始位置と最大20秒の使用時間を指定でき、任意の先頭画像も併用できます。出力にはVAE再構成音ではなく元の入力波形を使用します。

Web UIは自動尺（duration head）、Native Multishot用のショットエディタ、セッション番号付き生成履歴に対応します。履歴は既定で`outputs/history.sqlite3`に永続化されます。自動尺でReference条件を使う場合、出力尺が生成前に未確定のため、参照の配置位置は先頭（0%）または末尾（100%）に限定されます。

LoRAは`loras/`直下へ`.safetensors`を配置し、Web UIの「再読込」から選択します。1ジョブにつき最大4本を合成でき、各強度は-2.0〜2.0です。ジョブ終了時（失敗時を含む）にadapterをアンロードするため、次の生成へ設定は持ち越されません。LTX-2/LTX-2.5のDiffusers transformer互換LoRAを使用してください。旧LTX-Video用やComfyUI固有キーのLoRAは変換なしではロードできない場合があります。

`IC-LoRA / Reference`は通常のReference条件とは別の専用モードです。IC-LoRAを1本と参照シート画像または参照動画を指定し、参照latentを生成列へ追加トークンとして連結します。画像は出力フレーム数と同じ長さの静止参照動画へ内部変換されます。汎用経路はstage 1のみ、固定尺、推奨768×448です。HDRのscene embedding、DubItの音声参照など追加入力を要求する専用IC-LoRAには対応しません。

一覧APIはsafetensorsメタデータの`model_version`と`reference_downscale_factor`を読み取ります。現在の汎用IC-LoRAモードは参照縮小率1のみ対応し、それ以外はUIで生成前に拒否します。

`AIでLTX-2.5向けに変換`はOpenAI互換の`/chat/completions`を使用します。`.env`に`LLM_BASE_URL`と`LLM_MODEL`を設定し、必要な場合は`LLM_API_KEY`も設定してください。未設定でも通常の生成機能は利用できます。

Web UIでモードを選び、条件ファイルをアップロードして生成できます。APIでは先にアセットを登録し、返却された32桁の`id`を生成リクエストで使います。

```bash
# I2V
ASSET_ID=$(curl -s -F 'file=@first.png' http://127.0.0.1:48125/api/assets \
  | python -c 'import json,sys; print(json.load(sys.stdin)["id"])')

curl -X POST http://127.0.0.1:48125/api/jobs \
  -H 'content-type: application/json' \
  -d "{\"mode\":\"i2v\",\"prompt\":\"The camera slowly moves forward.\",\"conditions\":[{\"asset_id\":\"$ASSET_ID\",\"kind\":\"image\",\"index\":0,\"strength\":1.0}]}"
```

FLF2Vでは同じ要領で2枚を登録し、`conditions`に先頭を`index: 0`、末尾を`index: -1`として指定します。一般条件モードでは、動画の`kind`を`video`にします。

```bash
curl -X POST http://127.0.0.1:48125/api/jobs \
  -H 'content-type: application/json' \
  -d '{"prompt":"雨の東京を飛ぶ白い鶴。映画的なカメラ。遠くで雷鳴。","seed":42}'
```

返された`id`を使って `GET /api/jobs/{id}` を呼び、`completed`になったら`video_url`からMP4を取得します。GPUを同時実行で枯渇させないよう、生成は1本ずつ処理します。

## 設定

- `OFFLOAD_MODE=model`: 推奨。モデル単位CPUオフロード
- `OFFLOAD_MODE=sequential`: 最小VRAM、最も低速
- `OFFLOAD_MODE=none`: 全モデルをGPUへ配置。大容量VRAM向け
- `MODEL_REVISION`: 再現性のため初期値は確認済みコミットに固定
- `MAX_QUEUE_SIZE`: Modal spawn 前允许的 queued+running 总容量（既定4）；超限返回 HTTP 429
- `HISTORY_DB`: セッションと生成履歴のSQLiteファイル（既定`outputs/history.sqlite3`）
- `LLM_BASE_URL`: プロンプト変換に使うOpenAI互換APIの`/v1`までのURL
- `LLM_MODEL`: 外部LLMのモデル名
- `LLM_API_KEY`: 外部LLMのAPIキー（ローカルAPIでは空でも可）
- `INPUT_DIR`: アップロードした条件アセットの保存先（既定`inputs`）
- `LORA_DIR`: LoRA `.safetensors` の配置先（既定`loras`）
- `MAX_UPLOAD_SIZE_MB`: 1ファイルの上限（既定500MB）
- `LTX25_DECODER`: 2倍高解像度化後のデコード方式。`diffusion`（既定・diffusion decoderによる高品質デコード。NATTEN導入後の追加コストは約18秒）または`vae`（従来の畳み込みVAE・最速）。リクエストの`decoder`フィールドでジョブ単位に上書きできます。2倍高解像度化がOFFのジョブは常にVAEデコードです
- `LTX25_VIDEO_CRF`: 出力MP4のlibx264 CRF（既定18）。全経路（draft/high、全デコーダ）に適用されます。従来の既定CRF~23より高ビットレートで、圧縮によるディテール損失を抑えます
- `LTX25_TRANSFORMER_PRECISION`: transformerの精度。`nf4`（既定・bnb 4bit）、`fp8`（bf16重みをlayerwise castingでfp8_e4m3fnストレージ化・演算はbf16）、`bf16`（リリース重み約38GB）。`fp8`は品質がbf16同等のまま実測ピークVRAMが静止画26.5GB / 動画121フレーム28.9GBに収まる48GB級GPU向けの推奨構成です（castはCPU上で適用するためGPU側の一時38GBピークは発生しません。要: bf16 transformerシャード約38GBのHFキャッシュ）。`bf16`は96GB級GPU向け、24GB級では`nf4`のまま使ってください。text_encoderはいずれの値でもNF4です。**`nvfp4`（2026-09追加・sm_120 Blackwell専用）**: Lightricks公式配布のBlackwellネイティブFP4蒸留transformer（常駐約19GB）を`torch._scaled_mm`のFP4 GEMMで直接実行します（`ltx25/acceleration/nvfp4.py`。GEMM素でbf16比3.2〜3.8倍、リアルタイム用途の最速構成。`LTX25_NVFP4_CKPT`でローカルファイルを指定可、未指定ならHF Hubから自動取得）
- `LTX25_CUDA_GRAPH`: `1`でtransformer forward全体をCUDA Graph capture/replayし、denoiseのCPUカーネル起動コストを消します（`ltx25/acceleration/cuda_graph.py`）。出力はeagerと**bit完全一致**。`OFFLOAD_MODE=none`前提（それ以外では警告して無効）。LoRAを使うジョブは自動でeagerに落ちます。効果は小解像度×少ステップほど大きい（下記「リアルタイム生成と高速化」参照）
- `LTX25_CUDA_GRAPH_MAX_CAPTURES`: graphを保持するshape数の上限（汎用既定8、dual-resident Modal Director は既定1）。解像度・フレーム数・fps・モード（t2av/a2v）の組ごとに1本captureされ、**上限超過のshapeは警告ログの上、黙ってeagerにフォールバック**します。多shape運用ではQwen headroomを確認してから引き上げてください
- `LTX25_COMPILE_BLOCKS`: 【実験的・非推奨】per-block torch.compile（`ltx25/acceleration/compile.py`のdocstring参照）。probeではgraph単体に勝つがサーバE2Eでは利得なし・小解像度では退行、と実測済みのため既定`off`
- `LTX25_VIDEO_ENCODER`: `nvenc`（既定・h264_nvenc）または`x264`。NVENC不在環境はx264へ自動フォールバック
- `LTX25_NVENC_PRESET`: NVENCプリセット（`p1`最速〜`p7`最高品質、既定`p7`）。リアルタイム用途は`p4`でエンコード0.1〜0.15s短縮
- `LTX25_DECODE_SINGLE_TILE`: diffusion decoderのタイル方針（`auto`既定/`on`/`off`）。空きVRAMが許せば単一タイル（約1.23倍速・継ぎ目なし）
- `LTX25_STAGE_DEBUG`: `1`で段階別時間（text_encode/denoise/decode/encode）をログ出力（既定0）

## 品質と速度の実測（RTX PRO 6000 Blackwell 96GB、512×512→出力1024²、121フレーム、seed=42）

| 構成 | 合計時間 | ピークVRAM | 備考 |
|---|---|---|---|
| nf4 + VAEデコード + CRF18（既定、FastAPI経由・modelオフロード） | 76s | — | 従来互換。音声mux正常 |
| nf4 + diffusion decoder（NATTEN na3d、torch 2.11+cu130、FastAPI経由・modelオフロード） | 約90s | 17.3GB（decode時） | **うちデコード18.0s**。品質はflex経路と同水準（下記） |
| nf4 + diffusion decoder（旧: flex-attention、torch 2.9、全GPU常駐、初回） | 329s | 61.9GB | うちデコード302s（flexカーネルのコンパイル込み） |
| nf4 + diffusion decoder（旧: flex、同一プロセス2回目=ウォーム） | 314s | 61.9GB | うちデコード293s。コンパイル分の短縮は約10sのみで、デコード計算自体が支配的 |
| bf16 transformer + diffusion decoder（旧: flex、全GPU常駐） | 321s | 87.3GB | 96GB内に収まる。24GB級では不可 |

**diffusion decoderは既定のVAEデコード比で細部品質を大きく改善します**。平滑領域の微細テクスチャ保持（min 128pxパッチ分散）が1.67→2.41へ向上し、VAEデコード特有の偽グレイン様の高周波ノイズが消えます（グローバルLaplacian分散36.3→25.2の低下はノイズ減少によるもの）。

**NATTEN カーネル（2026-08-19 導入）**: torch 2.11.0+cu130 へ更新し、`kernels` パッケージ経由で `shi-labs/natten` のプリビルト na3d カーネル（torch211-cxx11-cu130、sm_120 動作確認済み）を使う `LTX2VideoVaeNeighborhoodNattenProcessor` を diffusion decoder に適用した（`ltx25/models.py`。取得不可の環境では従来の compiled flex-attention へ自動フォールバックし、どちらが使われたかを起動ログに出力する）。decode 専用実測（scratch_ab/latents.pt、1024²×121f）: **293s（flex・ウォーム）→ 18.3s（約16倍）**、ピークVRAM 35.8GB → 17.3GB。品質指標も flex 経路と一致（Laplacian分散 25.13 vs 25.17、平滑部min 128pxパッチ分散 2.399 vs 2.414、raw frame 平均絶対差 0.066/255）。デコードが約18秒まで短縮されたため、**既定デコーダは `diffusion` に変更済み**（`LTX25_DECODER=vae` で従来経路に戻せる）。

## リアルタイム生成と高速化（2026-09 実測）

### 目的と限界(まず読むこと)

この高速化スタックの**目的は「小解像度 × 少ステップ × 同一設定の連投」というサービング/リアルタイム用途のレイテンシ最小化**です(生成時間 < 再生時間の達成)。以下の限界を理解した上で使ってください:

- **大解像度・多ステップの品質重視生成にはほぼ効きません**。CUDA Graph の利得は CPU カーネル起動が律速になる小 shape に集中し(384×288 で -20%、512×288 で -4%)、1024×576 以上・8step・2倍高解像度化パスでは GPU 実行が支配的で数%程度です
- **低 VRAM 運用とは両立しません**。CUDA Graph は `OFFLOAD_MODE=none`(全常駐)前提で、model/sequential オフロード(24〜48GB 級の省 VRAM 構成)とは併用不可。省 VRAM と最速化は別軸の選択です
- **検証済みの組み合わせは nvfp4 のみ**。`fp8`(layerwise casting)+ CUDA Graph は hook 機構との相互作用が未検証です
- **LoRA を使うジョブは自動で eager に落ちます**(graph の恩恵なし)
- **shape(解像度・フレーム数・fps・モード)を変えるたびに初回 capture 費(+1.5〜2s)**が乗るため、毎回設定を変える一発生成の使い方では償却できません
- torch.compile の併用は実測で利得なし(§compile 参照)
- モデル自体の品質限界(fps=16 の周期揺らぎ、長尺の stall)は高速化では解決しません — 下記の品質節を参照

nvfp4 + CUDA Graph + NVENC の組み合わせで、**生成時間 < 再生時間（リアルタイム比 1.0x 未満）のストリーミング生成**が成立します。実測はすべて RTX PRO 6000 Blackwell 96GB（sm_120）、蒸留σ・4step・約5秒クリップ・t2av、`LTX25_TRANSFORMER_PRECISION=nvfp4 OFFLOAD_MODE=none LTX25_CUDA_GRAPH=1 LTX25_NVENC_PRESET=p4`。

### CUDA Graph の効果（bit一致・実測）

| 条件（4step） | graphなし | graphあり |
|---|---|---|
| 512×288×121f | 2.48s | 2.38s（-4%） |
| 384×288×81f | 2.30〜2.37s | **1.83〜1.88s（-20%）** |

解像度が小さいほどCPUカーネル起動律速の比率が上がるため効果が大きくなります。映像framemd5・音声md5ともeagerと完全一致。新しいshapeの初回のみcapture費（+1.5〜2s）が乗ります。

### リアルタイム解像度上限（約5秒クリップ・4step・リアルタイム比1.0x基準）

| fps | 安全圏（≤0.90x） | ギリギリ（〜0.98x） |
|---|---|---|
| 20fps（97f） | 768×512 | 768×576 |
| 24fps（121f） | 704×480 | 768×480 |

8stepは境界が約半分（16fps基準で〜34万画素）。a2v（音声条件付け）はt2av比で平均+0.15sのみ（境界セルだけ注意）。

### 【重要】fps=16 は品質NG

fps=16指定は**4.0秒（64フレーム）周期のモーション揺らぎ**（一瞬停止/僅かな巻き戻り感）が構造的に発生します（LTX-2.5の学習が24/25fps中心で、RoPE時間座標が分布外になるため。同一seed・同一内容のA/Bで20fps・24fpsでは消失を確認）。**リアルタイム用途は20fps推奨**（品質クリーンかつ予算が24fpsより広い）。

### 長尺単発生成

704×416・8step・16fpsで30秒（APIの`num_frames≤481`上限）まで速度・VRAMとも成立（30秒クリップを29.4s・ピーク40.9GBで生成）。ただし**25秒級の長尺ではseed依存の「モーション停止→ジャンプ」が散発**します。`experiments/probes/detect_motion_stall.py`（約0.5s/本、検出時exit 1）を生成後QCに使い、検出時は別seedでリトライする運用を推奨します。

### experiments/probes/

開発時の検証スクリプト群を同梱しています: CUDA Graph の等価性・速度（`probe_cudagraph*.py`）、torch.compile 併用の検証記録（`probe_compile_*.py`、結論は「本番では利得なし」）、nvfp4 の量子化検証（`probe_nvfp4_*.py`）、モーション停止検出器（`detect_motion_stall.py`）。

## ライセンス

このリポジトリで独自に実装したアプリケーションコードは[Apache License 2.0](LICENSE)で提供します。

> [!IMPORTANT]
> Apache License 2.0はLTXモデルの重み、LTX由来のLoRA／チェックポイント、Gemmaモデル、その他の第三者製コンポーネントには適用されません。

LTX-2/LTX-2.5およびその派生物には、Lightricksの[LTX-2 Community License Agreement](https://github.com/Lightricks/LTX-2/blob/main/LICENSE)が適用されます。用途制限、配布時のライセンス同梱・告知義務などがあり、年間売上が1,000万米ドル以上の事業体による商用利用にはLightricksとの有償商用ライセンスが必要です。モデル、LoRA、生成結果を利用または配布する前に、必ず原文の最新版を確認してください。商用ライセンスについては[LTX Model Licensing](https://ltx.io/model/license)を参照してください。

Gemmaテキストエンコーダーを含む第三者のモデル・ライブラリ・カーネルは、それぞれの配布元が定めるライセンスと利用規約に従います。

`.env`と生成物はGit管理外です。公開サーバーとして運用する場合は、リバースプロキシ側で認証・TLS・レート制限を追加してください。
