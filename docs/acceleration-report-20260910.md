# LTX-2.5(22B)を diffusers 直組みでリアルタイム化する — 高速化の全記録

2026-09-10 / 実測環境: RTX PRO 6000 Blackwell Workstation 96GB(sm_120)、torch 2.11.0+cu130、diffusers(git)、単GPU

## TL;DR

- 動画+音声 22B モデル(LTX-2.5)の diffusers 直組みサーバで、**5秒クリップを 1.83 秒で生成(リアルタイム比 0.37x)** まで到達した
- 積んだ手法は4層: **蒸留ステップ間引き(4step)× NVFP4(FP4 GEMM)× CUDA Graph(起動コスト除去)× NVENC/非同期エンコード**
- CUDA Graph は **出力 bit 完全一致**のまま小解像度 4step で **-20%**。ComfyUI で同種の報告が無いのはアーキテクチャ上の必然(後述)
- torch.compile 併用は custom_op トリックで fullgraph まで通したが、**本番 E2E では利得なしという負の結果**(過程と数値を全公開)
- 副産物として **fps=16 指定は 4 秒周期のモーション揺らぎが構造的に出る**(学習分布外の RoPE 時間座標)という品質知見を発見 — リアルタイム用途は 20fps 推奨
- コードは全て公開: [animede/diffusers-ltx2_5](https://github.com/animede/diffusers-ltx2_5)(検証スクリプトは `experiments/probes/`)

---

## 1. 前提と高速化スタック全景

LTX-2.5 は 22B の音声同時生成 DiT。本サーバは ComfyUI ではなく diffusers パイプラインを FastAPI で直接組んでいる。高速化は独立な4層で、それぞれ別のボトルネックを潰す:

| 層 | 手法 | 潰すもの | 効果(実測) |
|---|---|---|---|
| ステップ数 | 蒸留8σ + 等間隔間引きで 4step | denoise 回数 | 8→4step で -31% |
| 演算 | NVFP4(公式 FP4 蒸留 ckpt + `torch._scaled_mm`) | GEMM 時間 | GEMM 素 3.2〜3.8x(層全体 ~1.8x) |
| 起動 | CUDA Graph capture/replay | CPU カーネル起動 | 小解像度 4step で -20%(bit 一致) |
| 後処理 | NVENC p4 + mp4 非同期エンコード + uint8 GPU 化 | エンコード・転送 | 計 -0.5s 級/クリップ |

このほか、2倍高解像度化パス(非リアルタイム)の diffusion decoder には NATTEN プリビルトカーネルを適用済み(decode 293s→18.3s、既報)。

## 2. NVFP4 — Blackwell ネイティブ FP4 の直接実行

Lightricks 公式配布の nvfp4 量子化蒸留 transformer(ComfyUI 単一ファイル形式、18.7GB)を、diffusers 側で直接ロードして `torch._scaled_mm`(cuBLAS block-scaled FP4、sm_120 ネイティブ)で回す(`ltx25/acceleration/nvfp4.py`)。活性化の動的 FP4 量子化は自前 Triton カーネル(torch に bf16→fp4 cast が存在しないため)。

**実装の罠(bf16 リファレンスとの全数値照合で特定)**:
1. checkpoint の nibble 順は cuBLAS パック規約と逆(ロード時に byte 内スワップが必要)
2. `weight_scale` は**既に cuBLAS blocked layout で格納**されており、正規の swizzle を重ねると二重適用になる。しかもブロック単位のスケール置換なので分布は正常に見え、**層単体の cosine 検査では検出できない**(自作 dequant と自作 GEMM が「同じ誤解釈」で一致してしまう)。検出には公式 bf16 重みとの直接照合が必要だった
3. `_scaled_mm` にブロックスケールを素の行順で渡すと**エラーにならず黙って壊れた値**が出る

## 3. CUDA Graph — 「起動律速」の正体と bit 一致の -20%

### 方式

Lightricks 公式 LTX-2 パッケージの `cudagraph_capture.py`(v1.2.0+)の方式 — side stream で warmup → cuBLAS workspace クリア → 共有 mempool で capture → 毎回 replay — を、diffusers の `LTX2VideoTransformer3DModel.forward` **全体**に適用した(`ltx25/acceleration/cuda_graph.py`)。パイプラインが RoPE 座標を事前計算して渡し、蒸留経路(cfg=1 / stg=0)は forward 内に CPU 依存分岐が無いため、公式(ブロックループのみ)より広い境界が成立する。

- capture キー = 全テンソル引数の shape/dtype + 非テンソル引数。shape ごとに1本、初回のみ capture 費 +1.5〜2s
- **出力は eager と bit 完全一致**(映像 framemd5・音声 md5 で E2E 検証)
- LoRA ジョブは自動で eager に落とす(graph は重みアドレスを焼き込むため)

### 効果と「効く条件」

| 条件(4step) | eager | graph | 差 |
|---|---|---|---|
| 512×288×121f | 2.48s | 2.38s | -4% |
| 384×288×81f | 2.30〜2.37s | **1.83〜1.88s** | **-20%** |

**小解像度ほど効く**。CUDA イベントでの切り分け: eager は host-launch 1.95s ≈ GPU 2.0s(CPU がカーネルを発行しきれず GPU に隙間)、graph 化で CPU 側が 0.001s になり GPU 実行の下限が露出する。大きい shape では GPU 実行自体が支配的になるため差が縮む。

### 計測の罠(重要)

非同期実行下では、**denoise ループのコールバック間隔はホスト側の発行時間であって GPU 時間ではない**。当初この計測で「denoise 1.82s→0.48s(3.8x)」と誤読した(実際の GPU 側は 250→225ms/fwd = 1.10x で、E2E 利得は上表のとおり)。GPU 時間は必ず cuda Event で測ること。

### なぜコミュニティ(ComfyUI)に報告が無いか

不可能ではないが、条件が構造的に噛み合わない:
1. **動的メモリ管理と相性最悪** — graph は重みの GPU アドレスを焼き込む。ComfyUI の中核(lowvram 部分ロード・モデル退避)は「重みが動く」仕組みそのもの
2. **ModelPatcher / ノードの動的性** — LoRA・ControlNet・attention パッチは forward を実行時に差し替える。ワークフローを1ノード変えるだけで capture が無効化(最悪、黙って stale 出力)
3. **使われ方が毎回違う shape** — capture 費は同一 shape の連投でしか償却できない。「固定 shape のクリップを何百本も流す常駐サーバ」という運用形態自体が ComfyUI 的には特殊

本家 LTX-2 パッケージには実装があり(我々の参照元)、ComfyUI 統合はそれを使っておらず、diffusers 直組み人口は少ない — という三すくみで報告が空白になっている。

## 4. torch.compile 併用 — 負の結果の全記録

「graph は CPU 起動を消すが GPU 側の小カーネル群は残る → Inductor の融合で潰せるはず」という仮説を最後まで検証した(`experiments/probes/probe_compile_*.py`)。

1. **素の per-block compile は fullgraph 不可** — 自前 FP4 カーネルの `float4_e2m1fn_x2` dtype で Inductor の Triton codegen がクラッシュ
2. **custom_op 化で解決** — nvfp4 linear を `torch.library.custom_op` で opaque 化すると、graph break 解消 + fp4 隠蔽が同時に達成され、fullgraph=True が通る(custom_op 単体は bit 一致)
3. **単体プローブでは勝つ** — compiled+graph 250→168ms/fwd(1.48x)
4. **しかし本番 E2E では利得なし** — 同 shape で誤差範囲、リアルタイム小 shape では**退行**(1.83s→2.39s)。極小テンソルでは Inductor 生成カーネルの固定費が eager の ATen カーネルに負ける
5. 途中の罠: `dynamic=False` を付けないと2つ目の shape で dynamic 汎用カーネルが capture されさらに悪化(2.77s)
6. 数値面: compile 出力は eager と bit 不一致(ブロックあたり cosine ~0.9999 の系統誤差が 48 ブロックで累積し 1 forward で ~0.965)。aot_eager は bit 一致・`emulate_precision_casts` でも不変 → 単一のバグではなく融合カーネル群の微小差の累積で、実映像では「構図の微ドリフト」として現れる(品質自体は目視同等)

**結論: この規模の小 shape 領域では compile は効かない。** コードは実験フラグ(`LTX25_COMPILE_BLOCKS`、既定 off)として残してある。

## 5. リアルタイム運用マップ(実測)

約5秒クリップ・4step・上記フル構成での、リアルタイム比 1.0x 基準の解像度上限:

| fps | 安全圏(≤0.90x) | ギリギリ(〜0.98x) |
|---|---|---|
| 20fps(97f) | 768×512 | 768×576 |
| 24fps(121f) | 704×480 | 768×480 |

- 8step(品質側)は境界が約半分
- a2v(音声条件付け+アンカー画像 = リップシンク経路)は t2av 比 **平均 +0.15s のみ**
- 長尺単発は 704×416・8step で **30 秒(API 上限)まで速度・VRAM とも成立**(29.4s / 40.9GB)

### 【発見】fps=16 は品質 NG

予算を広げるために fps=16 を試したところ、**4.0 秒(64 フレーム)ちょうどの周期でモーションが「淀む→ジャンプ」する構造的アーティファクト**を発見した(64f 折り畳み位相プロファイルで定量化)。同一 seed・同一内容の A/B で 20fps・24fps では消える。機構は「LTX-2.5 の学習が 24/25fps 中心で、RoPE の時間座標(秒単位)が 16fps では分布外になる」で説明がつく。**リアルタイム用途は 20fps 推奨**。

### 長尺の品質 QC

25 秒級以上の単発生成では、seed 依存の「モーション停止→ジャンプ」が散発する(同一 seed なら同位置に決定論的に再現、seed を変えると消える/移動する)。`experiments/probes/detect_motion_stall.py`(フレーム間差分の停滞帯→ジャンプ型検出、~0.5s/本、検出時 exit 1)を生成後 QC に入れ、検出時に別 seed でリトライする運用で実用になる。

## 6. 運用上の罠まとめ

- capture 上限(`LTX25_CUDA_GRAPH_MAX_CAPTURES`、既定8)を超えた shape は**警告ログだけで黙って eager に落ちる**。多 shape 運用・ベンチマーク時は必ず引き上げ、ログを確認すること(本記事の測定でも一度これで汚染データを掴んだ)
- a2v は音声トークン数が t2av と異なるため **capture キーが別**。モードごとに初回 capture 費が乗る
- graph は `OFFLOAD_MODE=none`(全常駐)前提。model/sequential オフロードとは併用不可

## 7. 実装レシピ(レポート単独で再現するための詳細)

### 7.1 CUDA Graph capture/replay(全文 ~170 行、`ltx25/acceleration/cuda_graph.py`)

**前提条件(1つでも欠けると成立しない)**:
- transformer forward 内に CPU 依存の分岐・`.item()`・CPU テンソル生成が無いこと。LTX2 パイプラインは RoPE 座標(`video_coords`/`audio_coords`)を forward の外で事前計算して渡すためこれが成立する(渡さないと forward 内の座標計算が CPU テンソルを作り、capture 中の H2D copy で落ちる)
- 蒸留経路であること(cfg=1・stg=0。STG 経路には `torch.zeros((B,))` の CPU テンソル生成がある)
- 全モデル GPU 常駐(オフロード無し)、生成は単一スレッド、LoRA 無し

**capture 手順(順序が重要)**:
```python
# 1. 全テンソル引数を clone して静的入力バッファを作る
statics = {k: v.clone() for k, v in kwargs.items() if isinstance(v, torch.Tensor)}
static_kwargs = {**kwargs, **statics}

# 2. side stream で warmup 3回(Triton JIT・cuBLAS ワークスペース確保・
#    カーネル選択を capture の外へ出す)
side = torch.cuda.Stream()
side.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(side):
    for _ in range(3):
        orig_forward(**static_kwargs)
torch.cuda.current_stream().wait_stream(side)
torch.cuda.synchronize()

# 3. cuBLAS ワークスペースをクリア(graph mempool への二重計上を防ぐ)
torch._C._cuda_clearCublasWorkspaces()

# 4. 入力を入れ直してから capture(warmup が書き換えた可能性への保険)
for k, sbuf in statics.items():
    sbuf.copy_(kwargs[k])

# 5. 共有 mempool で capture(複数 shape の中間メモリを共有するため)
pool = torch.cuda.graph_pool_handle()      # プロセスで1つ使い回す
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph, pool=pool, capture_error_mode="global"):
    outputs = orig_forward(**static_kwargs)  # outputs の参照を保存
```

**replay 手順と出力の契約**:
```python
for k, sbuf in statics.items():
    sbuf.copy_(new_kwargs[k])   # 静的バッファへ copy_
graph.replay()
return outputs                   # capture 時の静的出力バッファをそのまま返す
```
出力は静的バッファなので、**呼び出し側が次の replay より前に消費する**ことが契約。LTX2 パイプラインは transformer 出力を直後に `.float()` でコピーするため安全(自前パイプラインに組む場合はここを確認すること)。

**差し替えとキー設計**:
- 導入は `transformer.forward = runner`(インスタンス属性への代入。`nn.Module.__call__` はインスタンス属性の forward を優先するため、これだけで効く)
- capture キー = 全テンソル引数の `(名前, shape, dtype, device)` + 非テンソル引数の値。キーごとに graph 1本
- フォールバック: 未知型の引数 → eager / キー数が上限超過 → eager / capture 中の例外 → そのキーを恒久 eager 化(いずれもサービスを止めない)
- 無効化が必要な条件: モデルオフロード(重みアドレスが動く)、LoRA の付け外し(同上。LoRA ジョブは eager に落とし、ジョブ後に全 capture を破棄)

**検証**: 出力等価性は動画の framemd5 / 音声 md5(bit 一致するはず)。速度は cuda Event(ホスト側の時間はレイテンシ隠蔽で無意味 — §3 の罠)。

### 7.2 NVFP4 checkpoint フォーマットと forward(`ltx25/acceleration/nvfp4.py`)

**checkpoint(ComfyUI 単一ファイル形式)のテンソル仕様**(量子化層ごと):

| テンソル | dtype / shape | 意味 |
|---|---|---|
| `<module>.weight` | U8 `[out, in/2]` | e2m1 を 2 値/byte パック。**high nibble = 先頭要素(cuBLAS 規約と逆、ロード時に byte 内スワップ必須)** |
| `<module>.weight_scale` | F8_E4M3 `[out, in/16]` | 16 要素ブロックスケール。**既に cuBLAS blocked layout(swizzle 済み)で格納** — 正規の swizzle を重ねると二重適用で壊れる |
| `<module>.weight_scale_2` | F32 スカラー | グローバルスケール |
| `<module>.input_scale` | F32 スカラー | 活性化の静的スケール |

非量子化層(norm / bias / adaln 等)は bf16。ヘッダの `_quantization_metadata` に量子化層一覧がある。

**forward の演算列**:
```
x(bf16, [.., in]) → 2D化 → M を 16 の倍数へ pad
  → Triton カーネルで動的量子化: (packed fp4 [M, in/2], scales F8 [M, in/16])
  → 活性化側 scale を to_blocked() で swizzle(重み側はロード時に済ませてある)
  → torch._scaled_mm(xq, W.T, scale_a, scale_b, out_dtype=bf16)
  → × (input_scale × weight_scale_2) → + bias → 元 shape へ
```
`to_blocked` は cuBLAS の 128 行×4 列タイル並べ替え。**素の行順で渡してもエラーにならず黙って壊れる**ため、検証は必ず公式 bf16 重みとの出力照合で行う(層単体の自作 dequant との cosine 比較は「同じ誤解釈」で一致してしまい検出できない)。

### 7.3 蒸留ステップ間引き

公式蒸留 σ 列(8 段):
```
[1.0, 0.99609375, 0.9765625, 0.9375, 0.8515625, 0.578125, 0.28125, 0.109375]
```
steps=n(<8)指定時は**先頭と末尾を必ず含む等間隔**で n 個選ぶ(`round(linspace(0, 7, n))` のインデックス)。4step 実測 -31%、静止フレーム品質は破綻なし。

### 7.4 エンコード側

- `h264_nvenc`(preset は用途で p4〜p7)。NVENC 不在なら libx264 へ自動フォールバック
- mp4 エンコードはワーカースレッドで**次ジョブの denoise と重畳**(効くのはクライアントが完了前に次ジョブを投入する場合のみ)
- フレームの uint8 化は `output_type="pt"` で GPU 上 ×255。**bf16 のまま ×255 すると bit 不一致になる — `.float()` を挟むこと**(実測で確認した罠)

### 7.5 ファイル対応表

| 手法 | 実装 | 検証スクリプト |
|---|---|---|
| CUDA Graph | `ltx25/acceleration/cuda_graph.py` + `ltx25/runtime.py`(install/ガード) | `experiments/probes/probe_cudagraph.py` / `probe_cudagraph_gputime.py` |
| NVFP4 | `ltx25/acceleration/nvfp4.py` | `experiments/probes/probe_nvfp4_*.py` |
| compile(負の結果) | `ltx25/acceleration/compile.py` | `experiments/probes/probe_compile_*.py` |
| steps 間引き | `ltx25/runtime.py` の `_subsample_distilled_sigmas()` | — |
| stall 検出器 | — | `experiments/probes/detect_motion_stall.py` |

## 8. まとめ

- 蒸留 × FP4 × CUDA Graph × NVENC の4層で、22B 音声同時生成モデルの**リアルタイム越え(0.37x)**を単GPUで実証した
- 効果の主戦場は「小解像度×少ステップ×固定 shape 連投」= serving 用途。ここは diffusers 直組みが ComfyUI に対して構造的優位を持つ領域で、だからこそ community に前例が無い
- 負の結果(torch.compile)と品質知見(fps=16 の周期揺らぎ)も含め、全て再現可能な形で公開している

再現・検証は [animede/diffusers-ltx2_5](https://github.com/animede/diffusers-ltx2_5) の README と `experiments/probes/` を参照。
