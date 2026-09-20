# Architecture

这个仓库只维护一件事：**把 LTX-2.5 与可扩展的图像/视频生成模型作为高性能 Modal Director Runtime 提供给导演台。**

## 主链路

```text
Frontend
   |
   v
ltx25/api.py
   |
   v
ltx25/modal_client.py
   |
   | Modal RPC
   v
modal_app.py
   |
   v
ltx25/director.py            # resident engine routing
   |        \
   |         `--> Qwen-Image 2.1 BF16
   v
ltx25/runtime.py              # LTX workflow
   | \
   |  \--> ltx25/encoding.py
   v
ltx25/models.py
   |
   v
ltx25/acceleration/
   |-- nvfp4.py
   |-- cuda_graph.py
   `-- compile.py
   |
   v
Diffusers / LTX-2.5
```

前端内部同样只有一条生产链：

```text
frontend/src/main.js
        |
        v
      App.vue
        |
        +--> studio/components/*   # 纯视图
        |
        v
useStudioRuntime.js                # state / polling / upload / actions
        |
        +--> studio/model/*        # 纯业务规则与 selectors
        |
        `--> runtime/api.js        # 唯一 HTTP boundary
```

Vue 组件不直接 `fetch`、不自己维护轮询；生产入口和测试共享同一个
`useStudioRuntime()`，避免 UI 与测试各自维护一套运行时逻辑。

## 目录职责

```text
ltx25/
├─ api.py            # HTTP、上传协议、下载重定向、HTTP 错误映射
├─ modal_client.py   # 本地 -> Modal：Job / Dict / warmup / cancel
├─ media_store.py    # 单物理媒体 backend：Volume / generic S3-compatible storage
├─ media_storage.py  # 稳定 store_id、primary/fallback 路由、MediaRef
├─ director.py       # 常驻模型集合、engine 路由；不依赖 Modal/FastAPI
├─ runtime.py        # LTX generation workflow 与采样编排
├─ models.py         # 模型 load/unload、decoder、precision、加速安装
├─ encoding.py       # NVENC / ffmpeg 输出编码
├─ schemas.py        # 请求、任务、资产等数据契约
├─ config.py         # runtime 配置
└─ acceleration/
   ├─ nvfp4.py       # Blackwell FP4 GEMM
   ├─ cuda_graph.py  # transformer.forward capture/replay
   └─ compile.py     # 实验性 torch.compile

modal_app.py         # 唯一 Modal 部署入口
frontend/            # 导演台
experiments/         # benchmark / probe，不进入生产依赖
scripts/             # 模型准备与工具
tests/               # 行为 + 架构边界
```

Director Runtime 默认保持旧客户端兼容：`engine=auto` 与 `engine=ltx` 都走 LTX；
`engine=qwen` 目前只允许纯 `t2i`。Modal worker 启动时按配置一次性加载 LTX-2.5
NVFP4 与 Qwen-Image 2.1 BF16，之后 job 只做 engine dispatch，不做模型卸载/重载。
Qwen 使用与现有 t2i 一致的最终尺寸语义：请求中的 base width/height 在 `upscale=true`
时直接渲染为 2× 最终 PNG。

持久化按生命周期拆开：

```text
ltx25-models   # LTX 模型权重，只读挂载到 GPU worker
qwen-image21-cache # Qwen-Image 2.1 HF cache，只读挂载到 GPU worker
ltx25-state    # Volume 模式的 inputs/outputs + 始终保留的 LoRA/state
ltx25-kernels  # NATTEN 等已加载 shared-library kernel cache，不参与 state reload
ltx25-jobs     # Modal Dict：任务状态
S3-compatible  # 可选媒体 data plane；inputs/outputs 与浏览器直传
```

`ltx25-jobs` 同时维护轻量索引：

```text
session:<session>:jobs  # 当前会话 job id 列表，轮询不再扫描整个 Dict
cancel:<job>            # durable cancel tombstone，防止 worker 状态回写复活任务
asset:<asset>           # 上传时间/object key，用于输入资产 TTL 清理
upload:<asset>          # prepare -> PUT -> finalize 期间的临时上传意图
```

浏览器进入 Studio 时获取短期 GPU warm lease（默认 90 秒），前端每 30 秒续租；
active job 也会自动续住 worker。租约过期且没有 active job 时，本地 Router 自动将
Modal worker 的 `scaledown_window` 收到 2 秒，因此浏览器崩溃/网络断开也不会让
Router 永久续命 GPU。页面正常离开时仍发送 keepalive unload 立即释放租约。

媒体传输保持控制面/数据面分离的边界：

```text
/api/jobs/status          # 轮询只返回 UI 所需的轻量 Job summary
/api/jobs/{id}            # 需要复用参数时才读取完整 GenerateRequest
/api/assets/prepare       # 返回 Volume proxy 或 S3 presigned 上传计划
/api/assets/{id}/content  # Volume fallback 的 raw binary PUT
/api/assets/{id}/complete # finalize + size verify + durable asset registration
/api/assets/from-job/{id} # Retake/Extend: data-plane server-side copy
/outputs/{name}           # Volume FileResponse / S3 signed redirect
```

默认 `VolumeMediaStore` 保持零配置兼容；`S3MediaStore` 始终只描述一个物理
S3-compatible backend。多存储选择由 `MediaStorage` 负责：对象元数据持久化稳定
`store_id + key`，当前生产配置为 R2 primary、SG-JP MinIO fallback。`store_id` 是
稳定标识而不是 `primary/fallback` 角色，因此以后角色互换不会让历史对象指向错误位置。

浏览器直传先在当前 store 内做有限 retry；只有字节传输仍失败时才请求新的 fallback
upload plan。一次 upload intent 从 prepare 到 complete 固定在一个 store，multipart ETag
绝不跨 store 复用。输出同样记录 `output:<filename> -> store_id + key`，外部 API 仍只
暴露稳定 `/outputs/<filename>`。

Retake / Extend 不再把已生成视频下载到浏览器后重新上传：Volume 使用
`Volume.copy_files()`，S3 使用 `CopyObject`。`/api/health` 同时暴露媒体
backend 与 upload/download/copy 指标；直传 WAN 时间由浏览器 finalize 时回报。

MP4 主编码路径写入 `+faststart`，将 `moov` 元数据前置。输出先在 GPU container
本地 scratch 完成 mux，再通过 `MediaStorage` 上传；primary 不可用时可直接写 fallback。
输入在每个 job 下建立独立本地 namespace：小对象（默认 <=32 MiB）和 fallback
对象并行 prefetch 到本地 scratch；较大的 primary R2 对象继续通过只读
CloudBucketMount fast path。这样避免小图/音频承担 mount 首读延迟，也避免大视频
无意义地完整复制。`runtime.py` 始终只看到普通本地 `input_dir`。

任务 admission 使用 `active:jobs` 紧凑索引；`MAX_QUEUE_SIZE` 在 Modal
`.spawn()` 前硬限制 queued+running 总数，超限返回 HTTP 429。每个 job 记录
`queue/state_reload/input_staging/generation/output_store/worker` 分阶段耗时。

## 依赖规则

```text
api ----------> modal_client ----------> Modal SDK
                    |
                    `----------> media_storage ----> media_store ----> Volume / S3 API

modal_app ----> runtime ----> models ----> acceleration
                   |
                   `--------> encoding

api / modal_client / modal_app / runtime ----> schemas
models / runtime ----------------------------> config
```

禁止：

```text
runtime       -X-> FastAPI
runtime       -X-> Modal
runtime       -X-> media_storage / media_store
media_store   -X-> media_storage
models        -X-> FastAPI / Modal / runtime
acceleration  -X-> serving 层
modal_client  -X-> runtime / models
api           -X-> Modal SDK / runtime / models
```

## 为什么只拆到这里

按“独立变化原因”拆，而不是按功能名拆。

- Modal transport 会独立变化，所以有 `modal_client.py`。
- 模型生命周期和 generation workflow 的变化原因不同，所以拆成 `models.py` / `runtime.py`。
- NVFP4、CUDA Graph、torch.compile 是独立性能实验，所以单独放 `acceleration/`。
- MP4/NVENC 与模型无关，所以保留 `encoding.py`。
- `t2v / i2v / a2v / retake / extend` 目前大量共享同一 Diffusers pipeline，不为每个 mode 建文件。

只有当某个 workflow 出现独立模型、独立生命周期或明显独立测试压力时，再从 `runtime.py` 抽出。

## 唯一运行方式

本地：

```text
python -m uvicorn ltx25.api:app --host 127.0.0.1 --port 48125
```

云端：

```text
modal run modal_app.py::prepare_models
modal deploy modal_app.py
```

旧的本地 GPU standalone server 和远程 Modal CPU gateway 已移除。Git 历史保留其实现，不再在主分支维护第二、第三套 serving 主链。
