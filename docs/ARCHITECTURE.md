# Architecture

这个仓库只维护一件事：**把 Diffusers/LTX-2.5 作为高性能 Modal runtime 提供给导演台。**

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
ltx25/runtime.py
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
├─ api.py            # HTTP、上传下载、HTTP 错误映射
├─ modal_client.py   # 本地 -> Modal：Job / Dict / Volume / warmup / cancel
├─ runtime.py        # generation workflow 与采样编排
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

Modal 持久化也按生命周期拆开：

```text
ltx25-models   # 模型权重，只读挂载到 GPU worker
ltx25-state    # inputs / outputs / LoRA；每个 job 开始前可 reload
ltx25-kernels  # NATTEN 等已加载 shared-library kernel cache，不参与 state reload
ltx25-jobs     # Modal Dict：任务状态
```

`ltx25-jobs` 同时维护轻量索引：

```text
session:<session>:jobs  # 当前会话 job id 列表，轮询不再扫描整个 Dict
cancel:<job>            # durable cancel tombstone，防止 worker 状态回写复活任务
asset:<asset>           # 上传时间/Volume path，用于输入资产 TTL 清理
```

浏览器进入 Studio 时显式 warm GPU；真正离开页面时发送 keepalive unload，
本地 Router 将 Modal worker 的 `scaledown_window` 收到 2 秒。Router 自身退出时
lifespan 也执行同样的 2 秒回收兜底。

## 依赖规则

```text
api ----------> modal_client ----------> Modal SDK

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
