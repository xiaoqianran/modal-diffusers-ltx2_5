# 架构说明

这个仓库只有一条主链路：

~~~text
Browser / Vite
      |
      v
backend/api/local.py
      |  HTTP only
      v
backend/control/modal.py
      |  Modal SDK / Job / Volume / warmup
      v
deploy/modal.py
      |  Modal Worker
      v
backend/runtime/engine.py
      |  LTX-2.5 inference
      v
backend/runtime/acceleration/
      +-- nvfp4.py
      +-- cuda_graph.py
      +-- compile_blocks.py
~~~

## 目录职责

~~~text
backend/
├─ api/
│  └─ local.py              # 主 API。只处理 HTTP、上传验证、响应映射
│
├─ control/
│  └─ modal.py              # Modal 控制面：任务、Volume、Worker、warmup、cancel
│
├─ runtime/
│  ├─ engine.py             # GPU 推理主流程
│  ├─ encoding.py           # MP4 编码
│  └─ acceleration/
│     ├─ nvfp4.py           # NVFP4
│     ├─ cuda_graph.py      # CUDA Graph
│     └─ compile_blocks.py  # 实验性 torch.compile
│
├─ contracts.py             # API / Job 共用数据结构
├─ config.py                # Runtime 配置
│
└─ compat/                  # 非主链路，仅兼容旧运行方式
   ├─ standalone.py         # 单机 FastAPI + 本地 GPU
   ├─ jobs.py               # standalone 的进程内 JobManager
   └─ modal_gateway.py      # 旧 Modal CPU Gateway

deploy/
└─ modal.py                 # Modal 部署装配、GPU Worker

frontend/                   # Vite UI
scripts/                    # 模型准备 / 运维脚本
experiments/probes/         # 性能与正确性实验，不属于生产主链路
tests/                      # 行为测试 + 架构边界测试
docs/                       # 设计与实验文档
~~~

## 依赖方向

只允许向下依赖：

~~~text
api  ───────> control ───────> Modal
 |               |
 └──> contracts  └──> contracts

deploy ─────> runtime ─────> acceleration
   |             |
   └──> contracts└──> config / contracts

compat ─────> runtime / config / contracts
~~~

禁止：

~~~text
runtime  ─X─> FastAPI
runtime  ─X─> Modal
control  ─X─> FastAPI
control  ─X─> api
api      ─X─> compat
~~~

这些约束由 `tests/test_architecture.py` 自动检查。

## 主入口

本地导演台：

~~~powershell
start-ltx25.bat
~~~

等价后端入口：

~~~powershell
python -m uvicorn backend.api.local:app --host 127.0.0.1 --port 48125
~~~

Modal 部署：

~~~powershell
modal run deploy/modal.py::prepare_models
modal deploy deploy/modal.py
~~~

`modal_app.py` 仅保留为旧命令兼容入口，新代码不要再放进去。

## 阅读顺序

第一次看项目只需要按这个顺序：

~~~text
1. backend/contracts.py
2. backend/api/local.py
3. backend/control/modal.py
4. deploy/modal.py
5. backend/runtime/engine.py
6. backend/runtime/acceleration/*
~~~

`backend/compat/` 和 `experiments/` 默认不用读。

## 设计原则

1. **主链路唯一**：新增功能优先落在 `api -> control -> runtime`，不要再创建平行 API。
2. **部署不承载业务逻辑**：`deploy/modal.py` 只做资源、镜像、Worker 装配。
3. **Runtime 不知道 HTTP/Modal**：推理引擎只接收 `GenerateRequest` 和文件路径。
4. **实验与生产隔离**：探针、benchmark、负结果都放 `experiments/`。
5. **按压力拆分**：只有模块出现真实变化压力时再继续拆，避免为了“分层”产生几十个空壳文件。
