# 16.2 可观测性：请求、引擎、GPU 三条时间线

Prometheus `/metrics` 是服务观测入口，但名称和 histogram 定义会变化，按版本保存原始指标。[vLLM Production Metrics](https://docs.vllm.ai/en/stable/usage/metrics/)

| 手段 | 记录什么 | 回答的问题 | 典型入口 |
|---|---|---|---|
| Metrics | 按时间窗口聚合的计数、Gauge、Histogram | 何时、哪些请求群体或设备退化？ | Prometheus 抓取、Grafana 展示；DCGM Exporter 提供 NVIDIA 设备指标，`nvidia-smi` 用于现场快照 |
| Log | 带时间戳的离散事件及上下文 | 哪个请求或 rank 遇到了什么错误？ | 服务结构化日志、Runtime / NCCL 错误日志 |
| Trace | 一个请求跨服务、进程与阶段的 Span 树 | 排队、Tokenize、Prefill、Decode、传输分别耗时多久？ | OpenTelemetry 等链路埋点与 Trace 后端 |
| Profiling | CPU/GPU 操作、Kernel 时间线与硬件计数器 | 已确认的热点算子为什么慢？ | PyTorch Profiler、Nsight Systems、Nsight Compute；CUPTI 是底层采集接口 |

**关联示例**：一次请求的 HTTP 入口生成 `request_id`，跨网关与引擎传播 `trace_id`；`queue`、`tokenize`、`prefill`、每轮或采样后的 `decode`、`stream` 建立 Span。日志按 `timestamp / level / request_id / trace_id / rank / device / event / error_code` 输出加载失败、OOM、CUDA Error、NCCL timeout、NaN、请求超时等事件。Metrics 聚合 QPS、错误率、OOM 次数、P50/P95/P99 TTFT/TPOT、waiting/running、KV 使用量，以及 GPU busy、显存使用、SM/Tensor Core 活动、DRAM 吞吐等；设备计数器能否采到及其定义依硬件和采集器而定。Trace 的单请求耗时解释与 Metrics 的总体分布需要对齐同一观测窗口和模型版本。

```text
request span: queue 10 ms → tokenize 2 ms → prefill 40 ms
           → decode 500 ms（attention / GEMM / sampling）→ stream 5 ms
```

若 GPU busy 低，先看请求 Trace 和 Nsight Systems 时间线：Kernel 之间若有空洞，查 CPU launch、同步、数据搬运和通信；若 GPU 持续有长 Kernel，再用 Nsight Compute 检查 DRAM 吞吐、Tensor Core、Occupancy 与 Warp Stall。单一 GPU busy 数字不能区分这些原因。生产环境控制 Trace 采样和日志量，并避免在日志、Span 属性或 Metric label 中记录原始 Prompt。

| 信号组合 | 优先调查 |
|---|---|
| waiting 上升、GPU 持续繁忙 | 负载超过容量、长请求 / 公平性、admission |
| waiting 上升、GPU 很空 | CPU 前处理、跨进程阻塞、通信 / IO、调度 |
| KV 接近容量、抢占增加 | 上下文 / 并发预算、缓存保留与重算 |
| TTFT 正常、TBT P99 变坏 | Decode 被长 Prefill / 通信 / 同步打断 |
| 错误率随请求类型变化 | 媒体限制、模板、Adapter、Schema 支持 |
| 重启或加载耗时升高 | 探针、镜像 / 权重存储、环境不兼容 |

```bash
curl -s http://127.0.0.1:8000/metrics > /tmp/vllm-metrics.txt
rg 'vllm:|vllm_' /tmp/vllm-metrics.txt
```

日志用 request ID 连接排队、执行、取消和响应；Trace 用相同关联标识追踪网关 / Engine / Connector。限制高基数 label，不把每个 Prompt 或 request ID 都塞进 Prometheus label。Histogram 聚合应合并 bucket 再算分位数，不能平均多个副本的 P99。

告警优先围绕错误率、SLO 违反、队列持续增长、OOM / 重启和容量趋势。GPU busy、Prefix 命中率是归因信号，不是用户体验的替代指标。
