# 9.3 V1 引擎：进程划分、统一预算与 Persistent Batch

```text
HTTP / Tokenize → Async 前端 → EngineCore 请求队列
  → Scheduler 本轮计划 + KV 分配 → Worker / ModelRunner
  → Attention Backend / Linear Kernel → Sampler
  → 更新 computed/output 状态 → Detokenize / stream response
```

前端与执行核心分离的目的之一是让 CPU 处理与设备执行重叠。持久批次保留运行请求及部分设备缓冲，每轮用增量更新减少重建；它不能消除 tokenization、sampling、序列变更和跨进程传输成本。

阅读源码时追踪三条记录：请求何时进入 / 离开队列；本轮 scheduled tokens 与 computed tokens；物理 KV 引用和释放。跨进程消息延迟、调度耗时与 GPU 执行耗时分开计时。框架版本重构会改类名和入口，按固定 commit 记录真实路径。[vLLM V1 设计](https://docs.vllm.ai/en/stable/usage/v1_guide/)

V1 用请求的已计算进度与目标 token 数之差组织调度，而不必给每个请求维护独立的 Prefill / Decode 执行循环。候选 token 也进入预算，实际接受数量决定提交进度。Persistent Batch 的优化目标是减少重复输入组织和分配；请求结束、抢占、迁移和缓存命中仍需显式更新状态。
