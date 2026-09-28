# 13.2 系统架构：DistServe、Splitwise、TaiChi 与 Microserving

[DistServe](https://arxiv.org/abs/2401.09670) 围绕 TTFT / TPOT 约束优化 Goodput，联合选择阶段资源、并行策略和放置。[Splitwise](https://arxiv.org/abs/2311.18677) 强调两阶段不同的性能、功耗与成本特征，以及快速状态传输。两篇论文提供设计思路，其报告的加速比受实验硬件和工作负载约束。

[TaiChi](https://arxiv.org/abs/2508.01989) 将聚合与解耦统一考虑，按阶段能力和 SLO 分配请求，提示我们不必永久固定一种部署形态。[MLC Microserving](https://blog.mlc.ai/2025/01/07/microserving-llm-engines) 把引擎动作暴露为可组合 API：D侧 `prep_recv` 准备接收，P侧 `remote_send` 发送 KV，再在 D侧 `start_generate`；路由器可组织解耦、部分 Prefill 转移与上下文迁移。这些不是 vLLM Connector 的同名接口，阅读时区分研究系统、MLC 编排 API 和 vLLM 实现。

```text
Gateway / admission → P池调度 → Prefill
  → D侧预留 KV → Connector / KV transfer → 完成通知
  → D池生成 → Gateway stream → 双侧回收
```

工程上应先与 Chunked Prefill 的混合部署比较。高带宽网络、长输出与充足池规模可能有利于解耦；短请求、低负载或高传输成本下，新增网络和调度可能更贵。
