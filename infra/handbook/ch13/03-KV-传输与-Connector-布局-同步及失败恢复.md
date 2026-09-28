# 13.3 KV 传输与 Connector：布局、同步及失败恢复

解耦并非仅给两个服务加标签：Prefill 侧必须产出可消费的 KV，Decode 侧分配目标槽位，Connector 传输并完成依赖同步，路由器再把生成进度接起来。模型 revision、KV dtype、层 / Head 划分、TP 分片和位置状态都要匹配。

```text
传输下界 T_transfer ≥ KV_payload / effective_link_bandwidth
请求 TTFT ≈ P侧排队 + Prefill + 传输关键路径 + D侧排队 + 首次生成
网络需求 ≈ 到达率 × 平均每请求实际传输的 KV 字节
```

教学例：每请求传 `2 GiB`，有效带宽 `50 GiB/s`，单次传输下界 `40 ms`；到达率 `20 req/s` 产生 `40 GiB/s` 平均数据流，尚未包含复制、同步和竞争。若新增传输与排队超过减少的阶段互扰，Goodput 可能下降。可重叠传输要用时间线验证，不能简单把所有耗时相加或全部忽略。

池配比单独见 13.5。

| 失败点 | 状态处理 | 观察项 |
|---|---|---|
| 目标槽位不足 | 先预留 / admission，避免无处写入的大传输 | D池 KV、等待长度 |
| 传输超时 / 中断 | 丢弃不完整状态、释放两侧引用；按策略重算或失败 | 字节数、重试、request ID |
| 请求取消 | 中止后续调度，完成安全回收 | 孤儿块、延迟释放 |
| 布局不兼容 | 启动校验或转换支持；禁止按错误 layout 读取 | 分片、dtype、scale |

部署时从 Connector 官方示例验证单机，再测跨节点网络与混合 workload；模型已加载、网络连通、服务可用三者要分别检查。[vLLM NIXL Connector](https://docs.vllm.ai/en/stable/features/nixl_connector_usage/)
