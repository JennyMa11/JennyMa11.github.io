# 13.6 实战：从官方 Connector 示例到混合负载验收

vLLM 将 disaggregated prefilling 标为实验功能，具体 Connector 支持与示例随版本变化。先选已支持的模型、KV 格式及拓扑，再按[官方 PD 示例](https://docs.vllm.ai/en/stable/features/disagg_prefill/)和[NIXL 配置指南](https://docs.vllm.ai/en/stable/features/nixl_connector_usage/)建立 P / D 服务与代理，保存实际生效的 `kv_transfer_config`。

| 步骤 | 验收输出 |
|---|---|
| 单机双实例 / 支持的 Connector | 正确输出、传输完成与槽位释放日志 |
| 跨节点 | 实际传输字节、有效带宽、TTFT 时间线 |
| 与混合服务对照 | 相同总设备 / 成本下的 Goodput 曲线 |
| 长短请求混合与突发 | P99、背压、取消和峰值 KV |
| 故障注入 | 传输中断、D侧 OOM / 退出后的重试或失败语义 |

当前环境没有执行多 GPU / 跨节点实验；表格是验收方案，不能当作性能结果。

---
