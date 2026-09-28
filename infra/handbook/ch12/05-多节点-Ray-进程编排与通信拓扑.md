# 12.5 多节点 Ray：进程编排与通信拓扑

Ray 负责资源发现和分布式执行编排，GPU 数据路径仍由执行后端、collective 和互联决定。各节点固定相同镜像、模型 revision、驱动兼容性与挂载路径；检查 GPU 可见性、placement、地址可达性和 NCCL 网络选择。

```text
验证顺序：节点 / GPU 枚举 → rank 放置 → 两 rank collective
  → 小模型跨节点生成 → 目标模型 → 并发 / 取消 / 故障
```

先核对官方 Ray 集群启动步骤，再从 head 节点启动服务；设置 TP×PP 的资源需求，确认 rank 实际落点。Ray 显示节点存活不证明 RDMA 数据通道正常。保存 topology、通信基线与所有 rank 的日志。[vLLM 多节点指南](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/)
