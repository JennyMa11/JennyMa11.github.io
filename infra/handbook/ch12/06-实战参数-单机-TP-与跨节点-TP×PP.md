# 12.6 实战参数：单机 TP 与跨节点 TP×PP

以下是 GPU 环境命令模板，模型路径、卡数与网络需替换，没有在当前机器运行：

```bash
# 单机两卡；模型结构 / 后端必须支持该 TP 数。
vllm serve /models/model --tensor-parallel-size 2 \
  --max-model-len 4096 --generation-config vllm

# 已建立 Ray 集群且至少有 4 张可用 GPU；核对实际 rank 放置。
vllm serve /models/model --tensor-parallel-size 2 \
  --pipeline-parallel-size 2 --distributed-executor-backend ray \
  --max-model-len 4096 --generation-config vllm
```

若希望在单机两卡上使用两个完整副本而非拆分一份模型，可在支持该模型的版本上使用内部 DP 负载均衡；每份副本都必须独立装下模型与 KV：

```bash
vllm serve /models/model --data-parallel-size 2 \
  --tensor-parallel-size 1 --max-model-len 4096 \
  --generation-config vllm
```

DP=2、TP=2 则至少需要 4 卡；`max_num_seqs` 是每 DP rank 的预算，不等于全服务 admission 上限。Dense 的独立服务也可由外部路由器分流；MoE 的 DP+EP 仍有专家层协作，外部 DP 启动语义另查支持矩阵。[vLLM DP 部署](https://docs.vllm.ai/en/stable/serving/data_parallel_deployment/)

按安装版本检查 `vllm serve --help`。先验证单请求数值与终止语义，再比较相同 workload 的 TTFT、ITL、Goodput、KV 容量及 collective 占比；故障测试包含 rank 退出和请求取消。DP / EP 的启动选项及支持矩阵另查对应版本官方指南，不能从上述 Dense 模板直接推断。

---
