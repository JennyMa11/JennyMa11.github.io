# 12.2 Tensor Parallel：矩阵切分与通信账本

对 `Y=XW`，Column Parallel 沿 W 的输出维切分，得到局部输出；后续 Row Parallel 沿输入维切分，局部矩阵积求和还原输出。Attention Head 切分和 GQA 的 KV Head 划分需符合后端约束。

```text
列切分：W=[W0 W1] → Y0=XW0、Y1=XW1
行切分：W=[W0;W1]、X=[X0 X1] → Y=X0W0+X1W1
T_TP ≈ 局部计算 + collective 关键路径 + 输入组织
```

权重容量通常近似除以 TP 数，复制参数、workspace 与 KV 分片例外要另算。低 batch Decode 的 GEMM 已很小，增大 TP 可能被 collective 启动延迟抵消；先在同一高速互联域测试 TP=1/2/4，而不是用卡数预测线性加速。[vLLM 并行与扩展](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/)
