# 9.5 配置调优：容量、Token Budget 与并发上限

| 参数 / 约束 | 限制什么 | 调大后要观察什么 |
|---|---|---|
| `max_num_seqs` | 本轮可运行序列数上限 | KV 容量、长尾请求、Decode batch |
| `max_num_batched_tokens` | 本轮计算 token 预算 | Prefill chunk、TTFT / TBT 与临时激活 |
| `max_model_len` | 单请求允许的上下文上限 | 长请求的容量与位置能力，不是全局 token 预算 |
| `gpu_memory_utilization` | 引擎显存预算的比例配置 | 可用 KV 块、其他进程空间；不是 GPU 算力利用率 |
| KV 可分配块 | 运行时容量约束 | 抢占、重算、Prefix 保留与临时投机槽位 |

Chunked Prefill 下较小的 token budget 可以减少长 Prefill 对已有 Decode 的干扰，却可能延后长 Prompt 的 TTFT；较大预算有利于更快推进 Prefill，也增加单轮耗时和峰值工作区。固定值不是跨 GPU 的最佳答案，按真实输入 / 输出长度与到达率扫描。[vLLM 调优说明](https://docs.vllm.ai/en/stable/configuration/optimization/)

**实验矩阵**：短入短出、长入短出、短入长出、混合长短请求；每组记录等待、TTFT、TBT P99、实际 batch token、抢占与 KV 占用。先判断是否被 KV 容量卡住，再判断是 Prefill 预算还是并发预算不足。
