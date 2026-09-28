# 23.6 把论文结论变成可执行的适配和 Profiling 清单

| 新机制 | 接入时新增的状态或 Kernel | 首要验证 |
|---|---|---|
| DSA / CSA / CSA2 | Indexer、Top-k、Gather、压缩主 KV、跨层引用 | 选中索引、mask/position、长上下文 logits；分开计时索引与主 Attention |
| HCA + SWA | 不同压缩率、窗口 KV、尾部状态 | `m−1/m/m+1` 和 `m′−1/m′/m′+1` 边界；释放与前缀复用 |
| CED + Bounded Replay | Encoder 输出投影、Decoder 局部重算 | 冷/热与不同缓存命中位置的数值差异、TTFT、Replay 耗时 |
| FP4 主 KV | 打包值、scale、反量化与分配器格式 | 压缩后的真实字节、逐层误差、DRAM 流量、长文质量 |
| MoE/EP 与 DSpark | 专家路由/通信、Draft KV、候选预算 | 各 rank 次序、通信暴露时间、接受率、每有效 Token 成本 |

**最小实验**：先固定模型 revision、权重、Tokenizer、dtype、硬件和请求集；构造短 Prompt、长 Prompt、长输出、重复前缀与多轮 Agent 五组负载。逐项记录 `TTFT / TPOT P50-P99 / token/s / HBM KV / 主机或 SSD KV / indexer 与 gather 耗时 / 压缩与 replay 耗时 / EP 通信占比 / 输出质量`。先做单请求逐层对齐，再测多请求容量和 Goodput；用 Nsight Systems 定位空洞，用单 Kernel Profiler 解释热点。论文报告数字作为待复现假设，测得结果连同质量代价和资源分母一起写出。

**面试追问**：为何 DSA 的 `O(Lk)` 主 Attention 不代表总复杂度是 `O(Lk)`？为何 V4.1 的 890 字节与八分之一不矛盾？为什么 SWA Bounded Replay 需要数值验收？回答时分别指出 Indexer、缓存层级与近似重算。

---
