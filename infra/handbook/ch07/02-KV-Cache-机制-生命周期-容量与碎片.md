# 7.2 KV Cache 机制：生命周期、容量与碎片

**速查**：显存 = 模型权重 + KV Cache + Activation + CUDA Graph Buffer + Workspace（+ 框架开销）。

| 组成 | 说明 | 随什么增长 |
|---|---|---|
| **Model Weight（模型权重）** | 模型参数，固定 | 模型大小、精度 |
| **KV Cache（键值缓存）** | 历史 K/V，最大变量 | 并发数 × 序列长度 |
| **Activation（激活值）** | 前向中间结果 | Batch × 序列长度（Prefill 峰值高） |
| **CUDA Graph Buffer（图缓冲）** | Graph 需要的静态输入/输出缓冲 | Graph 数量 × Batch 桶 |
| **Workspace（工作空间）** | Kernel 临时空间、通信缓冲 | 算子与并行策略 |

**在线 Serving 的关键结论**：随着并发增加，**KV Cache 很容易超过模型权重**，成为显存瓶颈。所以推理框架通常按「剩余显存」动态给 KV Cache 定容（如 vLLM 的 `gpu_memory_utilization` 参数）。

**术语解释**：

- **Serving（在线服务）**：把模型部署成可并发接收请求、流式返回结果的服务系统。

KV 在 Prefill 时写入、Decode 时追加；请求结束或取消后释放引用，前缀缓存可保留无活跃引用的块等待复用。共享块、空闲块和缓存块不是互斥集合：是否可复用取决于引用计数与缓存策略。分页消除大段连续分配需求，仍有尾块浪费、元数据和预留容量。详细寻址与状态机见[第 6 章](ch06.html)。
