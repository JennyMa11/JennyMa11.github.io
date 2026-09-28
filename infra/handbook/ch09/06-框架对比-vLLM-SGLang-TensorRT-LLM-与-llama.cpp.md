# 9.6 框架对比：vLLM、SGLang、TensorRT-LLM 与 llama.cpp

| 框架 | 定位 | 核心特性 | 适合场景 |
|---|---|---|---|
| **vLLM** | 通用 LLM Serving | PagedAttention + Continuous Batching + Scheduler | 通用在线服务、研究基线 |
| **SGLang** | 强调前缀复用与 Agent | RadixAttention、Prefix 复用、结构化输出 | Agent、多轮对话、高前缀复用 |
| **TensorRT-LLM** | NVIDIA 推理栈 | TensorRT、CUDA Graph、**In-flight Batching**、低精度 Kernel | NVIDIA GPU 上的优化与生产部署 |
| **llama.cpp** | 本地 / 端侧 | **GGUF（GPT-Generated Unified Format，一种含量化权重的单文件模型格式）**、低比特量化、CPU/GPU/Metal 跨平台 | 本地推理、端侧、Mac |

**术语解释**：

- **In-flight Batching（飞行中批处理）**：TensorRT-LLM 对 Continuous Batching 的叫法。
- **Agent（智能体）**：能自主调用工具、多轮规划与执行的 LLM 应用形态，其请求通常有很长且高度重复的上下文。
- **结构化输出（Structured Output）**：用正则或 JSON Schema 约束解码过程，保证输出符合指定格式。

**追问：vLLM 和 SGLang 的区别？**
> 两者都基于 PagedAttention + Continuous Batching。SGLang 的差异点在于 **RadixAttention**（用基数树做更灵活的前缀复用，覆盖多轮/Agent 场景）和**结构化输出**（约束解码），具体收益取决于前缀分布、后端与版本，应在同一工作负载下比较两者的命中率、Goodput 和成本。

---
