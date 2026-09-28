# 第 0 章 全局地图：从请求到 GPU 与 Token

> **目标**：沿同一条请求链路学习硬件、算子、缓存、调度、分布式、性能与正确性。每章遵循「速查 → 原理 → 执行图 → 伪代码/最小实现 → 工业源码 → 性能 → Debug → 追问」。示例分为**可运行最小实现**与**机制伪代码**；伪代码不宣称可直接生产部署。
>
> **阅读方法**：先运行 CPU/PyTorch reference 并记录 shape、dtype、误差；再写 CUDA/Triton Kernel；接着把算子接到简化推理循环；最后用同一批输入做 Profiling 与逐层比对。没有 NVIDIA GPU 时可以完成 Python 路线，GPU 章节的 CUDA 示例需要相应环境。
>
> **工程约定**：以下源码路径以链接所指版本为准；框架实现变化较快，阅读时先定位入口和调用关系，再查看当前分支。性能结论都依赖硬件、shape、batch 和并发，必须实测。
> **图码编号**：既有配图和代码示例沿用原编号，章节重排后编号不一定等于所在章号；以当前小节标题定位内容。

推理部分按 [AIInfraGuide · 模块四「推理优化」](https://github.com/caomaolufei/AIInfraGuide/tree/a3b63eeb81d6d36a3c42c8cfc5a1bdd96e36bab1/docs/guides/%E6%A8%A1%E5%9D%97%E5%9B%9B-%E6%8E%A8%E7%90%86%E4%BC%98%E5%8C%96) 的 12 章大纲组织，对应本手册第 7–18 章。参考快照：2026-09-27，commit `a3b63ee`。本手册在对应小节中保留原有实现与推导，并补充工程账本、配置模板和验证方法；API 以各小节链接的官方文档及安装版本为准。

## 学习路线与交付物

```text
体系结构 → GPU/CUDA → Memory → 并行算法 → GEMM/Softmax
  → Transformer 算子 → Attention/FlashAttention → KV/PagedAttention
  → LLM 基础 → 引擎技术 → vLLM 源码 → 量化 → 投机解码
  → 分布式 → PD 解耦 → 生产特性 → Benchmark → 运维 → 端到端 → 端侧
  → Debug → Bug → 能力验收
  → GPU/NPU 模型适配与验收
```

| 阶段 | 完成后应能交付 |
|---|---|
| 1–3 | 向量 Kernel、Reduction、Softmax、GEMM；附数值和带宽基线 |
| 4–6 | RMSNorm、RoPE、Attention、分页 KV；说明每个张量的布局与生命周期 |
| 7–18 | 推理优化模块：12 章沿用参考仓库大纲；交付调度状态机、量化 / 投机对照、SLO 压测与部署方案 |
| 19–21 | 一份 Profiling 报告、一份逐层数值定位记录、一份故障复盘 |
| 22 | 一份跨设备模型适配记录：环境矩阵、算子清单、数值对齐、性能与稳定性验收 |
| 23 | 一份前沿技术报告精读：逐项拆解新架构的计算、KV、通信、精度与可验证收益 |

---


> **承上**：先看整条请求链路，明确每层为什么存在。
