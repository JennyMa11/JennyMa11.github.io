# 23.5 同期对照：Qwen3.8-Next 的另一条路径

[Qwen3.8-Next 架构报告（2026-08）](https://arxiv.org/abs/2608.30320)提供另一种长上下文方案：以 **Gated DeltaNet（GDN）固定大小循环状态**承担多数层的 Token 混合，每四层保留一层全局 Attention，并在后续训练中将其改为按 micro-block 选择的 Qwen Sparse Attention（QSA）。报告另用 Gated Residual（GR）扩大残差流，用从主机内存预取的 n-gram Embedding 增加容量。固定状态降低 Decode 缓存增长，但全局层、索引器与主机预取仍会产生成本；适配时要把循环状态、Attention KV 与外置 Embedding 分开管理。

与 DeepSeek 的 CSA/HCA/CSA2 相比，两者都在压缩长历史和减少主 Attention 工作量，但压缩对象不同：Qwen 的 GDN 把历史写入循环状态，DeepSeek V4 系列保留可检索的压缩 KV 条目。这是从原报告结构作出的系统层归纳；哪种方案更快须在**相同硬件、模型质量要求、长度分布与缓存命中条件**下实测，不能横比不同论文的加速数字。[Qwen3.8-Next 原文](https://arxiv.org/html/2608.30320)
