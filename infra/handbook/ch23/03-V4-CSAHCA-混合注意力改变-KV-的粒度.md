# 23.3 V4：CSA/HCA 混合注意力改变 KV 的粒度

V4 的 **CSA（Compressed Sparse Attention）**先把约 `m` 个历史 Token 压成一个 KV 条目，再在压缩条目上使用索引器选 Top-k；**HCA（Heavily Compressed Attention）**用更大的 `m′` 压缩比例，在更短的压缩序列上做密集 Attention。两类层交错，并用 Sliding Window Attention（SWA）保留局部细节。CSA 的 `Top-k` 与 HCA 的更高压缩率解决不同问题；HCA 不执行 CSA 的稀疏选择。[V4 原文 §2.3](https://arxiv.org/html/2606.19348#S2.SS3)

```text
原始历史长度 L
CSA：压缩后约 L/m 条 → Indexer Top-k → 主 Attention 读 k 条 + 本地 SWA
HCA：压缩后约 L/m′ 条（m′≫m）→ 在短序列上密集 Attention + 本地 SWA
```

**缓存实现比公式多一层**：CSA/HCA 的层有不同的压缩率与条目大小；SWA 只保留窗口；尚不够组成一个压缩组的尾部 Token 及隐藏状态也要暂存。V4 把压缩 KV 与 SWA/未压缩尾部状态分开管理，完整压缩块按 `lcm(m,m′)` 个原始 Token 对齐，并让稀疏 Kernel 与块布局共同设计。共享前缀从磁盘命中时，完整压缩块可复用；尾部不足一组的 Token 需恢复或重算。只把传统 PagedAttention 的一个固定 `[层, token, head, dim]` 块表套过来，会漏掉不同层的更新和淘汰规则。[V4 原文 §3.5](https://arxiv.org/html/2606.19348#S3.SS5)

在报告给定的**百万 Token、V4-Pro 对 V3.2**比较中，作者报告单 Token 推理 FLOPs 为 27%、KV Cache 为 10%；这是特定模型与上下文的报告值，不能解释为通用请求端到端延迟缩短 73%。[V4 摘要](https://arxiv.org/abs/2606.19348)

V4 的训练和 Kernel 也值得读：**mHC**通过约束残差映射（例如双随机矩阵）控制深层信号传播；**Muon**用于矩阵类参数，其他部分仍用 AdamW；EP 将专家 Dispatch/Combine 与两段计算做细粒度流水重叠。报告还介绍 TileLang 开发、批次不变且确定性的 Kernel，以及用于前缀复用的磁盘 KV 缓存。判断收益时分别检查训练稳定性、Kernel 正确性、通信暴露时间与真实 Serving 指标；模型能力分数不能代替这些系统证据。[V4 原文 §2.2、§2.4、§3](https://arxiv.org/html/2606.19348)
