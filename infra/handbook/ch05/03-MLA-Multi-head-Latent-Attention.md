# 5.3 MLA（Multi-head Latent Attention）

**速查**：MLA 把 K/V 联合压缩成低维 **latent（潜在向量，此处指压缩后的低维表示）** 并缓存，同时保留模型所需的位置分量。它能显著减少特定模型的 KV 字节；压缩率和质量需按架构与任务评估，Kernel 可通过吸收投影避免完整物化 K/V。

**术语解释**：

- **MLA（Multi-head Latent Attention，多头潜在注意力）**：DeepSeek-V2 提出的注意力结构，通过低秩压缩 KV 表示。
- **latent vector（潜在向量）**：压缩后的低维表示，不是直接可用的 K/V，需要经上投影矩阵还原。
- **上投影 / 下投影（Up / Down Projection）**：下投影把高维压到低维，上投影把低维还原到高维。
- **decoupled RoPE（解耦旋转位置编码）**：把位置信息单独放在一小段带 RoPE 的维度上，避免位置编码破坏压缩结构。

**原理**：

- 传统做法缓存 `K` 和 `V` 两个 `[head_dim × num_heads]` 张量；MLA 缓存低维 `c_KV` 和必要的位置分量，读取时用投影或等价的矩阵吸收路径参与 Attention。
- 相比 MHA，特定 DeepSeek 配置的 KV payload 可大幅降低；应以 `latent 维度 + 位置维度 + scale/元数据` 计算真实缓存字节，不能把单一压缩比套到其他模型。

**追问：MLA 和 GQA 谁更好？**
> 目标不同：GQA 是「共享 K/V Head」，实现简单、通用性好、算子成熟；MLA 是「压缩表示」，新增投影计算及专用 Attention 路径，实际 Kernel 可用矩阵吸收减少显式还原。MLA 需要模型在训练时采用该结构，不能给已有 GQA checkpoint 直接换 Kernel。DeepSeek 从训练阶段就设计了 MLA。

DeepSeek-V3.2 之后又在 MLA 之上加入稀疏选择，V4 系列继续改变缓存粒度与跨层复用；机制和验证方法见[第 23 章](ch23.html)。
