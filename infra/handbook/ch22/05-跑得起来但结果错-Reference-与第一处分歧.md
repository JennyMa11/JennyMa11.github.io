# 22.5 跑得起来但结果错：Reference 与第一处分歧

先固定输入、权重、tokenizer、位置、mask、随机种子与采样方式，比较**同一次 Forward 的 logits**。如果不同，按 Layer 输出二分定位第一个异常层，再拆成 RMSNorm → QKV → RoPE → Attention（score、softmax、输出）→ MLP → residual；Decode 还要比对每一步新写入的 K/V、block table 和历史长度。这里的“二分”是减少人工检查范围；若误差缓慢累积，还需看每层误差曲线与容差，而不能只选一个布尔失败层。

| 指标 | 计算/解释 | 注意点 |
|---|---|---|
| 最大/平均绝对误差 | `abs(ref−test)` | 适合观察整体规模与极值 |
| 相对误差 | `abs(ref−test)/(abs(ref)+ε)` | reference 接近零时单独看会夸大 |
| Cosine Similarity | 向量方向一致性 | 高相似度仍可能掩盖局部大错 |
| `allclose` | 按 `atol + rtol × abs(ref)` 判定 | 容差须按 dtype/算子设定 |
| Logits/Top-K/PPL | 模型级与任务级表现 | 采样微扰会放大为不同生成文本 |

先查系统性错误：权重转置、维度广播、RoPE 排列、mask 方向、scale 轴、量化 zero-point、累计精度；再查偶发错误：未初始化读、越界、Stream 竞争、KV 共享/释放。对 FP16/BF16，记录累加 dtype 与融合前后运算顺序。**验收不能仅看最终 token 是否完全一致**：接近并列的 logits 可能因微小浮点差异改变采样路径，但任务质量仍可接受；反过来，错误也可能暂未改变 argmax。
