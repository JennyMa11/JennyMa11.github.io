# 11.3 Self-Draft：Medusa、EAGLE-2/3 与 Draft Tree

这里把复用 Target 表示的方案归为 Self-Draft 学习路线；**不意味着完全没有额外参数**。Medusa 有训练后的预测头，EAGLE 有轻量 Draft 网络，都需要与 Target 匹配。

### 11.3.1 Medusa：多个 Head 预测多个未来位置

在主干 hidden state 上增加多个解码头，分别预测未来不同位置；每个位置取若干候选组合成树，再由 Target 验证。预测头轻，但多个未来位置的独立预测未必构成一条高概率自回归序列，候选树用于覆盖不同组合。

```text
Target hidden → head₁ / head₂ / head₃ → 多位置候选
              → 候选树 + Tree Attention → 验收路径 + 提交 KV
```

Head 训练、候选预算和验收模式要分别交代。Medusa 论文中的 typical acceptance 属于近似验收，不能把所有 Medusa 设置都声称为严格保持 Target 分布。[Medusa 论文](https://arxiv.org/abs/2401.10774)

### 11.3.2 EAGLE-2：用置信度动态分配候选树

EAGLE 的早期路线复用 Target 特征和 token 信息，递归预测特征并产生候选。EAGLE-2 的主要新增点是 **context-aware dynamic draft tree**：根据当前上下文的提议置信度分配候选节点，避免每个请求都使用同样的固定树。

高置信路径可以更深，低置信位置可以改变扩展宽度；节点预算仍有限，目标是把验证计算花在更可能被接受的路径上。论文利用 Draft 置信度与接受概率较接近的观察，但这不是对所有任务恒成立的定理，应检验业务域中的校准质量。[EAGLE-2 论文](https://arxiv.org/abs/2406.16858)

### 11.3.3 EAGLE-3：多层特征融合与直接 Token 预测

EAGLE-3 移除显式 feature prediction 的约束，使用 Target 的多层特征融合并直接预测 token；训练中模拟推理时多步递归的过程（training-time test），让 Draft 学习自身提议带来的输入变化。

它继承动态候选树思路，但主要贡献不能只写成“更大的树 / 置信度校准”。特征提取层、训练方式和 checkpoint 都需要与 Target 对齐；不同 EAGLE 版本也不能随意交换辅助权重。[EAGLE-3 论文](https://arxiv.org/abs/2503.01840)

### 11.3.4 Tree Attention：共享前缀，隔离兄弟分支

Medusa / EAGLE 等方案可构造候选树：根的多条子路径共享前缀，一次 Target 前向验证多个节点。Tree mask 必须让节点只看到正式上下文和自己的祖先，不能看到兄弟节点；position id 按**路径深度**设置，不能按 flatten 后数组位置直接递增。

```text
ctx ─ a ─ b
        └ c ─ d
节点 d 可看 ctx,a,c,d；不能看 b。
最终选择路径 a,c,d 后，只提交该路径对应的 KV。
```

增加候选宽度能提高命中机会，也增加验证 token 数、mask / metadata 和临时 KV。经典单链的 `min(1,p/q)` 不能不加说明地套到任意候选树；必须根据实际提议过程采用匹配的验收算法。Medusa 的 typical acceptance 是另一种验收模式，不享有上述精确随机采样保证。[Medusa 论文](https://arxiv.org/abs/2401.10774)

### 11.3.5 从 Token 到 Block：并行提出与联合验证

单链 Speculative Sampling 已经在一次前向验证一段候选，但按位置接受前缀；多头 / 并行 Draft 进一步减少提议侧串行次数。**一次算多个位置、构造候选树、从联合分布验收整个 block**是不同概念，不能混为同一种算法。

[Block Verification 论文](https://arxiv.org/abs/2403.10444)进一步联合利用整段候选的 Draft / Target 概率来决定可接受前缀和修正分布，而不是每个位置独立使用经典验收。论文在其验证问题的假设下证明每轮期望产出的最优性，并保持 Target 分布；它不是“整块全收或全拒”的简单二元规则。

学习时对比三件事：经典方法在首次拒绝处停止；Block Verification 利用后续候选的概率信息改善可接受前缀；两者都必须输出经过正确修正的后续 token。联合验收也不要求 Draft 一定并行生成。具体算法不能把联合残余分布随意换成逐位置 softmax，工程收益仍应把额外验收计算纳入计时。

MTP 也可从模型内置预测模块生成未来候选，但模块需训练与模型支持，仍要 Target 验收。方案演进的共同目标是降低提议串行成本并提高验证利用率，正确性与工程状态管理始终单独证明。

### 11.3.6 各方法对比


| 方法 | 额外状态 / 参数 | Draft 成本来源 | 主要约束 | 适合验证的场景 |
|---|---|---|---|---|
| Draft Model | 独立小模型与 KV | 自回归小模型前向 | tokenizer / 提议分布对齐，额外显存 | 通用生成 |
| MTP | 内置预测模块 | 特征复用与预测模块 | 模型需支持；预测头不自动保证验收正确 | 支持 MTP 的模型 |
| Medusa | 多个训练后的 Head | 多头预测与候选树 | tree mask / KV 路径提交；验收模式影响正确性 | 多候选验证 |
| EAGLE-2/3 | 轻量 Draft 网络 | 特征复用与 token 提议、树扩展 | 特征接口、训练、候选预算 | 有匹配 checkpoint 的模型 |
| N-gram / Lookup | 历史匹配状态 | 查找与候选拷贝 | 重复率；确定性提议也需匹配验收规则 | 有重复片段的代码 / RAG |

接受率不是由方法名称决定的固定等级；需比较实际任务、Draft 长度、候选预算、采样温度与训练质量。
