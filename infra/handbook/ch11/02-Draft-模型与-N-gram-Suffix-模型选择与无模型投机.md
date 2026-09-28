# 11.2 Draft 模型与 N-gram / Suffix：模型选择与无模型投机

### 11.2.1 独立 Draft：小只是起点，分布匹配才决定有效产出

独立小模型自回归提出候选，Target 验证。选择时一起比较 **提议耗时、接受长度、额外权重 / KV 显存**，不要只看参数量。一个稍大的 Draft 若显著提高每轮有效输出，可能比极小模型更划算。

| 检查项 | 具体问题 | 影响 |
|---|---|---|
| 大小与执行 | 每步 Draft 延迟、γ 步总耗时、是否需要额外 TP 通信 | 小模型也可能受 launch 或通信限制 |
| 架构与词表 | token ID / tokenizer 是否一致，后端是否有适配 | 不能把不同词表的概率直接相除 |
| 能力与任务 | 代码、中文、对话、推理与 Target 是否匹配 | 单独榜单分数高不等于提议分布接近 |
| 上下文 | Draft 最大长度、位置编码、长上下文能力 | 到长度边界可能退化或无法继续提议 |
| 训练与采样 | Base / Instruct、chat template、温度与约束 | 改变 q 与 p 的重叠和接受长度 |
| 资源 | Draft 与 Target 权重、两套 KV、验证临时槽位 | 原来的可并发请求数可能下降 |

同模型家族、相同 tokenizer 是便于验证的起点，但不是充分条件。若采用跨词表映射，需要实现匹配的提议与验收流程；vLLM 的对应支持有额外限制，应按版本核对。[vLLM Draft Model 文档](https://docs.vllm.ai/en/stable/features/speculative_decoding/draft_model/)

### 11.2.2 N-gram / Prompt Lookup：用历史片段提出候选

取当前上下文末尾的 n 个 **token**，在 Prompt 或可用历史中寻找相同片段，把匹配处后续 token 当作候选。比如历史中曾出现 `def add(a,b): return a+b`，末尾再次匹配到函数开头时，就可以提议后续片段；实际匹配对象是 token ID，不是字符串单词。

```text
当前后缀 → 查找历史中的相同 n-gram → 取后续 γ 个 token
       → Target 验证 → 保留接受前缀 → 不匹配处修正
无匹配 → 普通 Decode
```

n 太短容易撞上无关片段，n 太长又减少命中；候选长度也受匹配处剩余 token 限制。查找、拷贝与验证都仍有成本，不能说“零开销”。重复代码、引用已有文本、模板输出更适合尝试，是否加速由实际复用程度决定。[vLLM N-gram 配置](https://docs.vllm.ai/en/stable/features/speculative_decoding/n_gram/)

**确定性提议也可精确验收**：若提出固定 token d，则 q 是在 d 上的点质量。随机采样接受概率为 `p(d)`，拒绝后从去掉 d 并归一化的 Target 残余分布采样；贪心则比较 Target argmax。无模型只省去训练 / 小模型前向，验收机制仍不可省。

### 11.2.3 Suffix Decoding：从单次匹配到统计后续分支

Suffix 方法根据当前后缀与历史的匹配，在后缀索引 / 树中组织后续 token，并利用出现统计与置信信息选择提议深度或分支。它同样无需额外神经 Draft 模型，但有索引构建、查找、缓存与更新成本。

vLLM Suffix 配置包括树深度、缓存请求数、提议长度上限与最小估计概率。全局历史缓存可能帮助重复业务，也消耗内存；比较实验要记录是否启用以及热身顺序，否则不同方案看到的历史可能不同。具体提议与验收行为以版本实现为准。[vLLM Suffix 文档](https://docs.vllm.ai/en/stable/features/speculative_decoding/suffix/)
