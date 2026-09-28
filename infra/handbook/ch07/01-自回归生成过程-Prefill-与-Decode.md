# 7.1 自回归生成过程：Prefill 与 Decode

### 单请求的真实执行账本

```text
prompt token_ids 长度 P
Prefill: 输入 P 个 token → 写 P 份 KV → 只取最后位置 logits → 采样 y0
Decode: 输入 y0(position=P) → 写第 P+1 份 KV → 采样 y1
... → EOS / max_new_tokens / stop 条件 → 释放或保留可复用 KV
```

```python
def generate_one(prompt_ids, model, kv, max_new_tokens):
    logits = model.forward(prompt_ids, positions=range(len(prompt_ids)), kv=kv)
    token = sample(logits[-1])
    for step in range(max_new_tokens):
        yield token
        if token == EOS or step + 1 == max_new_tokens:
            break
        logits = model.forward([token], positions=[len(prompt_ids) + step], kv=kv)
        token = sample(logits[-1])
```

这是接口伪代码，模型必须区分写入新 KV 与读取历史 KV，且处理空 prompt 与特殊 token。性能上，Prefill 大 GEMM、Attention 计算与激活显存更显著；Decode 单 token 小矩阵和历史 KV 读取常受带宽与 launch/调度开销影响，**具体界限仍需按 Batch 和硬件测量**。用两组实验验证：固定输出长度增加 prompt，观察 TTFT；固定 prompt 增加并发，观察每步有效 token/s 与 TPOT。

### Prefill 阶段

**速查**：一次性并行处理整个 Prompt，计算密集，决定 TTFT，同时把 K/V 写进 KV Cache。

**原理**：

- 输入是 `[num_tokens, hidden]`，所有 Token 位置已知，Attention 可以用 **Causal Mask（因果掩码，保证每个位置只能看到自己及之前的 Token）** 并行算。
- 输出只需要**每个序列最后一个 Token 的 Logits**（用于采样第一个新 Token），所以 LM Head 的开销是 `O(num_seqs)` 而非 `O(total_tokens)`——这是很常见的优化点。
- 长 Prompt 的 Prefill 会瞬间占用大量显存（Activation + KV Cache），所以需要 Chunked Prefill 兜底。

**术语解释**：
**码 1-1 · Prefill 的 LM Head 只算每序列最后一个 Token**

```python
# packed varlen 批：hidden 是 [total_tokens, hidden]，cu_seqlens 是各段边界
last_idx = cu_seqlens[1:] - 1          # 每条序列最后一个 token 在 packed 张量里的下标
logits = lm_head(hidden[last_idx])     # [num_seqs, vocab]，开销 O(num_seqs)

# 反例：写成 lm_head(hidden) 会得到 [total_tokens, vocab]
# 长 prompt 下 99% 以上的 logits 永远不会被采样到，纯属白算
```

> **为什么值得看代码**：`vocab` 通常 10 万级，这一行把 LM Head 的 FLOPs 从 `O(total_tokens × vocab)` 降到 `O(num_seqs × vocab)`——长 Prompt 下是数量级的差别，而工程代价几乎为零。面试里这是很好用的「小改动大收益」例子。

- **Activation（激活值）**：前向传播过程中产生的中间张量，需要临时占用显存。
- **Batch（批）**：一次前向同时处理的请求集合；**num_seqs** 指批内序列数。

**追问：Prefill 能怎么加速？**
> 三条路径：① 算子层——用 packed varlen（**variable length，变长序列打包**）Kernel 一次 launch 覆盖整个 batch，减少 Python 循环和 Kernel Launch；② 消除 GPU→CPU 的 scalar 同步（`.item()` 是同步点）；③ 减少实际要算的 Token——Prefix Cache 命中后只 Prefill 增量。

### Decode 阶段

**速查**：每步只处理 1 个新 Token（每请求），复用历史 KV Cache，访存密集，决定 TPOT。

**原理**：

- 新 Token 的 Q 只有 1 行，但需要和**全部历史 K/V** 做 Attention，所以必须维护 KV Cache。
- 每步计算量小、Kernel 短、形状固定 → **Kernel Launch（核函数启动）开销占比高**，这正是 CUDA Graph 的用武之地。
- 每步都要把全部权重读一遍 → **量化收益最大的阶段**。

**术语解释**：
**码 1-2 · Decode 单步到底在算什么**

```python
# x 是当前 token 的 hidden state；cache 里是这条序列的全部历史 K/V
q = proj_q(x)                                   # [1, H_q, D]  ← 只有 1 行
k_new, v_new = proj_kv(x)                       # [1, H_kv, D]

k_cache = torch.cat([k_cache, k_new], dim=0)    # [S+1, H_kv, D]  ← 用显存换计算
v_cache = torch.cat([v_cache, v_new], dim=0)

scores = (q @ k_cache.transpose(-1, -2)) / math.sqrt(D)   # [1, H_q, S+1]
out = torch.softmax(scores, dim=-1) @ v_cache             # [1, H_q, D]
```

这三行代码解释了 KV Cache 的全部必要性：

1. `q` 只有 1 行 → **计算量是 O(S)**，很小，GPU 算得飞快；
2. 但 `k_cache` / `v_cache` 每一步都要**全量读一遍** → **访存是 O(S)**，且 S 随生成长度增长；
3. 计算小、访存大，正好落在「访存受限」区间 → 所以 Decode 的瓶颈是带宽不是算力。

> 注意 `cat` 只是示意：真实实现不会每步分配新张量，而是用 PagedAttention 按槽位**原地写入**（见码 3-2）。`cat` 写法在长序列下会因为反复分配/拷贝显存而直接崩掉。

- **Kernel（核函数）**：在 GPU 上并行执行的函数，一次启动称为 **Kernel Launch**；每次启动都有固定的 CPU→驱动→GPU 提交开销。
- **自回归（Autoregressive，AR）**：每一步的输出作为下一步的输入，天然串行。

**追问：Decode 为什么不能像 Prefill 一样并行？**
> 因为自回归依赖：第 t 个 Token 的输入是第 t−1 个 Token 的输出，天然串行。投机解码就是用来打破这个串行链的（一次验证多个）。
