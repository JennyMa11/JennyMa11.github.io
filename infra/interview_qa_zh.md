# 面试问答准备：nano-vLLM 推理引擎优化

> 使用方式：每题先给「30 秒版」——面试官问完先说这段，把骨架立住；被追问时再用
> 「展开版」。所有数字都标注了仓库出处，可以当场翻给面试官看。
>
> 环境口径统一为 **RTX 3060 Laptop 6 GiB + Qwen3-0.6B BF16**。根 `README.md`
> 里出现的 "RTX 4070 Laptop" 是上游继承的旧表，与本地实测环境不符，不要引用。

---

## Q1 Prefill 阶段如何提升速率？

### 30 秒版

Prefill 有三条加速路径，按收益排序：

1. **算子层**：把原来逐层、逐序列读 `cu_seqlens[i].item()` 做 Python slicing 的
   写法，换成 packed varlen Triton kernel——一次 launch 覆盖整个 packed batch 的
   所有 query token。
2. **消除 GPU scalar 同步**：调度元数据全部在 CPU 侧算好，`aten::_local_scalar_dense`
   从 224 次降到 0 次。
3. **减少实际要算的 token**：prefix cache 命中后只 prefill 增量（实测 300-token
   prompt 第二轮只算 44 token）。

短 prompt 收益最大：8 请求 × 32 token 时 prefill 吞吐 1280.35 → 5382.16 tok/s
（**4.20×**），TTFT 200.02 → 47.65 ms。

### 展开版

#### 1. Packed varlen：一次 launch 覆盖整个批

`nanovllm/layers/flash_attn.py:905` 的 `varlen_prefill_attention_kernel`，
grid = `(total_query_tokens, num_heads)`。每个 program 对应一个
(query token, query head) 对，靠 CPU 侧构造的三个索引数组定位自己：

| 数组 | 含义 |
|---|---|
| `prefill_seq_ids` | 这个 query token 属于哪条序列 |
| `prefill_query_positions` | 它在完整序列中的绝对位置 |
| `prefill_key_starts` | 非 paged 时对应的 K/V 起点 |

这三个数组在 `model_runner.py:196 prepare_prefill` 里，与已有的 CPU 调度循环一起
生成，随 `input_ids` / `positions` 一起 pin memory 后异步拷贝上卡。

#### 2. 干掉 GPU scalar 同步（最容易被忽略、但最实的一刀）

阶段 0 的 profiler 里 `aten::_local_scalar_dense` 出现 **224 次**。根因是原实现
每层、每条序列都要读一次 `cu_seqlens_q[i].item()` / `cu_seqlens_k[i].item()` 来做
Python slicing，每次 `.item()` 都是一次 device→host 同步；28 层 × 8 序列 = 224 次。

现在两条路径都不读 GPU scalar：

- **Triton 路径**：靠上面三个索引数组，kernel 内部用 `tl.load` 取；
- **SDPA fallback 路径**：靠 `cu_seqlens_q_cpu` / `cu_seqlens_k_cpu` 两个 CPU tuple
  做 Python slicing（`flash_attn.py:1111-1138`）。

阶段 3 两组 profile（2×64 Triton、2×256 SDPA fallback）都是 **0 次**。

#### 3. 减少实际要算的 token 数

- **Prefix cache**：`block_manager.hash_blocks` 把算完的完整 block 写进
  `hash_to_block_id`；下一次同 prefix 请求在 `num_cached_blocks()` 命中，只 prefill
  增量。实测 2×300-token prompt 第二轮只处理 44 token。
- **Chunked prefill**：长 prompt 按 token budget 切片，避免一次性占满显存
  （`_fit_resumed_prefill` + `reserve()`）。
- **lm_head 只算最后一个 token**：`ParallelLMHead.forward` 在 prefill 时执行
  `x[context.cu_seqlens_q[1:] - 1]`，所以 lm_head 开销是 O(num_seqs) 而非
  O(total_tokens)。

#### 4. 明确的适用边界（必须主动说）

RTX 3060 Laptop、Qwen3-0.6B、8 请求、16 output，SDPA/Triton 同进程交替、每种 3 轮：

| Input/request | SDPA prefill | Triton prefill | 比值 | SDPA/Triton TTFT |
|---:|---:|---:|---:|---:|
| 32 | 1280.35 tok/s | 5382.16 tok/s | **4.20×** | 200.02 / 47.65 ms |
| 128 | 6045.64 tok/s | 9820.63 tok/s | **1.62×** | 169.46 / 104.35 ms |
| 256 | 10978.75 tok/s | 6952.25 tok/s | **0.63×** | 186.64 / 294.65 ms |

原因：这个 kernel **每个 query token 独立遍历历史 K/V**——没有做 query 维分块
（`BLOCK_M = 1`），也没有用 tensor core（`tl.sum(k * q[None, :], axis=1)` 而不是
`tl.dot`）。prompt 短时收益来自「去掉 Python 循环 + SDPA 分发 + 同步」；prompt 变长
后 SDPA 的 query/key tile 复用优势反超。

自动策略因此是（`attention.py:337-340`）：

```text
max_seqlen_q <= 128  -> Triton varlen
max_seqlen_q > 128   -> PyTorch SDPA with CPU offsets
```

#### 追问预案

**「为什么不把 query 也分块，做成真正的 FlashAttention？」**

这是当前 kernel 最大的改进空间，我明确记录为下一步。加 `BLOCK_M > 1` 的 query
tiling + `tl.dot` 就能把阈值往上推，但要重新处理 causal mask：现在靠循环上界
`context_len = query_position + 1` 免掉了 mask 矩阵，分块后同一 tile 内不同 query 的
`context_len` 不同，必须退回 tile 内掩码。另外本机只有一张 RTX 3060，没有做过对比
验证，所以我没把它写成已完成能力。

**「prefix cache 怎么保证正确？」**

- 哈希：`BlockManager.compute_hash` 用 xxhash64 链式哈希，前一块的 hash 作为
  `prefix` 一起 update，所以哈希本身编码了整条前缀路径。
- 抗碰撞：哈希碰撞用 token_ids 二次校验兜住——
  `num_cached_blocks` 里 `if block_id == -1 or self.blocks[block_id].token_ids != token_ids: break`。
- 一个反直觉点：`num_cached_blocks` 用 `range(seq.num_blocks - 1)`，**故意跳过最后一个
  block**。必须留至少一个 block 重算，否则 `num_tokens` 会是 0，就没有 logits 可采样。
  代价是 prompt 恰好为 block_size 整数倍时，最后一个满块不复用，最多多算
  `block_size - 1` 个 token——这是 block 粒度 prefix cache 的固有代价，不是 bug。
- 引用计数：`Block` 有 `ref_count`，多请求共享同一 prefix block；`_allocate_block` 在
  重新分配带 hash 的块时会从 `hash_to_block_id` 摘除并计入 `cache_evictions`。

**「chunked prefill 会不会白算 token？」**

会。`postprocess` 里 `if is_prefill and seq.num_cached_tokens < seq.num_tokens: continue`
——中间 chunk 的采样结果被丢弃（因为 `ParallelLMHead` 只对每序列最后一个 token 算
logits，而 chunk 中间那一步的"最后一个 token"并不是真正的末 token）。所以 chunked
prefill 每轮会浪费一个 token 的采样，这是为支持流式 chunk 付出的代价。

---

## Q2 Continuous Batching 如何调度？

### 30 秒版

核心是**一次 `schedule()` 返回两个 batch，共享同一份 token budget**：先排 decode
（每个 running 请求 1 token），剩余预算再排 prefill。这样长 prompt 不会阻塞已经在
跑的请求。running 队列用确定性 round-robin 防止饥饿；显存不够时**只抢占本轮尚未
进入执行的 victim**，释放后靠 prefix cache 或 recompute 恢复。

### 展开版

#### 1. 一轮调度返回什么

`Scheduler.schedule()`（`nanovllm/engine/scheduler.py:88`）返回
`[(decode_seqs, False), (prefill_seqs, True)]`，两批**共享**同一个 `budget`
（`max_num_batched_tokens`）和 `seq_slots`（`max_num_seqs`），不是各拿一份。

```text
shared token/sequence budget
    ├─ decode batch: running round-robin, 1 token/request
    └─ prefill batch: FIFO waiting, consume remaining budget
```

#### 2. 为什么不合批（面试官最爱问）

三种 shape 的执行路径完全不同，强行合并得不偿失：

| 维度 | Decode | Prefill |
|---|---|---|
| 形状 | 等长 `[bs, 1, H, D]` | 变长 packed `[total_tokens, H, D]` |
| attention kernel | Paged Attention v1/v2（读 paged cache + block table） | varlen Triton 或 SDPA |
| 执行模式 | CUDA Graph（按 batch size 桶捕获，形状必须固定） | eager |
| lm_head | 每个 token 都要 logits | 只取每序列最后一个 token |

所以统一的是 **admission 和预算**，不是 kernel。

#### 3. Decode 优先 + 抢占的一个关键细节

decode 阶段先把 `self.running` 整体搬进私有的 `decode_candidates` 并清空 `running`
（`scheduler.py:97-99`）。两个作用：

- 同一序列不可能在一轮里被选两次；
- **已经进入本轮 decode batch 的序列绝不会被选为 victim**——victim 只从
  `decode_candidates` 的**尾部**取（本轮还没轮到的候选），避免 ModelRunner 正在用它的
  block table 时把 cache 释放掉。

`preempt()`（`:197`）= 置 `WAITING` + `is_prefill=True` + `deallocate()` +
`waiting.appendleft()`。插到**队首**，所以被抢占的请求比新请求优先恢复。

#### 4. Round-robin 的实现

`：124-126`：`self.running = 未选中的候选 + 已选中的序列`。下一步未选中的自然排在
前面。固定输入下顺序可复现，不会出现"连续轮次只服务同一批 request"。

#### 5. Prefill 的两种形态（容易答错）

分支键是 **`not seq.block_table`**，不是"是不是新请求"：

- `block_table` 为空 → `_fit_new_prefill`：先查 prefix 命中数，再 `can_allocate`
  试探、逐步缩小 chunk 直到可行；`allocate()` 会写
  `num_cached_tokens = num_cached_blocks * block_size`。
- `block_table` 非空 → `_fit_resumed_prefill` + `reserve()`：增量补新跨越的 block。

因为 `deallocate()` 会清空 `block_table`，**被抢占的序列也走 `_fit_new_prefill`**，
重新查 prefix cache——这正是"抢占后能便宜恢复"的原因：被释放的 block 只要没被重新
分配，`hash_to_block_id` 里的条目还在。

#### 6. 增量 KV 分配

首次 prefill 只为 `cached_tokens + scheduled_chunk_tokens` 分配，而不是在 admission
时按完整 prompt 预留。确定性压测（block size 4、6 请求）：一次性预分配需要
**16** blocks，增量策略实测峰值 **9** blocks → 占用 56%，约 **1.78×** 并发。

降级循环的一个性质：每轮只让出**恰好一个 block**
（`num_tokens = (target_blocks - 1 - num_cached_blocks) * block_size`），可行性对
`num_new_blocks` 单调，所以首次成功的轮次给出的就是**最大可行 chunk**。

#### 7. 一个必须主动承认的边界

`if seq.status == SequenceStatus.WAITING: break`（`:182-183`）——**一轮最多只调度一个
未完成的 chunked prefill**。所以 waiting 队首的长 prompt 会阻塞后面的 prefill 请求
（head-of-line blocking）。这是为简化预算与 block 记账做的取舍，不是 bug，但确实是
可改进点。

#### 追问预案

**「抢占之后怎么恢复？」**

两条路：

1. 被释放的 block 还在 `hash_to_block_id` 里 → `num_cached_blocks()` 直接命中，
   只需重算未缓存部分；
2. 完全被覆盖了 → 从头 recompute。

另外 `can_append` 只在 `len(seq) % block_size == 1`（正好跨块）时才需要新 block，
其余 decode 步不需要——所以抢占**只在 decode 序列恰好跨块且空闲池为空时**才触发。

**「`can_allocate` 里的 `free_cached_blocks` 是不是重复计数了？」**

不是。`C = free_cached_blocks` 是"缓存命中、但当前躺在空闲池里"的块。认领它们会把
它们移出空闲池，所以必须预留。判定式 `F >= N + C` 是对的：先认领 C 个（剩余 F−C），
再分配 N 个新块。

**「12 请求的公平性测试怎么做的？」**

这是 CPU 确定性仿真（`bench_phase4_scheduler.py`，block size 4、每请求生成 4 token）：

| Case | Token budget | KV blocks | 请求数 | Rounds | Preemptions | Leak |
|---|---:|---:|---:|---:|---:|---:|
| mixed ample cache | 8 | 24 | 6 | 10 | 0 | 0 |
| mixed cache pressure | 6 | 7 | 6 | 12 | 0 | 0 |
| decode fairness | 4 | 12 | **12** | 12 | **0** | 0 |
| decode preemption pressure | 12 | 4 | 8 | 12 | **4** | 0 |

GPU 侧另有一次真实混合负载（6 请求，prompt 32/384/64/256/96/512，CUDA Graph），
结束 93/93 blocks free、0 preemption、0 leak。**要如实说这些是小样本、以仿真为主。**

---

## Q3 量化和反量化如何进行，发生在哪些地方，会不会有精度损失

### 30 秒版

**量化发生在 KV 写入时**，融合在 Triton store kernel 里完成；**反量化有两个地方**：
decode 在 Paged Attention kernel 里**融合反量化**（不物化浮点 cache），prefill 则是
**物化 gather + 反量化**再走 SDPA——后者正是 prefill 慢 25% 的根因。

粒度是 per-token / per-KV-head 对称量化，scale 存 FP16。精度上 attention 输出与基线
kernel 的 cosine similarity 是 0.999958，但我**没有做 perplexity / 下游质量评估**，
这是明确的边界。

### 展开版

#### 1. 量化：发生在写入时，融合在 store kernel 里

代码：`nanovllm/layers/attention.py:138 store_kvcache_int8_kernel`，
grid = `(num_tokens, num_kv_heads)`，每个 program 处理一个 head vector（head_dim=128）。

```python
key_scale = tl.maximum(tl.max(tl.abs(key), axis=0) / 127.0, 1.0e-8)
quantized_key = key / key_scale
quantized_key = tl.where(quantized_key >= 0,
                         quantized_key + 0.5,
                         quantized_key - 0.5).to(tl.int8)
```

三个值得说的设计选择：

- **`/127` 而不是 `/128`**：最大元素恰好映射到 ±127，`+0.5` 后是 127.5，
  `.to(tl.int8)` 向零截断仍回 127 → **永不溢出，整个 kernel 不需要 clamp**。
- **舍入是 round-half-away-from-zero**，不是 PyTorch `round()` 的 half-to-even。写法是
  `where(q >= 0, q + 0.5, q - 0.5)` 再向零截断——对称，对正负号一致。
- **粒度是 per-(token, KV head)**，而不是 per-tensor 或 per-block：在精度、scale
  开销、decode 索引复杂度之间折中。per-tensor scale 精度差；粗化到整个 block
  （128 个 token 共用一个 scale）对长尾分布不友好。

**存储收益**：head_dim=128 时

```text
BF16 : 128 × 2 bytes              = 256 B
INT8 : 128 × 1 byte + 1 FP16 scale = 130 B
ratio: 130 / 256                   = 50.78%
```

实测可用 block 数 **92 → 179**（**1.95×**）。注意 KV cache 是按剩余显存定容的，
所以 `kv_storage_bytes` 几乎没变（2.701 → 2.669 GB）——涨的是**块数/容量**，不是
省了显存。这个表述要准确。

#### 2. 反量化路径 A：decode 融合在 Paged Attention 里（"好"的那条）

代码：`flash_attn.py:162 paged_attention_int8_kernel`。它是 v1 kernel 的近乎克隆，
多出来的只有两处：

```python
key_scale = tl.load(k_scale_ptr + k_scale_offsets, mask=mask_t, other=0.0)
key = tl.load(k_cache_ptr + k_offsets, mask=mask, other=0.0
             ).to(tl.float32) * key_scale[:, None]
```

即：K/V 从 INT8 cache 读进寄存器 → 转 FP32 → 乘 per-token scale → 直接进 online
softmax。**全程不分配 BF16 cache 临时副本**，省下来的就是带宽。

RTX 3060 microbenchmark（batch 8、16 Q heads / 8 KV heads、head_dim 128、ctx 511）：

| Backend | Latency | Cache bytes | Ratio |
|---|---:|---:|---:|
| FP16 Paged Attention v1 | 88.06 µs | 16,777,216 | 1.000 |
| INT8 fused dequant | 72.29 µs | 8,519,680 | 0.508 |

INT8/FP16 延迟比 0.821，**局部 kernel 提升约 1.22×**。

注意 INT8 永远走 v1 形状的 kernel：`paged_attention` 里
`if k_cache.dtype == torch.int8` 提前 return（`:793-805`），绕过了
`select_paged_attention_backend`，所以 **INT8 拿不到 v2 的 GQA KV 复用和 tensor core**。

#### 3. 反量化路径 B：prefill 是物化的（"代价"的那条）

代码：`flash_attn.py:859 gather_quantized_cache`：

```python
values = cache[block_ids, block_offsets].to(torch.float32)
token_scales = scales[block_ids, block_offsets].float()
return (values * token_scales.unsqueeze(-1)).to(dtype)
```

它按 block table 把当前请求实际引用的 KV **物化成一个 `[seqlen, kv_heads, head_dim]`
的浮点张量**，再交给 SDPA。目的是保持 prefix/chunk 正确、显存上界清晰；代价是每层
都要重新 gather + 反量化 + 写一遍浮点张量。

**这就是 INT8 端到端变慢的根因。** 4×128 input / 16 output、CUDA Graph：

| KV dtype | Blocks | Prefill tok/s | Decode tok/s | E2E |
|---|---:|---:|---:|---:|
| BF16 auto | 92 | 7728.78 | 634.49 | 161.58 ms |
| INT8 | 179 | 5772.94 | 590.82 | 191.10 ms |

prefill **−25.3%**、decode 端到端 −6.9%、整体 **+18.3% 慢**。所以 INT8 在当前实现里
是「**用吞吐换上下文容量**」的可选模式，不设为默认。

#### 4. 精度损失——分三层说，越往下越诚实

**第一层 · 算子层（自动化测试）**

CUDA 测试里 fused INT8 decode 对显式反量化的 PyTorch reference，`atol=rtol=3e-2`
通过；恢复值最大误差不超过约半个 quantization step。

**第二层 · Attention 输出层（microbenchmark）**

mean / max absolute error = 0.000520 / 0.002747，**cosine similarity 0.999958**
（batch 8、ctx 511、合成数据）。

**第三层 · 端到端层（必须主动说）**

64 个 autoregressive greedy token 的 exact agreement 只有 **46.875%**，首 token 是
4/4 一致。原因**不是**"量化错了"，而是量化引入的极小 logits 差异在接近 tie 时会翻转
argmax，然后自回归 rollout 把分歧逐步放大。

**exact token agreement 不是语言质量指标。** 我目前只能声称 attention 数值接近、
执行正确，**不足以宣称下游质量无损**——生产使用前需要在目标数据集上测
perplexity / 任务指标。这个边界我明确写在报告里，没有包装。

#### 追问预案

**「为什么带宽减半，kernel 只快 1.22×？」**

一个可观察的信号：v1 用 Triton 默认 `num_warps=4`，而 INT8 版本显式用 `num_warps=8`
——**线程数翻倍**，因为反量化把 `[256, 128]` 的 K/V tile 提到了 FP32 精度，寄存器
需求顶上去了。所以 INT8 decode 是「用寄存器压力换带宽」，收益被抵消了一部分。
这是从代码得出的分析，ncu 因权限（`ERR_NVGPUCTRPERM`）没跑成，我没把它写成结论。

**可优化方向**（代数等价，减少 ALU 与寄存器压力）：

- K：`scores[t] = k_scale[t] × sum(k_int8[t] · q) × scale`——先算点积再乘
  `[block_size]` 的 scale 向量，而不是先缩放整个 `[block_size, head_dim]` tile
  （乘法次数 32768 → 256）。
- V：把 `v_scale` 折进 `p`（保留未缩放的 `p` 供 normalizer `l` 使用），同样约 128×
  减少乘法。

**「全零向量怎么办？」**

`tl.maximum(..., 1e-8)` 防的是除零。但这里有个我知道的细节：`1e-8` 小于 FP16 的最小
次正规数（约 5.96e-8），所以全零向量的 scale 存进 FP16 会变成 0。数据同为 0，乘积
仍然是 0，结果正确；但测试里 `assertTrue(all(flat_ks > 0))` 这个不变量严格说只对非零
向量成立。

**「scale tensor 和 data tensor 的生命周期怎么管？」**

两者共用相同的 physical block ID，prefix cache 的 key 仍然是 token block 元数据，
所以 scale 不改变哈希、引用计数和回收逻辑。2×300 input / 4 output 的 prefix smoke
里 BF16 与 INT8 都命中 2 blocks / 512 prefix tokens，结束时 179/179 blocks 回收，
无 leak。

---

## 通用追问预案

| 问题 | 答案要点 |
|---|---|
| **你的数字怎么来的？** | 每个阶段都有五件套：① PyTorch / eager reference ② 边界、dtype、prefix/chunk/Graph 单元测试 ③ 同进程 microbenchmark 或端到端 workload ④ 硬件+模型+shape+模式+原始 JSON ⑤ 收益适用区间、失败实验和 fallback。全量 59 tests，24 份 baseline JSON。 |
| **硬件环境？** | RTX 3060 Laptop 6 GiB + Qwen3-0.6B BF16。注意根 README 写的是 RTX 4070，那是上游继承的旧表。 |
| **为什么没做多卡？** | 只有一张卡。TP 代码路径存在（`world_size` 参与 `num_kv_heads` 切分），但没有多卡 NCCL 实机验证，不作为已完成能力。 |
| **最大的不足？** | ① INT8 prefill 未融合反量化 ② 没有 PPL / 下游质量评估 ③ varlen prefill 没有 query 维分块和 tensor core ④ chunked prefill 有 head-of-line blocking ⑤ Split-K 收益跨运行不稳定（0.96～1.34×，方向不定），只保留实验入口，不进 `auto`。 |
| **Split-K 是什么？为什么没进 auto？** | 把长 context 切成 4 段，各段输出局部 max / normalizer / FP32 accumulator，再用第二个 kernel 按 log-sum-exp 合并。数值正确，但 4095-token / batch 1 的多次独立运行观察到约 0.96～1.34×，**方向不稳定**，所以只留 `backend="v2_split"` 显式入口。 |
| **CUDA Graph 和 v2 为什么不兼容？** | Graph 只按 batch size 捕获，无法在 replay 时更换 context-specific kernel。为避免短请求被捕获时的最大 context bucket 拖累，Graph 明确捕获 v1；v2 heuristic 只作用于 eager。若后续做 `(batch, context_bucket)` 图池可以重新评估。 |

---

## 一页速记

| 主题 | 关键数字 | 关键限定 |
|---|---|---|
| Varlen prefill | 8×32 = 4.20×，TTFT 200.02 → 47.65 ms | 8×256 = 0.63×，回退 SDPA；无 query 分块 / 无 tensor core |
| GPU scalar 同步 | 224 → 0 次 `aten::_local_scalar_dense` | 靠 CPU 侧索引数组 + CPU tuple |
| Paged Attention v2 | batch 32 / ctx 4095 最高 1.87× | kernel 局部收益；E2E 比值 0.958，不声明整机加速 |
| 统一调度 | decode-first 共享 budget | 一轮只跑一个未完成 chunked prefill |
| 增量 KV | 16 → 9 blocks（56%，1.78× 并发） | CPU 确定性仿真，block size 4 |
| 公平性 / 抢占 | 12 请求 0 泄漏；8 请求 4 次抢占 0 泄漏 | CPU 仿真；GPU 侧是 6 请求 smoke |
| INT8 KV 容量 | 50.78% bytes/block；92 → 179 blocks（1.95×） | 涨的是块数，不是省显存 |
| INT8 精度 | cosine 0.999958；max abs err 0.002747 | 没有 PPL 评估；64-token agreement 46.875% |
| INT8 代价 | E2E +18.3%（prefill −25.3%） | 可选模式，非默认；prefill 未融合反量化 |

---

## 相关文档

- [stage2_paged_attention_report.md](stage2_paged_attention_report.md)
- [stage3_varlen_prefill_report.md](stage3_varlen_prefill_report.md)
- [stage4_scheduler_cache_report.md](stage4_scheduler_cache_report.md)
- [stage8_int8_kv_report.md](stage8_int8_kv_report.md)
- [project_final_report.md](project_final_report.md)
- [resume_project_zh.md](resume_project_zh.md)
