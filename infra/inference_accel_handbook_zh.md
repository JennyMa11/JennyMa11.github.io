# AI Infra 工程实践指南：从 CUDA Kernel 到 LLM Serving 与模型适配

> **目标**：沿同一条请求链路学习硬件、算子、缓存、调度、分布式、性能与正确性。每章遵循「速查 → 原理 → 执行图 → 伪代码/最小实现 → 工业源码 → 性能 → Debug → 追问」。示例分为**可运行最小实现**与**机制伪代码**；伪代码不宣称可直接生产部署。
>
> **阅读方法**：先运行 CPU/PyTorch reference 并记录 shape、dtype、误差；再写 CUDA/Triton Kernel；接着把算子接到简化推理循环；最后用同一批输入做 Profiling 与逐层比对。没有 NVIDIA GPU 时可以完成 Python 路线，GPU 章节的 CUDA 示例需要相应环境。
>
> **工程约定**：以下源码路径以链接所指版本为准；框架实现变化较快，阅读时先定位入口和调用关系，再查看当前分支。性能结论都依赖硬件、shape、batch 和并发，必须实测。
> **图码编号**：既有配图和代码示例沿用原编号，章节重排后编号不一定等于所在章号；以当前小节标题定位内容。

## 学习路线与交付物

```text
体系结构 → GPU/CUDA → Memory → 并行算法 → GEMM/Softmax
  → Transformer 算子 → Attention/FlashAttention → KV/PagedAttention
  → Prefill/Decode → Scheduler/Serving → 量化/投机 → Runtime
  → 分布式/PD → 框架源码 → Profiling → Debug → Bug → 端到端复盘
  → GPU/NPU 模型适配与验收
```

| 阶段 | 完成后应能交付 |
|---|---|
| 1–3 | 向量 Kernel、Reduction、Softmax、GEMM；附数值和带宽基线 |
| 4–7 | RMSNorm、RoPE、Attention、分页 KV；说明每个张量的布局与生命周期 |
| 8–12 | 简化 Scheduler 和 Request 状态机；画出从 API 到采样的完整调用链 |
| 13–17 | 一份 Profiling 报告、一份逐层数值定位记录、一份故障复盘 |
| 18 | 一份跨设备模型适配记录：环境矩阵、算子清单、数值对齐、性能与稳定性验收 |

---

# 第 0 章 全局地图：从请求到 GPU 与 Token

> **承上**：先看整条请求链路，明确每层为什么存在。

## 0.1 一条请求与三份账本

```text
HTTP 请求 → tokenizer → Request(id, token_ids, sampling_params)
         → Scheduler(可运行 token 数 + 可用 KV 块)
         → ModelRunner(输入打包、position、block table)
         → GPU: embedding → N×Transformer Block → logits
         → sampler → 新 token → KV 追加 / 释放 → 流式响应
```

读任何优化都记三份账：**时间**（TTFT、TPOT、排队/计算/传输）、**字节**（权重、激活、KV、通信）、**状态**（请求阶段、已计算 token、KV 所有权）。例如 PagedAttention 主要管理 KV 的空间和寻址，并不等于 FlashAttention 的 IO tiling；Continuous Batching 主要提高多请求混批效率，并不自动减少单请求计算量。

**实战起点**：固定模型版本、tokenizer、prompt、采样参数、dtype、随机种子和硬件。先测无并发基线，再分别改变输入长度、输出长度、并发量，记录 TTFT、TPOT、吞吐、显存峰值。后面每章都回到这组实验解释变化。

## 0.2 一张图看懂四层


面试里最容易被问的开场题：**「LLM 推理加速你了解哪些方法？」** 不要一上来就报菜名，先给框架。

> **术语先立住**：**LLM（Large Language Model，大语言模型）** 指参数量在十亿级以上的 Transformer 类自回归语言模型；本手册讨论的「推理（Inference）」特指**训练完成后的前向计算与在线服务**，不含训练与微调。

![图 0-1 LLM 推理加速的四层结构](figures/fig_00_four_layers.png)
*图 0-1 · LLM 推理加速的四层结构：越往下越贴近硬件，越往上越贴近业务*

**为什么这样分层？** 因为四层的**优化对象不同**：模型层改「要算什么」，算子层改「单个算子怎么算」，执行层改「怎么把算子交给 GPU」，系统层改「多个请求怎么排队共享资源」。四层互不冲突、可以叠加，但收益与工程代价差异巨大。

## 0.3 四层总览表（可直接背）


| 层次 | 技术 | 核心作用 | 简单例子 |
|---|---|---|---|
| **模型层** | 量化 | 用更低精度表示权重/激活，减少显存和计算量 | FP16→FP8/INT8/INT4；AWQ、GPTQ |
| | GQA / MQA | 减少 K/V Head 数，降低 KV Cache 和 Attention 访存 | Llama-2 70B 用 GQA；MQA 多 Q Head 共用一组 KV |
| | MLA | 把 KV 压缩到低维 latent，进一步压缩 KV Cache | DeepSeek-V2/V3 的 Multi-head Latent Attention |
| | MTP / 投机解码 | 一次预测多个 Token，减少逐 Token Decode 次数 | DeepSeek MTP；Draft + Target Speculative Decoding |
| **算子层** | FlashAttention | 分块计算 Attention，减少 HBM 读写，不物化完整 Attention 矩阵 | FlashAttention-2/3 |
| | PagedAttention | 像 OS 分页一样管理 KV Cache，避免连续显存和碎片 | vLLM PagedAttention |
| | 融合 RMSNorm/RoPE/SiLU | 多个小算子合并，减少 Kernel Launch 和显存读写 | RMSNorm+Residual、SiLU+Mul、RoPE Fusion |
| | Triton / CUDA Kernel | 手写高性能 GPU 算子，提高访存/并行/Tensor Core 利用率 | Triton Attention、CUDA SGEMM |
| | FP8 | 8-bit 浮点计算，提高 Tensor Core 吞吐、降低带宽 | H100 FP8 GEMM、Transformer Engine |
| **执行层** | CUDA Graph | 把一系列 Kernel 预先捕获后整体执行，降低 CPU Launch 开销 | vLLM Decode CUDA Graph |
| | torch.compile | 编译 PyTorch Graph，做算子融合、图优化、Kernel 生成 | TorchInductor + Triton |
| | 异步 Pipeline | CPU 调度 / GPU 计算 / 数据传输并行，隐藏等待 | CPU 准备下一 Batch 时 GPU 正在 Decode |
| **系统层** | Continuous Batching | 请求动态加入/退出 Batch，提高 GPU 利用率 | vLLM、SGLang |
| | Chunked Prefill | 长 Prompt 不一次 Prefill，而是分块执行 | 8192 Token Prompt 拆成 512/1024 Token Chunk |
| | Prefix Cache | 相同 Prompt 前缀直接复用已有 KV Cache | System Prompt、多人共享长 Context |
| | KV Cache 管理 | 动态分配/释放/复用/换出 KV Cache | Block Manager、KV Cache Offload |
| | TP / PP / EP | 把大模型拆到多 GPU 上 | Tensor / Pipeline / Expert Parallel |
| | PD 分离 | Prefill 和 Decode 放到不同 GPU/机器，分别优化 | Prefill GPU 高算力、Decode GPU 高带宽 |
| | 高效 Serving | 调度+缓存+流式+负载均衡形成完整在线服务 | vLLM、SGLang、TensorRT-LLM |

> 上表中出现的缩写，正文对应章节都会给出全称与解释；此处先建立整体印象即可。

## 0.4 加速的本质：三个瓶颈


所有推理加速技术，最终都在解决下面三个瓶颈之一。**面试时先把问题定位到瓶颈，再谈技术，会显得非常有条理。**

![图 0-2 三个瓶颈：算力 / 带宽 / 容量](figures/fig_00_bottlenecks.png)
*图 0-2 · 三个瓶颈与各自的典型对策*

| 瓶颈 | 表现 | 典型对策 |
|---|---|---|
| **算力（FLOPs）** | 大 GEMM、长序列 Attention，SM 打满 | 低精度计算（FP8/INT8）、稀疏化、投机解码 |
| **显存带宽（HBM BW）** | Decode 阶段每 token 都要读全部权重 + KV | 量化、GQA/MLA、Kernel 融合、FlashAttention |
| **显存容量** | 并发一高 KV Cache 就爆，OOM 或降并发 | PagedAttention、KV 量化、Prefix Cache、Offload、PD 分离 |

**术语解释**：

- **FLOPs（Floating-point Operations，浮点运算次数）**：衡量计算量的单位，注意与 **FLOP/s**（每秒浮点运算次数，衡量算力）区分。
- **GEMM（General Matrix Multiply，通用矩阵乘法）**：`C = A × B` 形式的稠密矩阵乘，是 Transformer 里 Linear 层的主体，也是 GPU 上最擅长、最成熟的算子形态。
- **SM（Streaming Multiprocessor，流多处理器）**：GPU 的基本计算单元，一块 GPU 由几十到上百个 SM 组成；「SM 打满」指所有计算单元都在忙。
- **OOM（Out Of Memory，显存不足）**：显存分配失败导致进程报错退出。
- **Offload（卸载）**：把暂时不用的数据从显存搬到 CPU 内存或 NVMe，需要时再搬回。

> **一句话记忆**：Prefill 卡在**算力**，Decode 卡在**带宽**，高并发卡在**容量**。

## 0.5 核心指标词典


面试官经常会用这些缩写，听不懂会直接扣分。

| 指标 | 全称 | 含义 | 关注方 |
|---|---|---|---|
| **TTFT** | Time To First Token（首 Token 延迟） | 从发出请求到收到第一个 Token 的耗时，主要由 Prefill 决定 | 用户体感（首字延迟） |
| **TPOT** | Time Per Output Token（每输出 Token 耗时） | 平均每个输出 Token 的耗时，主要由 Decode 决定 | 流式体验流畅度 |
| **TBT** | Time Between Tokens（Token 间隔） | 相邻 Token 之间的时间间隔及其抖动，Chunked Prefill 主要优化它 | 流式体验稳定性 |
| **Latency** | End-to-End Latency（端到端延迟） | TTFT + TPOT × 输出长度 | 单请求 |
| **Throughput** | 吞吐量 | 单位时间处理的 Token 数 / 请求数 | 服务成本 |
| **Goodput** | 有效吞吐量 | 只统计满足 SLO 的请求吞吐，比 Throughput 更真实 | 线上服务 |
| **MFU** | Model FLOPs Utilization（模型算力利用率） | 实际算力 / 峰值算力，衡量计算效率 | 训练 / 长 Prefill |
| **MBU** | Memory Bandwidth Utilization（显存带宽利用率） | 实际带宽 / 峰值带宽，衡量访存效率 | Decode |

**术语解释**：

- **Token（词元）**：模型处理的最小文本单位。文本先经 **Tokenizer（分词器）** 切成 Token，映射为整数 ID 后送入模型。
- **SLO（Service Level Objective，服务等级目标）**：对服务质量的可量化承诺，如「P99 TTFT < 200 ms」。

**关键认知**：**延迟与吞吐天然矛盾**。Batch 越大吞吐越高，但单请求 TPOT 变差。所以线上服务做的是「在满足 SLO 的前提下最大化吞吐」，这也是 PD 分离、调度策略存在的根本原因。

## 0.6 最重要的一个区分：Prefill 计算密集 vs Decode 访存密集


这是整本手册里**最值钱的一条认知**，几乎所有系统层设计都由它推导出来。

![图 0-3 Prefill vs Decode：两种完全不同的负载](figures/fig_01_prefill_vs_decode.png)
*图 0-3 · Prefill 与 Decode 在计算形态、瓶颈、优化方向上的全面对比*

| 维度 | Prefill | Decode |
|---|---|---|
| 一次处理多少 Token | 整个 Prompt（几十 ~ 几万） | 1 个（每请求） |
| 计算形态 | 大 GEMM，矩阵乘矩阵 | 小 GEMM / GEMV，矩阵乘向量 |
| 算术强度 | 高（~O(N)） | 低（~O(1)） |
| 瓶颈 | **计算（Compute-bound）** | **显存带宽（Memory-bound）** |
| MFU | 高（可到 50%+） | 低（常 < 5%） |
| 优化方向 | 更好的 Kernel、FP8、Tensor Core | 量化、KV 压缩、Batch 增大、CUDA Graph |
| 延迟指标 | TTFT | TPOT / TBT |

**术语解释**：

- **Prefill（预填充阶段）**：把用户输入的全部 Prompt Token 一次性并行前向，建立 KV Cache 的阶段。
- **Decode（解码阶段）**：自回归逐 Token 生成的阶段，每步只处理一个新 Token 并复用历史 KV Cache。
- **GEMV（General Matrix-Vector multiply，通用矩阵向量乘）**：`y = A × x`，其中 `x` 是向量，是 Decode 阶段的主要算子形态，访存受限。
- **算术强度（Arithmetic Intensity）**：每读写 1 字节数据所对应的浮点运算次数（FLOPs / Bytes），是判断「计算受限」还是「访存受限」的核心指标。
- **Compute-bound / Memory-bound（计算受限 / 访存受限）**：前者受限于算力峰值，加带宽没用；后者受限于带宽，加算力没用。
- **Tensor Core（张量核心）**：GPU 内专做矩阵乘累加的硬件单元，支持 FP16/BF16/TF32/FP8/INT8 混合精度，吞吐远高于普通 CUDA Core。

> **追问：为什么 Decode 是访存密集？**
> 生成一个 Token 需要把**全部模型权重**从 HBM 读一遍（MoE 只读激活专家），而计算量只有 2×参数量 FLOPs。以 70B 模型 FP16 为例：读 140 GB 权重，算 140 GFLOPs，算术强度 ≈ 1 FLOP/Byte，远低于 GPU 的机器平衡点（H100 约 300+）。所以**带宽是天花板，加算力没用**。
>
> **推论**：Decode 阶段增大 Batch 几乎不增加权重读取量（权重被复用），却能线性提升吞吐 → 这就是 Continuous Batching 的理论基础。

## 0.7 收益与代价的权衡原则


面试官喜欢问「XX 技术的代价是什么」。记住这张表，回答时永远先说收益、再说代价、最后说适用边界。

| 技术 | 收益 | 代价 / 边界 |
|---|---|---|
| 量化 | 省显存、省带宽 | 精度损失、需校准、算子支持有限 |
| CUDA Graph | 省 CPU Launch 开销 | 静态形状/地址、额外显存、需预捕获多套 |
| 投机解码 | 减少串行 Decode 次数 | Draft 成本、接受率不确定、显存换收益 |
| Chunked Prefill | 降低 TBT 抖动、防 OOM | 有 Token 被浪费、调度更复杂 |
| PD 分离 | 各自最优、互不干扰 | KV 传输开销、需要额外网络/机器 |
| MoE | 参数大而激活少 | 通信/负载均衡/工程复杂度陡增 |

---


> **接下来用在哪里**：下一章拆开 GPU：理解算子在哪些执行单元和存储层运行。

---

# 第 1 章 GPU 架构与 CUDA 执行模型

> **承上**：上一章确定了推理的算力、带宽和容量问题；本章落实到 GPU 的执行资源。

## 1.1 从 CPU 内存层次过渡到 GPU

CPU 用较少的强核、深缓存与复杂控制逻辑优化单线程延迟；GPU 用大量线程隐藏访存延迟，追求总吞吐。一次 Kernel launch 给出 **Grid → Block → Warp(通常 32 个 lane) → Thread**。Block 被调度到一个 SM；同一 Block 的线程能用 Shared Memory 和 Block 内屏障协作。不同 Block 没有普通 Kernel 内的全局屏障，跨 Block 归约通常要第二个 Kernel 或原子操作。

```text
Grid [block 0][block 1] ...
       └─ SM A    └─ SM B
Block → Warp 0: lane 0..31 同步发射指令，分支可导致活动 lane 不同
      → Warp 1: lane 0..31
Thread → 私有寄存器；Block → 共享内存；Device → HBM/global memory
```

```cuda
__global__ void saxpy(const float* x, float* y, float a, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = a * x[i] + y[i];
}
// grid=(n + 255)/256, block=256；边界检查保护末 Block。
```

**实现检查**：一维连续数组让相邻 lane 访问相邻地址；二维矩阵先写出 `offset = row * stride_row + col * stride_col`。不要把 `tensor.shape` 当布局，非连续 Tensor 的 stride 会改变地址。Warp 内条件分支不一致时路径会串行执行；但边界 mask 只发生在末尾少数 lane，通常不是主要瓶颈。

**资源账本**：活跃 Block 上限受每 SM 的寄存器、Shared Memory、线程和 Warp 数共同约束。Occupancy = 活跃 Warp / 该架构可驻留 Warp，低占用可能无法隐藏延迟，但高占用也不保证快；寄存器变少导致 spill 到 local memory 时可能更慢。先测实际瓶颈，再调 `blockDim`、每线程元素数和 tile。

**调试**：`compute-sanitizer --tool memcheck ./a.out` 查越界；`--tool racecheck` 查部分 Shared Memory 数据竞争；在每次 Kernel launch 后检查 CUDA 错误，调试时 `cudaDeviceSynchronize()` 让异步错误尽早暴露。面试追问：为什么 Block 间同步不能用 `__syncthreads()`？它只覆盖同一 Block。

## 1.2 GPU 基础补课：Tensor Core / GEMM / 显存层次


面试推理岗，**必须**能说清 GPU 的存储层次和 GEMM 优化思路。

![图 3-3 GPU 存储层次](figures/fig_03_gpu_memory.png)
*图 3-3 · GPU 存储层次：越往上越快越小，越往下越慢越大*

**术语解释**：

- **Register（寄存器）**：GPU 最快的存储，线程私有，容量极小（KB 级）。
- **Shared Memory（共享内存）**：SM 内所有线程共享的片上存储，需手动管理，**Bank Conflict** 就发生在这里。
- **L2 Cache（二级缓存）**：跨 SM 共享的片上缓存，容量 MB 级。
- **HBM（High Bandwidth Memory，高带宽显存）**：GPU 主显存，GB 级，带宽是大多数推理负载的瓶颈。
- **内存墙（Memory Wall）**：算力增速长期快于带宽增速，导致系统越来越受带宽限制的现象。
- **双缓冲（Double Buffering）**：在处理当前数据块的同时预取下一块，用计算掩盖访存延迟。

```text
寄存器 (Registers)    ~KB     最快
    ↓
共享内存 / SRAM        ~100KB/ SM   需手动管理，Bank Conflict 就发生在这里
    ↓
L2 Cache              ~MB     全局共享
    ↓
HBM（显存）            ~GB     最慢，带宽是瓶颈
```

**GEMM 优化三件套**：**Tiling（分块提高数据复用）**、**双缓冲（Prefetch 隐藏延迟）**、**Tensor Core（矩阵乘专用单元）**。用不用 Tensor Core 通常有数倍差距。

## 1.3 Shared Memory Bank Conflict


**速查**：一个 **Warp（线程束，GPU 调度的最小线程组，通常 32 个线程）** 内多个线程同时访问 Shared Memory，如果地址落在**同一个 Bank 的不同地址**，访问会被串行化，降低访存效率。

![图 3-4 Shared Memory Bank Conflict](figures/fig_03_bank_conflict.png)
*图 3-4 · 地址分散在不同 Bank 时无冲突；多个地址撞同一 Bank 则被串行化*

**术语解释**：

- **Bank（存储体）**：Shared Memory 被划分为 32 个 Bank（通常每个 4 字节宽），可并行访问。
- **Bank Conflict（存储体冲突）**：同一 Warp 内多个线程访问同一 Bank 的不同地址，硬件无法并行，只能串行化。
- **Broadcast（广播）**：若多个线程访问同一 Bank 的**同一地址**，硬件可一次广播，不算冲突。
- **Padding（填充）**：在数组每行末尾补几个元素，改变地址步长以避开冲突。
- **Swizzle（混淆/重排）**：通过地址位重排打散访问模式，是比 Padding 更现代的解法。

| 场景 | 冲突原因 | 解法 |
|---|---|---|
| 列访问 `[row][col]` 行主序 | 步长等于 Bank 数 | **Padding**：数组宽度 +1 |
| 转置 | 读列写行 | Padding 或 Swizzle |
| 大 stride 访问 | 多个线程撞同一 Bank | 改变数据布局 |

**码 3-5 · Bank Conflict 的地址算术（可实跑验证）**

```python
# 行主序 [row][col] 的地址 = row * width + col；Shared Memory 有 32 个 Bank
for width in (32, 33):
    # 一个 warp 的 32 个线程同时访问第 3 列（row = 0..31）
    banks = [(row * width + 3) % 32 for row in range(32)]
    print(width, "冲突路数 =", max(banks.count(b) for b in banks))

# 输出：
# 32 冲突路数 = 32    ← 步长 32，32 个线程全撞同一个 Bank，硬件串行化 32 次
# 33 冲突路数 = 1     ← padding 1 后步长 33，33 mod 32 = 1，依次错开，无冲突
```

> 输出是实跑结果。这就是「Padding +1」的全部原理：**让访问步长 `stride mod 32 ≠ 0`**。注意冲突是「同一 Bank 的**不同地址**」——若 32 个线程访问的是同一 Bank 的**同一地址**，硬件会广播，不算冲突。

**追问：为什么 Padding 能解决冲突？**
> 假设 32×32 的矩阵按行主序存，列访问时同一列的元素地址间隔 32，恰好都落在同一 Bank。把每行宽度改成 33（padding 1），列元素地址间隔变成 33，模 32 后依次错开，就分散到不同 Bank 了。

---


> **接下来用在哪里**：下一章用 CUDA 把线程、索引和访存写成真实 Kernel。

---

# 第 2 章 CUDA 编程、存储与并行算法

> **承上**：理解 Warp/SM 后，才能判断代码中的索引、同步与访存是否正确。

## 2.1 存储与访问：先画地址再写优化

| 层 | 典型所有权 | 优化动作 | 常见误判 |
|---|---|---|---|
| Register | Thread | 复用标量、减少重复计算 | 过多寄存器造成 spill |
| Shared Memory | Block | tile 复用、协作装载 | 同步遗漏、Bank Conflict |
| L1/L2 | SM/设备缓存 | 改善局部性 | 命中率高不代表绝对带宽足够 |
| HBM / Global | 设备 | 合并访问、减少字节 | 连续逻辑索引不代表物理连续 |

**Coalescing**：32 lane 读连续 `float` 通常比跨 stride 的 32 次散读更容易合并为少量内存事务。**Bank Conflict**：共享内存同一 Warp 不同地址落同一 bank 会冲突；广播同地址是例外。先根据架构的 bank 组织和元素字节数算映射，再用 Nsight Compute 看 shared load/store 指标。Padding `tile[32][33]` 是列访问的经典例子，不是所有矩阵都应盲目加一列。

## 2.2 Reduction、Scan 与 Softmax 的共同骨架

```text
每线程处理若干元素 → Block 内树形合并 → 跨 Block 二次归约
Scan 多一个“前缀传播”阶段；Softmax 用 max 与 sum 两次归约
```

```python
import torch

def stable_softmax(x, dim=-1):
    m = x.max(dim=dim, keepdim=True).values
    e = torch.exp((x - m).float())
    return e / e.sum(dim=dim, keepdim=True)

def inclusive_scan(x):
    return torch.cumsum(x, dim=-1)
```

CUDA Block 归约伪代码（每步同步；最后一个 Warp 可用 shuffle）：

```cuda
shared[threadIdx.x] = local_sum;
__syncthreads();
for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset) shared[threadIdx.x] += shared[threadIdx.x + offset];
    __syncthreads();
}
if (threadIdx.x == 0) partial[blockIdx.x] = shared[0];
```

**优化路径**：先测试非 2 的幂长度、空行、极大值和不同 dtype；再减少跨 Block 原子争用、把多元素归约放到寄存器，检查寄存器压力与同步次数。Scan 不能把各 Block 独立扫描后直接拼接，必须把前序 Block 总和加到后序输出。

**Bug 案例**：只在部分线程进入 `__syncthreads()` 可能死锁；Softmax 忘记减 `max` 会溢出为 NaN；未 mask 的 padding 会改变分母。追问：在线 Softmax 需要保存哪两个统计量？历史最大值 `m` 与指数和 `l`。

## 2.3 Triton / CUDA Kernel


**速查**：当框架自带算子不够快时，用 Triton（Python DSL，上手快）或 CUDA C++（极致控制）手写算子，针对性优化访存模式、并行划分和 Tensor Core 使用。

**术语解释**：

- **Kernel（核函数）**：在 GPU 上并行执行的函数，是 GPU 编程的基本单位。
- **Triton**：OpenAI 开源的 GPU 编程语言，用 Python DSL 编写，编译器自动管理共享内存与寄存器，是 torch.compile 的默认后端。
- **DSL（Domain-Specific Language，领域特定语言）**：为特定领域设计的简化语言。

| 维度 | Triton | CUDA C++ |
|---|---|---|
| 语言 | Python DSL | C++ |
| 上手难度 | 低，自动管理 shared memory / 寄存器 | 高，手动管理 |
| 可移植性 | 跨 NVIDIA/AMD | 主要 NVIDIA |
| 控制力 | 中（block 级抽象） | 高（线程级） |
| 典型用途 | Attention Kernel、融合算子 | 极致优化、特殊硬件特性 |

**码 3-4 · 最小 Triton softmax kernel**

```python
@triton.jit
def softmax_kernel(out_ptr, in_ptr, n_cols, BLOCK: tl.constexpr):
    row = tl.program_id(0)                              # 一个 program 负责一行
    offs = tl.arange(0, BLOCK)
    x = tl.load(in_ptr + row * n_cols + offs,
                mask=offs < n_cols, other=-float("inf"))   # other 用 -inf，不污染 max
    x = x - tl.max(x, axis=0)                           # 数值稳定：先减 max
    num = tl.exp(x)
    tl.store(out_ptr + row * n_cols + offs,
             num / tl.sum(num, axis=0), mask=offs < n_cols)
```

> 对比一段等价的 CUDA C++，差异一眼可见：这里**没有线程索引 `threadIdx`、没有 `__shared__` 声明、没有 `__syncthreads()`、没有手写归约树**。`tl.arange` 返回的是 block 级的向量，共享内存分配、线程划分、跨线程归约与同步全部由编译器决定——这就是 Triton「block 级抽象」的含义。上手快是它最大的优点，代价是你拿不到线程级控制（想手写 swizzle 或 warp 特化时就得回到 CUDA）。

**追问：Triton 写 Attention 要注意什么？**
> ① **Block 划分**：`BLOCK_M`（Query 维）、`BLOCK_N`（Key 维）的取舍；② **是否用 `tl.dot`**——不用 Tensor Core 会慢很多；③ **Causal Mask**：可以用循环上界省掉 mask 矩阵，但分块后必须退回 tile 内掩码；④ **num_warps 调参**——寄存器压力大的 Kernel 需要更多 warp。


> **接下来用在哪里**：下一章把这些规则用在 GEMM、Softmax 与 Tensor Core 上。

---

# 第 3 章 GEMM、Tensor Core 与 Triton 算子

> **承上**：有了 CUDA 与并行算法，才有能力讨论 Transformer 里最大的矩阵算子。

## 3.1 GEMV → GEMM：Batch 如何改变瓶颈

Decode 的小 Batch 线性层接近 GEMV，权重从 HBM 读取后复用少，常受带宽限制；Prefill 的大量 token 组成 GEMM，权重在 tile 内复用，容易转向计算限制。`C[M,N]=A[M,K]@B[K,N]` 约有 `2MNK` 次浮点操作（把一次乘加计 2 FLOPs）。实际端到端性能还包含转置、量化解包、launch 与通信。

```text
A 的 M×BK tile ─┐
               ├─ Shared Memory → 寄存器片段 → MMA/Tensor Core → C 的 BM×BN tile
B 的 BK×N tile ─┘
               K 维循环，累加用 FP32 或目标实现规定的精度
```

```python
def tiled_gemm_reference(a, b, tile=32):
    m, k = a.shape
    kb, n = b.shape
    assert k == kb
    c = a.new_zeros((m, n))
    for i in range(0, m, tile):
        for j in range(0, n, tile):
            for p in range(0, k, tile):
                c[i:i+tile, j:j+tile] += a[i:i+tile, p:p+tile] @ b[p:p+tile, j:j+tile]
    return c
```

这个 Python 版本用于解释 tile 和尾块，**不是性能实现**。真正 CUDA/Triton Kernel 需要协作装载、边界 mask、寄存器 accumulator、合适布局与 Tensor Core 指令。先用 `torch.matmul`/cuBLAS 作正确性与性能基线。Tensor Core 是否启用取决于架构、dtype、shape/alignment、编译器和指令路径；不要只凭 GPU 型号推断。

**为什么自写 GEMM 比 cuBLAS 慢？** 固定相同 shape、dtype、stride、预热与计时方法，逐项查 CTA/Warp tiling、寄存器分块、向量化加载、shared 布局、MMA 指令、流水线深度、寄存器压力与尾块比例。一个配置通常无法兼顾大方阵、细长矩阵和小 Batch；可为常见 Shape 做专用 Kernel 与 dispatch，并保留通用尾块路径。以端到端实际 Shape 分布选择优化目标，不把某个大矩阵的峰值结果当作所有请求的收益。

**`cp.async` 与双缓冲**：在支持它的 NVIDIA 架构上，异步拷贝可把 Global Memory 的 tile 搬到 Shared Memory，让搬运下一块与当前块的计算重叠；关键收益是隐藏等待，而不是保证单次搬运本身更快。概念流水线为 `load(tile 1) ∥ compute(tile 0) → load(tile 2) ∥ compute(tile 1)`。实现要确保提交、等待与缓冲复用顺序正确；tile 太大可能抬高 Shared Memory/寄存器占用并降低驻留并行度。测吞吐、eligible warps 与资源用量，不能只追求高 Occupancy。

## 3.2 最小 Triton 算子：行 Softmax

```python
import triton
import triton.language as tl

@triton.jit
def row_softmax(X, Y, N: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    col = tl.arange(0, B)
    x = tl.load(X + row * N + col, col < N, other=-float("inf"))
    x = x - tl.max(x, 0)
    ex = tl.exp(x)
    y = ex / tl.sum(ex, 0)
    tl.store(Y + row * N + col, y, col < N)

# B = triton.next_power_of_2(n); row_softmax[(rows,)](x, y, n, B)
```

假设输入是连续二维张量且每行能放进目标 Kernel 的资源预算。调优时改变 `num_warps`、行宽、每 CTA 行数，检查寄存器和 occupancy。PyTorch CUDA Extension 的最小路径是 `torch.utils.cpp_extension.load` 编译 C++ binding + `.cu` Kernel；binding 用 `TORCH_CHECK(x.is_cuda() && x.is_contiguous())` 校验输入，再以当前 CUDA stream 发射 Kernel，返回 `torch::Tensor`。这能把自写 Kernel 接入 PyTorch，但要单独处理 dtype、stride、Autograd 和多 stream 生命周期。

**工业源码入口**：[Triton 教程](https://triton-lang.org/main/getting-started/tutorials/01-vector-add.html)、[CUTLASS GEMM 组织](https://docs.nvidia.com/cutlass/media/docs/cpp/gemm_api.html)、[PyTorch C++/CUDA Extension](https://docs.pytorch.org/tutorials/advanced/cpp_custom_ops.html)。阅读顺序：先看 tensor layout，再看 CTA tile/warp tile，最后看流水线和 epilogue。追问：为什么盲目增大 tile 会变慢？Shared Memory 与寄存器占用可能减少并发，还会增加尾块浪费。


> **接下来用在哪里**：下一章把基础算子拼成 Transformer Block。

---

# 第 4 章 Transformer 基础算子与位置编码

> **承上**：GEMM 等基础算子输出了 Q/K/V 和隐藏状态；本章补齐归一化与位置编码。

## 4.1 从矩阵算子拼成 Decoder Block

```text
x → RMSNorm → QKV Linear → RoPE(Q,K) → Attention(KV 写入/读取)
  → Output Linear → Residual → RMSNorm → Gate/Up Linear
  → SiLU(gate) * up → Down Linear → Residual
```

```python
import torch

def rmsnorm(x, weight, eps=1e-6):
    xf = x.float()
    inv = torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + eps)
    return (xf * inv).to(x.dtype) * weight

def apply_rope(x, cos, sin):
    # x: [..., head_dim]；约定相邻两维组成一对
    even, odd = x[..., 0::2], x[..., 1::2]
    y = torch.empty_like(x)
    y[..., 0::2] = even * cos - odd * sin
    y[..., 1::2] = even * sin + odd * cos
    return y
```

RoPE 必须明确布局约定：有的模型将前半维和后半维配对，不能把相邻维实现直接套用。`cos/sin` 的位置索引来自请求的**真实 position_id**，Prefix Cache、Chunked Prefill、抢占恢复、padding/packed batch 都可能改变“当前 token 在批中的列号”，但不能改变其在序列中的位置。**MRoPE** 对多模态位置使用多个坐标轴（如时间/高度/宽度）；引擎必须把每轴位置元数据传到模型，不能用一个递增标量代替。模型配置决定具体轴布局。

**优化**：RMSNorm 的平方和用 FP32 累加，输出转回模型 dtype；融合 residual/RMSNorm 减少一次中间张量 HBM 往返。若 fused Kernel 与 PyTorch 不一致，先分别比较归一化前的 residual、平方均值、rsqrt、最终乘 weight。追问：为什么 RoPE 应加在 Q/K 而不是缓存后的 K 再重复加？历史 K 只需按原位置旋转一次。

## 4.2 算子融合（RMSNorm / RoPE / SiLU）


**速查**：把多个连续的小算子合并成一个 Kernel，减少 Kernel Launch 次数和中间张量的 HBM 读写。

**术语解释**：

- **算子融合（Operator Fusion）**：把多个逐元素/归约算子合并进同一个 Kernel。
- **RMSNorm（Root Mean Square Layer Normalization，均方根层归一化）**：LLM 常用的归一化层，比 LayerNorm 更省计算。
- **SiLU（Sigmoid Linear Unit，也称 Swish 激活函数）**：`x · sigmoid(x)`，常用于 SwiGLU 门控结构。
- **SwiGLU**：用 SiLU 做门控的前馈网络结构，形式为 `SiLU(xW₁) ⊙ (xW₃)`。

| 融合模式 | 融合前 | 融合后收益 |
|---|---|---|
| **RMSNorm + Residual** | `add` 写 HBM → `rmsnorm` 读 HBM | 省一次读写 |
| **SiLU + Mul**（SwiGLU） | `silu` 写 HBM → `mul` 读 HBM | 省一次读写 |
| **RoPE** | 多次 reshape/mul/add | 合并为单 Kernel |
| **QKV Projection** | 三个 Linear | 合并为一个 GEMM |
| **整个 MLP** | Linear → SiLU → Mul → Linear | 减少中间激活 |

**码 3-3 · 融合前 vs 融合后**

```python
# 融合前：3 次 Kernel Launch，中间结果 3 次往返 HBM
h = x + residual                                              # add     → 写 HBM
h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)     # rmsnorm → 读 + 写 HBM
y = h * weight                                                # mul     → 读 + 写 HBM

# 融合后：1 次 Kernel Launch，h 只存在于寄存器 / SRAM
y = fused_add_rmsnorm(x, residual, weight, eps)
```

> `fused_add_rmsnorm` 是伪函数名，代表把三步塞进同一个 kernel。收益可以算得很清楚：**省 2 次 Launch 开销 + 省掉 `h` 的 3 次写、3 次读**。中间张量越大、算子越碎，这个收益越高——这正是 torch.compile 在 Decode 阶段（大量小算子、访存受限）特别有效的原因。

**追问：为什么融合能加速？**
> 因为 GPU 上「启动一个 Kernel」有固定开销，且每个 Kernel 都要把输入从 HBM 读到寄存器、把输出写回 HBM。融合后中间结果留在寄存器/SRAM 里，**既省 Launch 开销又省带宽**——对访存受限的 Decode 尤其有效。


> **接下来用在哪里**：下一章把 Q/K/V 送入 Attention，并追踪其 HBM 流量。

---

# 第 5 章 Attention 与 FlashAttention

> **承上**：已经知道 Q/K/V 从何而来；现在需要理解长序列的 Attention 如何计算。

## 5.1 从朴素 Attention 到分块执行

令 `Q:[B,Hq,Sq,D]`、`K/V:[B,Hkv,Sk,D]`，GQA 要求 `Hq % Hkv == 0`。数学参考实现先构造 `Sq×Sk` scores；这是正确性 oracle，却会把大矩阵写入显存。FlashAttention 通过 Q/K/V tile 与 online softmax 在片上更新状态。

```text
Q tile → 加载 K/V tile → S=QKᵀ×scale + mask
       → 行最大值 m_new → p=exp(S-m_new)
       → l_new=exp(m_old-m_new)×l_old+sum(p)
       → acc_new=exp(m_old-m_new)×acc_old+pV
       → 下一 tile；最终 O=acc/l
```

```python
import torch

def online_attention(q, k, v, block=64):
    # q:[Q,D], k/v:[K,D]；教学版，先用 float32 验证
    scale = q.shape[-1] ** -0.5
    out = []
    for qs in range(0, q.shape[0], block):
        qb = q[qs:qs + block].float()
        m = torch.full((qb.shape[0], 1), -float('inf'), device=q.device)
        l = torch.zeros_like(m)
        acc = torch.zeros((qb.shape[0], v.shape[-1]), device=q.device)
        for ks in range(0, k.shape[0], block):
            score = qb @ k[ks:ks + block].float().T * scale
            # causal: 全局 q 位置与 k 位置比较；空 tile 不参加更新
            qi = torch.arange(qs, qs + qb.shape[0], device=q.device)[:, None]
            ki = torch.arange(ks, ks + score.shape[1], device=q.device)[None, :]
            score = score.masked_fill(ki > qi, -float('inf'))
            new_m = torch.maximum(m, score.max(-1, keepdim=True).values)
            safe_m = torch.where(torch.isfinite(new_m), new_m, torch.zeros_like(new_m))
            alpha = torch.where(torch.isfinite(m), torch.exp(m - safe_m), 0)
            p = torch.exp(score - safe_m)
            acc = acc * alpha + p @ v[ks:ks + block].float()
            l = l * alpha + p.sum(-1, keepdim=True)
            m = new_m
        out.append(acc / l.clamp_min(1e-20))
    return torch.cat(out, dim=0)
```

上面针对整段 causal self-attention；Decode 时 `Q` 只有当前 token，但 `qi` 必须取其**绝对位置**，不能沿用局部行号。变长批还需 `cu_seqlens`、每请求长度和 mask；全 mask 行要定义输出，不可让 `-inf - -inf` 产生 NaN。工业实现可从 [FlashAttention 的 API 与 CUDA/Triton 实现](https://github.com/Dao-AILab/flash-attention) 先找入口和布局，再跟踪 tile 循环及 online softmax 状态。

## 5.2 GQA / MQA


**速查**：GQA 让多个 Query Head 共享一组 K/V Head（`num_kv_heads < num_q_heads`），MQA 是极端情况（只有 1 组 K/V）。直接减少 KV Cache 大小和 Attention 的 K/V 访存量；质量影响取决于模型训练与 Head 配置。

![图 2-2 MHA / GQA / MQA / MLA 演进](figures/fig_02_attention_variants.png)
*图 2-2 · 四种注意力变体的 Q Head / KV Head 配比与 KV Cache 压缩比*

**术语解释**：

- **MHA（Multi-Head Attention，多头注意力）**：标准注意力，Q/K/V 的 Head 数相同。
- **MQA（Multi-Query Attention，多查询注意力）**：所有 Query Head 共享唯一一组 K/V。
- **GQA（Grouped-Query Attention，分组查询注意力）**：Query Head 分组，每组共享一组 K/V，是 MHA 与 MQA 的折中。

```text
MHA (标准)      Q: n 组   K/V: n 组        KV Cache: 1.0×
GQA (分组)      Q: n 组   K/V: g 组 (g<n)  KV Cache: g/n×
MQA (极端)      Q: n 组   K/V: 1 组        KV Cache: 1/n×
```

**码 2-2 · GQA 的 K/V 广播**

```python
def repeat_kv(k, num_q_heads):
    """GQA：把 g 组 K/V 广播成 n 组。只在计算时复制，KV Cache 里始终只存 g 组。"""
    if k.shape[1] == num_q_heads:                 # 已经是 MHA，无需广播
        return k
    repeat = num_q_heads // k.shape[1]
    return (k[:, :, None, :, :]
            .expand(-1, -1, repeat, -1, -1)       # expand 是零拷贝视图
            .reshape(k.shape[0], num_q_heads, k.shape[2], k.shape[3]))  # reshape 才真正复制
```

> `expand` 不占显存、`reshape` 才物化——所以 GQA 省下的是 **KV Cache 容量**和 **K/V 的访存带宽**，省不掉 Q 侧那一份注意力计算量。面试被追问「GQA 到底省了什么」时，这个区分是加分点。

- Llama-2 70B：64 个 Q Head，8 个 KV Head → KV Cache 直接降到 1/8。
- Llama-3、Qwen 等新一代模型普遍采用 GQA。
- 代价：K/V 表达能力下降，但实验表明在合理分组下质量损失很小。

**追问：GQA 相比 MQA 好在哪？**
> MQA 压得太狠（1 组 K/V），质量下降明显且训练不稳定；GQA 在 MQA 的压缩比和 MHA 的质量之间取折中，可以按需调节 KV Head 数（如 8 组），是当前的工业标准。

## 5.3 MLA（Multi-head Latent Attention）


**速查**：MLA 把 K/V 联合压缩成一个低维 **latent（潜在向量，此处指压缩后的低维表示）** 向量（如 512 维）缓存，用时再上投影还原，把 KV Cache 压到 MHA 的 ~1/10 级别，同时质量不降反升。

**术语解释**：

- **MLA（Multi-head Latent Attention，多头潜在注意力）**：DeepSeek-V2 提出的注意力结构，通过低秩压缩 KV 表示。
- **latent vector（潜在向量）**：压缩后的低维表示，不是直接可用的 K/V，需要经上投影矩阵还原。
- **上投影 / 下投影（Up / Down Projection）**：下投影把高维压到低维，上投影把低维还原到高维。
- **decoupled RoPE（解耦旋转位置编码）**：把位置信息单独放在一小段带 RoPE 的维度上，避免位置编码破坏压缩结构。

**原理**：

- 传统做法缓存 `K` 和 `V` 两个 `[head_dim × num_heads]` 张量；MLA 改为缓存一个低维 `c_KV`（latent），推理时通过上投影矩阵还原出 K/V。
- 相比 MHA，KV Cache 可减少 **90%+**。

**追问：MLA 和 GQA 谁更好？**
> 目标不同：GQA 是「共享 K/V Head」，实现简单、通用性好、算子成熟；MLA 是「压缩表示」，压缩率更高但计算时要多一次上投影（增加计算量），且需要模型从头训练才能用（不是后训练可改的）。DeepSeek 选择 MLA 是因为从训练阶段就设计了结构。

## 5.4 FlashAttention（1/2/3 演进）


**速查**：FlashAttention 是 **IO-aware（访存感知）** 的 Attention 实现，把 Q/K/V 分块搬进 SRAM，用 online softmax 增量计算，**不物化完整的 N×N Attention 矩阵**，把显存从 O(N²) 降到 O(N)，速度提升数倍。

![图 3-1 FlashAttention 分块与 online softmax](figures/fig_03_flash_attention.png)
*图 3-1 · 把 Q/K/V 分块搬进片上 SRAM 计算，只有最终输出写回 HBM*

**术语解释**：

- **IO（Input/Output）**：此处特指 GPU 与显存之间的数据读写；**IO-aware** 指算法设计时把访存开销作为第一约束。
- **HBM（High Bandwidth Memory，高带宽显存）**：GPU 的主显存，容量大（GB 级）但相对片上存储慢很多。
- **SRAM（Static Random Access Memory，静态随机存储器）**：此处指 GPU 的**片上共享内存 / L1**，容量小（约 100 KB / SM）但带宽极高。
- **Tile / Tiling（分块）**：把大矩阵切成小块逐个处理，让数据在片上复用。
- **Online Softmax（在线 Softmax）**：分块计算 softmax 的技术，通过维护 running max 和 running sum，逐块增量更新结果，与一次性 softmax 数学等价。
- **重计算（Recomputation）**：反向传播时不保存中间结果，而是重新算一遍，用计算换显存。
- **物化（Materialize）**：把中间结果实际写进显存（与之相对的是「留在寄存器/片上」）。

```text
标准 Attention:
  S = QKᵀ        → 写入 HBM（N×N，巨大）
  P = softmax(S) → 读 HBM 再写 HBM
  O = PV         → 读 HBM
  问题：N×N 矩阵反复读写 HBM，带宽爆炸

FlashAttention:
  把 Q/K/V 切成 tile，逐个搬进 SRAM
  在 SRAM 内做 QKᵀ → online softmax → PV
  只有最终 O 写回 HBM
  代价：反向时需要重算（recompute）而不是存中间结果
```

**码 3-1 · online softmax 的参考实现（FlashAttention 的核心）**

```python
import math
import torch

def flash_attention_ref(q, k, v, block_n=128):
    """非 causal、float32 参考实现；单个 score tile 仍会物化。"""
    q, k, v = q.float(), k.float(), v.float()
    out = torch.zeros((q.shape[0], v.shape[-1]), device=q.device)
    m = torch.full((q.shape[0],), -float("inf"), device=q.device)
    l = torch.zeros(q.shape[0], device=q.device)
    for j in range(0, k.shape[0], block_n):
        k_blk, v_blk = k[j:j + block_n], v[j:j + block_n]
        s = q @ k_blk.T / math.sqrt(q.shape[-1])   # 只算当前块的分数，不写 HBM
        m_new = torch.maximum(m, s.max(dim=-1).values)
        p = torch.exp(s - m_new[:, None])          # 用新 max 重标定本块
        alpha = torch.exp(m - m_new)               # 旧累加项的修正因子
        l = l * alpha + p.sum(dim=-1)
        out = out * alpha[:, None] + p @ v_blk     # 旧结果先缩放，再并入新块
        m = m_new
    return out / l[:, None]
```

整段代码的精髓就是四行：`m_new`（新的 running max）、`alpha`（旧结果的修正因子）、`out = out * alpha + p @ v_blk`（缩放后合并）、`m = m_new`。

- **为什么必须有 `alpha`**：分块后当前块的最大值可能大于此前所有块的最大值，一旦 `m` 变大，之前用旧 `m` 算出的 `exp(s - m)` 全部偏大，必须整体乘 `exp(m_old - m_new)` 修正。`alpha` 把「重标定」从「重算整块」降成「一次缩放」——这就是 online 的含义。
- **数值校验方法**：用同一组 Q/K/V 比较本参考实现与 `torch.softmax(Q @ K.T / sqrt(D), dim=-1) @ V`，分别测多个序列长度和 block 大小；float32 下用合适的 `atol/rtol`，记录实际最大误差，不把数学等价误当逐 bit 相同。

**三代演进**：

| 版本 | 关键改进 | 收益 |
|---|---|---|
| **FA1**（2022） | Tiling + online softmax + 重计算，不物化 S/P | 显存 O(N²)→O(N)，速度 2–4× |
| **FA2**（2023） | 更好的工作划分（按序列维并行）、减少非矩阵乘 FLOPs | 比 FA1 约 2×，并行度更高 |
| **FA3**（2024） | Hopper 专属：**TMA（Tensor Memory Accelerator，张量内存加速器）**、**WGMMA（Warpgroup Matrix Multiply-Accumulate，线程组级矩阵乘指令）**、FP8、ping-pong 调度 | H100 上比 FA2 再快 1.5–2× |

**追问：FlashAttention 会减少 FLOPs 吗？**
> 不会，甚至因为重计算在**反向**时增加 FLOPs。它省的是**显存 IO**，把 Attention 从「访存受限」推向「计算受限」。所以它是 IO 优化，不是计算优化——这个区分面试官很爱考。

**追问：为什么 online softmax 是关键？**
> 因为要分块算 softmax，就必须解决「当前块的最大值可能不是全局最大值」的问题。online softmax 维护 running max 和 running sum，每处理一个新块就按新 max 重标定之前累加的结果，从而保证与一次性 softmax 数学等价。


> **接下来用在哪里**：下一章把历史 K/V 保存并分页，进入在线解码。

---

# 第 6 章 KV Cache、PagedAttention 与 Prefix Cache

> **承上**：FlashAttention 解决一次注意力计算的 IO；多轮生成还需要管理长期存活的 KV。

## 6.1 分页 KV 的最小状态机

每层 K/V 的逻辑 shape 可写成 `[request, token, kv_head, head_dim]`；分页后物理池按 Block 分配，`block_table[request][logical_block] = physical_block`。地址 `physical_block * block_size + token_pos % block_size`，实际内存布局还含 layer、K/V、head 和 dtype stride。

```python
from collections import deque

class BlockPool:
    def __init__(self, nblocks, block_size):
        self.free = deque(range(nblocks))
        self.refcount = [0] * nblocks
        self.tables = {}
        self.block_size = block_size

    def reserve(self, req_id, new_total_tokens):
        table = self.tables.setdefault(req_id, [])
        need = (new_total_tokens + self.block_size - 1) // self.block_size
        missing = need - len(table)
        if missing > len(self.free):
            raise MemoryError('KV blocks exhausted')
        for _ in range(missing):
            block = self.free.popleft()
            self.refcount[block] = 1
            table.append(block)
        return table

    def release(self, req_id):
        for block in self.tables.pop(req_id, []):
            self.refcount[block] -= 1
            if self.refcount[block] == 0:
                self.free.append(block)
```

这段只展示块分配，不含 Prefix Cache、共享块写时复制、跨层混合缓存或异步释放。真正管理器必须维护 `free + used + cached = total` 的守恒关系，释放不能重复入 free list，复用时要校验**模型、LoRA、token 序列、位置、KV dtype/布局等影响结果的上下文**。Prefix Cache 命中后要区分完整前缀和待计算尾部；是否需要重新计算一个 token 取决于引擎如何获得采样 logits，不能一概规定“必须留一个块重算”。

**源码对照**：[vLLM V1 `KVCacheManager.get_computed_blocks` / `allocate_slots`](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_manager.py) 先查已计算块，再判断新增槽位与可用块；当前实现对完整 Prefix 命中会限制最大命中长度为 `prompt_length - 1`，为取得 logits 重算末尾 token。这是该实现的明确选择；阅读其他引擎时仍要查看其 logits 取得方式。[SGLang RadixCache](https://github.com/sgl-project/sglang/blob/main/python/sglang/srt/mem_cache/radix_cache.py) 则从压缩前缀树的匹配、引用与淘汰入口读起。

**Paged Attention Kernel**：Q 由本轮计算得到，Kernel 依次从 Block Table 查 K/V 物理块，按有效长度 mask 读取，在线累加 softmax。它解决 KV 的非连续物理布局；与 FlashAttention 的 tiling 可以组合。常见 Bug：最后不满块的 mask 读入旧数据、Prefix 共享块被覆盖、抢占后表与长度不一致。调试时打印逻辑 token → `(block_id, slot)`、引用计数和前 3 个 K/V 值，与连续缓存 reference 逐 token 对齐。

## 6.2 KV Cache：推理显存大户


**速查**：KV Cache 缓存历史 Token 的 Key/Value，避免 Decode 每步重算，但显存占用随「层数 × KV Head 数 × 序列长度 × 并发数」线性增长，高并发时往往超过模型权重。

![图 1-2 KV Cache 随并发线性增长](figures/fig_01_kv_growth.png)
*图 1-2 · KV Cache 随并发线性增长，很快超过模型权重（示意数据）*

**术语解释**：

- **KV Cache（Key-Value Cache，键值缓存）**：把每层历史 Token 的 Key 和 Value 张量存下来，Decode 时直接复用，避免重复计算。它是用「显存换计算」的典型手段。
- **Head（注意力头）**：Attention 被切分成多个并行的子空间，每个子空间一个 Head；**num_kv_heads** 指 K/V 的 Head 数量。
- **head_dim（头维度）**：单个 Head 的向量维度。

**KV Cache 大小公式**：

```text
KV_bytes = 2 × num_layers × num_kv_heads × head_dim × seq_len × batch × dtype_bytes
           ↑
       K 和 V 各一份
```

**举个具体例子**：Llama-2 7B（32 层、32 KV Head、head_dim 128）FP16，单序列 4096 Token：

```text
2 × 32 × 32 × 128 × 4096 × 2 bytes ≈ 2.1 GB
```

看起来不大，但如果并发 128 路，就是 **270 GB**——远超模型权重本身（13 GB）。**这就是为什么 KV Cache 优化是推理系统的核心命题。**

**追问：KV Cache 为什么不能省？**
> 如果不缓存历史 K/V，每一步都要重复做历史 token 的 QKV 投影及前面各层计算；保留缓存后仍需对历史 K/V 做 Attention 读取，单步成本并非 O(1)。缓存可以压缩、换出或按模型结构调整，是否重算取决于显存和计算权衡。

## 6.3 PagedAttention


**速查**：PagedAttention 借鉴操作系统虚拟内存分页，把 KV Cache 切成固定大小的 **Block（块）**，用 **Block Table（块表）** 做逻辑到物理的映射，消除显存碎片，支持前缀共享，显存浪费从 60–80% 降到 <4%。

![图 3-2 PagedAttention 的 Block Table 机制](figures/fig_03_paged_attention.png)
*图 3-2 · 逻辑块经 Block Table 映射到不连续的物理块，支持共享与按需分配*

**术语解释**：

- **PagedAttention（分页注意力）**：vLLM 提出的 KV Cache 管理方案，用分页思想解决显存碎片。
- **Block（块）/ Block Table（块表）**：Block 是固定 Token 数的 KV 存储单元（如 16 个 Token）；Block Table 记录「逻辑块号 → 物理块号」的映射，物理块可以不连续。
- **显存碎片（Memory Fragmentation）**：**内部碎片**指分配了但没用满的部分；**外部碎片**指空闲空间分散、无法满足大块连续分配。
- **Copy-on-Write（写时复制）**：多个请求共享同一物理块，只有当某一方要修改时才复制出私有副本，用于 Beam Search、并行采样等场景。
- **引用计数（Reference Count）**：记录一个物理块被多少请求共享，归零才回收。

```text
传统 KV Cache：每个请求预留 max_seq_len 连续显存
  → 内部碎片（实际长度 < 预留长度）
  → 外部碎片（不同长度请求难拼合）

PagedAttention：
  Block = 固定 token 数（如 16）的 KV 存储单元
  Block Table：逻辑块号 → 物理块号（可以不连续！）
  → 按需分配，用多少占多少
  → 相同前缀可共享物理块（引用计数 + Copy-on-Write）
```

**收益**：
**码 3-2 · 逻辑块 → 物理槽位的映射**

```python
def slot_mapping(block_table, seq_len, block_size):
    """把逻辑位置翻译成物理 KV 槽位。block_table[k] 是第 k 个逻辑块的物理块号。"""
    slots = []
    for pos in range(seq_len):
        phys_block = block_table[pos // block_size]        # 间接寻址：逻辑 → 物理
        slots.append(phys_block * block_size + pos % block_size)
    return slots

# 例：block_size=4，block_table=[7, 2, 9]
# seq_len=10 → slots = [28,29,30,31, 8,9,10,11, 36,37]
#              └─ 逻辑连续 ─┘  └─ 物理跳号 ─┘
```

> `block_table` 里的物理块号**不连续**（上例从 7 跳到 2 再跳到 9），所以 kernel 只能拿着 `slots` 做 **gather**。这就是 PagedAttention 的性能代价：用非连续访存换掉了显存碎片和按最大长度预留的浪费。

- 显存浪费 <4%（论文数据），等价于并发提升 2–4×。
- 支持 **Prefix Sharing（前缀共享）**：多个请求共享同一前缀的物理块（如 system prompt、few-shot 示例）。

**追问：PagedAttention 的代价是什么？**
> ① Block Table 的间接寻址让 Kernel 复杂化（需要 **gather（按索引收集数据）**），非连续访问对带宽不友好；② 块粒度导致「最后一个不满块」无法复用（prefix cache 的固有代价）；③ 需要专门的 Paged Attention Kernel 来高效读取。

## 6.4 Prefix Cache / RadixAttention


**速查**：Prefix Cache 把已经算过的 KV Cache 按前缀哈希存起来，相同前缀的后续请求直接复用，避免重复 Prefill。SGLang 的 RadixAttention 用**基数树（Radix Tree，一种按公共前缀压缩存储的树结构）** 组织，支持更灵活的前缀匹配与 LRU 淘汰。

![图 5-3 Prefix Cache](figures/fig_05_prefix_cache.png)
*图 5-3 · 共享前缀的 KV 只算一次，后续请求直接命中复用*

**术语解释**：

- **Prefix Cache（前缀缓存）**：缓存已计算的前缀 KV，供后续同前缀请求复用。
- **RadixAttention（基数树注意力）**：SGLang 提出的方案，用基数树管理前缀 KV，支持任意长度的前缀匹配与淘汰。
- **LRU（Least Recently Used，最近最少使用）**：一种缓存淘汰策略，优先淘汰最久未访问的条目。
- **链式哈希（Chained Hash）**：后一块的哈希值由「前一块哈希 + 本块 Token」共同决定，使哈希编码完整前缀路径。

```text
请求 A: [System Prompt][User Q1]
请求 B: [System Prompt][User Q2]
        └─ 相同前缀，KV 只算一次 ─┘

实现：以 Block 为单位做链式哈希
     hash → physical_block_id 映射表
     命中则复用，未命中则新算
```

**码 5-3 · Prefix Cache 的链式哈希与碰撞兜底**

```python
import hashlib

def compute_hash(token_ids, prefix_hash=0):
    """链式哈希：把「前一块的哈希」当种子，使哈希值编码完整前缀路径。
    示意实现；生产中跨租户复用还需把模型与租户隔离等上下文编码进 key，并选择足够抗碰撞的哈希。"""
    payload = prefix_hash.to_bytes(8, "little") + b"".join(int(t).to_bytes(4, "little") for t in token_ids)
    return int.from_bytes(hashlib.blake2b(payload, digest_size=8).digest(), "big")

h, hits = 0, 0
for blk in chunked(token_ids, block_size):
    h = compute_hash(blk, h)                     # ← 依赖前一块：顺序敏感
    block = hash_to_block_id.get(h)
    if block is None:
        break                                    # 未命中：从这里开始全部重算
    if block.token_ids != blk:
        break                                    # 哈希碰撞 → 用 token_ids 二次校验兜住
    hits += 1

# 完全命中时是否重算最后一个 token，取决于引擎能否取得采样所需 logits。
```

> 链式哈希编码块顺序；示例用 token_ids 二次校验防碰撞。真实系统还需校验模型、LoRA、位置、租户隔离等上下文；完全命中时如何获得 logits 要看引擎设计。

**关键设计点**：

- **哈希必须编码完整路径**：链式哈希保证前缀顺序敏感。
- **抗碰撞**：哈希碰撞要用 token_ids 二次校验兜底。
- **完整命中后的 logits**：按引擎设计选择缓存 logits、重算末尾 token 或其他机制；不能把重算一块当普遍要求。
- **引用计数**：多请求共享同一物理块，需要 ref_count 管理生命周期。

**追问：Prefix Cache 什么时候收益最大？**
> ① 长 System Prompt / Few-shot 示例；② 多轮对话（历史上下文复用）；③ 共享长文档的 RAG（**Retrieval-Augmented Generation，检索增强生成**，先检索再生成）场景；④ Agent 场景反复调用同一上下文。**前缀越长、共享度越高，收益越大。**

## 6.5 KV Cache 管理与 Offload


**速查**：KV Cache 管理负责动态分配/释放/复用/换出。Offload 是把暂时不用的 KV 换到 CPU 内存或 NVMe，需要时再换回（用 PCIe 带宽换显存容量）。

**术语解释**：

- **Block Manager（块管理器）**：PagedAttention 的物理块分配器，维护空闲块列表与「哈希 → 块」映射。
- **Preemption（抢占）**：显存不足时强制释放部分请求的 KV，把它们放回等待队列。
- **Swap / Offload（换出 / 卸载）**：把数据从显存搬到 CPU 内存或 SSD。
- **PCIe（Peripheral Component Interconnect Express，外设互联总线）**：CPU 与 GPU 之间的数据传输通道，带宽远低于显存。
- **NVMe（Non-Volatile Memory express，非易失性内存标准）**：高速 SSD 接口协议。

| 机制 | 说明 |
|---|---|
| **Block Manager** | PagedAttention 的物理块分配器，维护 free list 与 hash→block 映射 |
| **抢占（Preemption）** | 显存不足时释放部分请求的 KV（置回 waiting 队列），优先恢复被抢占者 |
| **KV Offload / Swap** | 把 KV 换到 CPU DRAM / NVMe，PCIe 带宽成为新瓶颈 |
| **分层存储（Tiered Storage）** | GPU HBM → CPU DRAM → SSD 的多级 KV 存储 |

**追问：抢占之后怎么恢复？**
> 两条路：① 被释放的 Block 还在哈希表里 → Prefix Cache 直接命中，只需重算未缓存部分；② 已被覆盖 → 从头重算。所以「抢占」的代价取决于 Prefix Cache 的命中情况。

## 6.6 为什么是瓶颈


**速查**：KV Cache 同时吃掉**容量**和**带宽**——容量上它随并发线性增长，常超过模型权重；带宽上 Decode 每步都要把全部 KV 从 HBM 读一遍。所以它是推理系统第一优化对象。

```text
容量瓶颈：并发 ↑ → KV 总量 ↑ → OOM 或被迫降并发
带宽瓶颈：Decode 每步读 (权重 + KV) → 带宽决定 TPOT 下界
```

## 6.7 减少 KV Cache 的六类手段（面试必背）


![图 6-1 减少 KV Cache 的六类手段](figures/fig_06_kv_reduction.png)
*图 6-1 · 六类手段、机制与压缩比一览*

| 类别 | 手段 | 机制 | 压缩比 | 代价 |
|---|---|---|---|---|
| **① 数值精度** | KV Cache 量化 | FP16/BF16 → FP8 / INT8 / INT4 | INT8 ≈ 1/2，INT4 ≈ 1/4 | 精度损失、需融合反量化 |
| **② Head 数** | MQA / GQA | 减少 K/V Head 数 | 1/n ~ g/n | 轻微质量损失 |
| **③ 表示压缩** | MLA | KV 压到低维 latent | ~1/10 | 需模型原生支持、多一次上投影 |
| **④ 缓存长度** | 滑动窗口 / StreamingLLM | 只保留最近 N 个 Token | 与窗口大小成正比 | 丢失长程信息 |
| **⑤ 跨请求复用** | Prefix Cache / RadixAttention | 共享前缀的 KV | 取决于共享度 | 哈希/索引开销、Block 粒度损失 |
| **⑥ 存储层级** | KV Offload / Swap | 换出到 CPU/NVMe | 释放 HBM | PCIe/SSD 带宽瓶颈、延迟上升 |

**补充**：还有 **PagedAttention**（不减少 KV 总量，而是消除碎片、提升有效容量）和 **PD 分离**（不减少 KV，而是让 Prefill/Decode 各自最优）。

## 6.8 计算示例


**KV Cache 量化到 INT8 的存储收益**（head_dim = 128，per-(token, KV head) 对称量化，scale 存 FP16）：

```text
BF16 : 128 × 2 bytes                = 256 B
INT8 : 128 × 1 byte + 1 FP16 scale  = 130 B
比值 : 130 / 256                     ≈ 50.8%
```

**码 6-1 · KV Cache INT8 量化的 store kernel 片段**

```python
@triton.jit
def store_kvcache_int8(k_ptr, k_cache_ptr, k_scale_ptr, slot_ptr,
                       BLOCK_D: tl.constexpr):
    token, head = tl.program_id(0), tl.program_id(1)   # 一个 program = 一个 head vector
    offs = tl.arange(0, BLOCK_D)
    slot = tl.load(slot_ptr + token)                   # 逻辑位置 → 物理槽位（见码 3-2）

    key = tl.load(k_ptr + token * BLOCK_D + offs)
    # 除以 127 而不是 128：最大元素恰好映射到 ±127，加 0.5 后向零截断仍回 127
    # → 永不溢出，整个 kernel 不需要 clamp
    k_scale = tl.maximum(tl.max(tl.abs(key), axis=0) / 127.0, 1.0e-8)
    qk = key / k_scale
    # round-half-away-from-zero：先 ±0.5 再向零截断，正负号对称
    qk = tl.where(qk >= 0, qk + 0.5, qk - 0.5).to(tl.int8)

    tl.store(k_cache_ptr + slot * BLOCK_D + offs, qk)
    tl.store(k_scale_ptr + slot, k_scale.to(tl.float16))   # scale 单独存，粒度 = per (token, KV head)
```

> **两个设计选择值得记住**：
> ① **`/127` 而不是 `/128`**：最大元素映射到 ±127，`+0.5` 后是 127.5，向零截断回 127，因此**永不溢出、无需 clamp**，省掉整个 kernel 的一遍逐元素判断。
> ② **舍入是 round-half-away-from-zero**，不是 PyTorch `round()` 的 half-to-even。写法 `where(q >= 0, q + 0.5, q - 0.5)` 再向零截断，对正负号处理一致。
>
> **存储账**（head_dim = 128）：BF16 占 `128 × 2 = 256 B`，INT8 占 `128 × 1 + 1 × 2 = 130 B`，比值 **50.78%**——注意 scale 也是要占空间的，算收益时不能漏掉它。

> **一个容易被问倒的表述细节**：KV Cache 通常按「剩余显存」定容，所以量化后**显存占用总量几乎不变**，涨的是**可用块数/上下文容量**（约 1.95×）。要准确表述为「**用吞吐换上下文容量**」，而不是「省了显存」。

## 6.9 面试追问


**「KV Cache 量化后为什么端到端反而变慢？」**
> 因为反量化路径不统一。好的做法是 **Decode 融合反量化**（在 Paged Attention Kernel 里读 INT8 → 转 FP32 → 乘 scale → 直接进 online softmax，不物化浮点 Cache）；但如果 **Prefill 走的是「物化 gather + 反量化」**（按 Block Table 把 KV 还原成浮点张量再交给 **SDPA（Scaled Dot-Product Attention，PyTorch 官方提供的注意力算子封装）**），每层都要重新 gather + 反量化 + 写一遍浮点张量，这个开销会吃掉带宽收益。**结论：KV 量化的收益取决于反量化是否融合。**

**码 6-2 · 反量化的两条路径：融合 vs 物化**

```python
# 路径 A（好）：融合在 Paged Attention kernel 内部，不物化浮点 Cache
key = tl.load(k_cache_ptr + k_offsets, mask=mask, other=0.0
              ).to(tl.float32) * k_scale[:, None]
# → 直接进 online softmax，全程只多了一次乘法和一次类型转换

# 路径 B（有代价）：按 Block Table 把 KV 物化成浮点张量，再交给 SDPA
values = cache[block_ids, block_offsets].to(torch.float32)     # [seqlen, H_kv, D]
token_scales = scales[block_ids, block_offsets].float()
k_fp = (values * token_scales.unsqueeze(-1)).to(dtype)
# → 每一层都要重新 gather + 反量化 + 写一遍浮点张量
```

> 同样是把 INT8 还原成浮点，两条路径的成本差一个量级：路径 A 只在寄存器里过一遍，**带宽收益完整保留**；路径 B 要额外分配、写入、再读出 `[seqlen, H_kv, D]` 的浮点张量，**把省下的带宽又还回去了**。所以「KV 量化端到端到底快不快」，答案取决于 Prefill 侧走的是哪条路——这也是「量化收益取决于反量化是否融合」这句话的代码级解释。

**「为什么带宽减半，Kernel 只快一点？」**
> 反量化把 K/V tile 提到 FP32 精度，寄存器需求上升（往往需要更多 warp），**ALU（Arithmetic Logic Unit，算术逻辑单元）** 与寄存器压力抵消了一部分带宽收益。进一步优化方向是**代数等价改写**：先算 `k_int8 · q` 的点积，再乘 scale 向量，而不是先缩放整个 tile——乘法次数可从 O(block × head_dim) 降到 O(block)。

---


> **接下来用在哪里**：下一章用这些缓存完成单请求的 Prefill 和 Decode。

---

# 第 7 章 Prefill、Decode 与单请求生成

> **承上**：KV Cache 提供增量计算；现在可以追踪一个请求如何逐 Token 生成。

## 7.1 单请求的真实执行账本

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

## 7.2 端到端流程


**速查**：Tokenizer → Scheduler 分配资源 → Prefill 建立 KV Cache → Decode 逐 Token 生成 → Detokenize 返回。

![图 1-1 一次 LLM 推理的端到端流程](figures/fig_01_pipeline.png)
*图 1-1 · 端到端流程，以及 Prefill / Decode 两个阶段的内部构成*

```text
用户输入文本
    │
    ▼
Tokenizer ──► Token IDs
    │
    ▼
Scheduler ──► 分配计算预算 + KV Cache 资源
    │
    ▼
┌──────────────────────── Prefill ────────────────────────┐
│ Embedding → QKV Projection → RoPE → Attention → MLP →   │
│ LM Head → Logits                                        │
│ （并行处理整个 Prompt，建立 KV Cache）                    │
└─────────────────────────────────────────────────────────┘
    │
    ▼
┌──────────────────────── Decode ─────────────────────────┐
│ 每次只算 1 个新 Token，复用历史 KV Cache，循环直到         │
│ EOS 或达到 max_new_tokens                               │
└─────────────────────────────────────────────────────────┘
    │
    ▼
Detokenize ──► 返回文本（通常流式）
```

**术语解释**：

- **Scheduler（调度器）**：决定每一轮把哪些请求放进 Batch、分配多少 Token 预算和 KV Cache 空间的组件，是推理框架的核心大脑。
- **Embedding（嵌入层）**：把 Token ID 映射成稠密向量的查表层。
- **QKV Projection（QKV 投影）**：用三个（或融合为一个）线性层，把隐藏状态投影成 Attention 需要的 Query / Key / Value 三个矩阵。
- **RoPE（Rotary Position Embedding，旋转位置编码）**：通过旋转复数向量把位置信息注入 Q/K 的编码方式，是当前主流 LLM 的位置编码方案。
- **Attention（注意力机制）**：`softmax(QKᵀ/√d)V` 的计算，让每个 Token 按相关性聚合其他 Token 的信息。
- **MLP（Multi-Layer Perceptron，多层感知机）/ FFN（Feed-Forward Network，前馈网络）**：Transformer 块里 Attention 之后的两层全连接网络，通常含 SiLU 激活与门控，是参数量的主要来源。
- **LM Head（语言模型输出头）**：把最后一层隐藏状态投影到词表维度的线性层，输出 **Logits**。
- **Logits（未归一化分数）**：模型对词表中每个 Token 的原始打分，经 softmax 后成为概率分布，供 **Sampling（采样）** 使用。
- **EOS（End Of Sequence，序列结束符）**：模型输出该 Token 表示生成结束。
- **Detokenize（反分词）**：把生成的 Token ID 序列还原为可读文本。

## 7.3 Prefill 阶段


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

## 7.4 Decode 阶段


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

## 7.5 显存构成拆解


**速查**：显存 = 模型权重 + KV Cache + Activation + CUDA Graph Buffer + Workspace（+ 框架开销）。

| 组成 | 说明 | 随什么增长 |
|---|---|---|
| **Model Weight（模型权重）** | 模型参数，固定 | 模型大小、精度 |
| **KV Cache（键值缓存）** | 历史 K/V，最大变量 | 并发数 × 序列长度 |
| **Activation（激活值）** | 前向中间结果 | Batch × 序列长度（Prefill 峰值高） |
| **CUDA Graph Buffer（图缓冲）** | Graph 需要的静态输入/输出缓冲 | Graph 数量 × Batch 桶 |
| **Workspace（工作空间）** | Kernel 临时空间、通信缓冲 | 算子与并行策略 |

**在线 Serving 的关键结论**：随着并发增加，**KV Cache 很容易超过模型权重**，成为显存瓶颈。所以推理框架通常按「剩余显存」动态给 KV Cache 定容（如 vLLM 的 `gpu_memory_utilization` 参数）。

**术语解释**：

- **Serving（在线服务）**：把模型部署成可并发接收请求、流式返回结果的服务系统。

## 7.6 关键公式速查


```text
# KV Cache 显存
KV_bytes = 2 × L × H_kv × D × S × B × dtype_bytes

# Decode 理论下界（访存受限）
TPOT_min ≈ (模型权重字节数 + KV Cache 字节数) / 峰值带宽

# 单 Token 计算量（Decode）
FLOPs ≈ 2 × N_active_params        # 仅激活参数（MoE 只算激活专家）

# 算术强度
Intensity = FLOPs / Bytes
# Intensity < 机器平衡点 → 访存受限（Decode）
# Intensity > 机器平衡点 → 计算受限（Prefill、大 Batch）

# 投机解码期望输出 Token 数（含修正或额外 Token；接受率 α、Draft 长度 γ）
E[tokens] ≈ (1 - α^(γ+1)) / (1 - α)
```

---


> **接下来用在哪里**：下一章让多个请求动态共享一次 GPU 前向。

---

# 第 8 章 Continuous Batching 与 Serving Scheduler

> **承上**：单请求生成已通；并发服务必须调度 token 预算与 KV 块预算。

## 8.1 Scheduler 的两个预算与四种状态

请求状态可简化为 `WAITING → RUNNING → FINISHED`，显存不足时 `RUNNING → PREEMPTED → WAITING/RUNNING`。调度一次迭代要同时满足 **token 预算**（本轮前向的总 token）和 **KV 块预算**（本轮新增/保留物理页）；仅检查其中一个会在高并发时 OOM 或形成空转。

```python
def schedule_step(waiting, running, token_budget, pool):
    batch = []
    # 先处理正在运行的请求以保护 TPOT；策略可按 SLA 调整。
    for req in list(running):
        want = min(req.pending_tokens(), token_budget)
        if want == 0:
            continue
        try:
            table = pool.reserve(req.id, req.computed + want)
        except MemoryError:
            continue  # 实际系统可能抢占、重算、换出或拒绝
        batch.append((req, want, table))
        token_budget -= want
    for req in list(waiting):
        want = min(req.prompt_remaining(), token_budget)
        if want == 0:
            break
        try:
            table = pool.reserve(req.id, req.computed + want)
        except MemoryError:
            break
        waiting.remove(req)
        running.append(req)
        batch.append((req, want, table))
        token_budget -= want
    return batch
```

真实 vLLM V1 的 Scheduler 将请求进度表示为已计算 token 与待计算/投机 token 的差，统一覆盖 Chunked Prefill、Prefix Cache 和 Speculative Decode。读取 [vLLM Scheduler.schedule](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py) 时按五步找：`running/waiting` → 本轮 token 数 → KV 分配 → `SchedulerOutput` → 结果写回。上述代码是**教学伪代码**，并非逐行复制框架实现。

**服务指标**：同时画 TTFT、TPOT 的 P50/P95/P99 与输出 token/s，并记录 waiting/running 数、KV free blocks、preemption 次数、prefix hit tokens。总 token/s 变高而 P99 TPOT 变差时，优先检查长 Prefill 是否占用整轮预算、调度公平性和输出回传积压。追问：为何 GPU 利用率高仍可能服务差？SLA 由排队、每轮时延与尾延迟决定。

## 8.2 Continuous Batching


**速查**：Continuous Batching（连续批处理，也叫 **Iteration-level Batching，迭代级批处理**）让请求在**每次迭代**动态加入/退出 Batch，而不是等整批请求都结束。核心是提升 GPU 利用率——Decode 阶段权重读取被更多请求分摊。

![图 5-1 静态 Batching vs Continuous Batching](figures/fig_05_continuous_batching.png)
*图 5-1 · 静态 Batching 会空转；Continuous Batching 每步都重新组批，Batch 可以动态补位*

**术语解释**：

- **Continuous Batching（连续批处理）**：由 Orca 系统提出，在每次迭代（每生成一个 Token）结束时重新组 Batch。
- **Static Batching（静态批处理）**：传统做法，整批请求必须一起开始、一起结束。
- **Iteration（迭代）**：一次前向计算，Decode 阶段每迭代生成一个 Token。
- **GPU 利用率（GPU Utilization）**：需要明确是驱动显示的 busy 时间、SM active 还是运算单元利用率；Batch 不满时大量算力被浪费。

```text
静态 Batching（传统）:
  Batch = [R1, R2, R3]，必须等全部生成完才能接新请求
  → R1 早结束，它占的位置就浪费了（GPU 空转）

Continuous Batching（Orca / vLLM）:
  每个 iteration 结束都重新组 Batch
  R1 结束 → 立刻用 R4 补位
  → 尽量减少空槽，仍受请求到达与 KV 容量限制
```

**码 5-1 · Continuous Batching 的调度主循环**

```python
while waiting or running:
    budget = max_num_batched_tokens
    decode_seqs, budget = schedule_decode(running, budget)      # 先排 decode：每请求 1 token
    prefill_seqs, budget = schedule_prefill(waiting, budget)    # 剩余预算再排 prefill

    logits = model_runner.run(decode_seqs + prefill_seqs)       # 一轮前向，两批共享预算

    for seq in decode_seqs:                                     # 谁结束，谁立刻退出
        if seq.is_finished():
            running.remove(seq)
    for seq in prefill_seqs:                                    # prefill 完，立刻转 running
        running.append(seq)
```

> 关键就在最后两个循环：**退出与加入都发生在「每一轮迭代之后」**，而不是等整批跑完——这就是 Continuous Batching 与 Static Batching 的全部差别。另外注意 `decode` 与 `prefill` 共用同一个 `budget`（`max_num_batched_tokens`）和序列槽位，不是各拿一份，所以长 Prompt 的 Prefill 只能吃掉 Decode 剩下的预算，不会阻塞已在跑的请求。

**追问：Continuous Batching 的调度怎么做？**
> 一次 `schedule()` 通常返回两批——**Decode 批**和 **Prefill 批**，共享同一个 **Token Budget（Token 预算，一轮允许处理的最大 Token 数）**。先排 Decode（每个 running 请求 1 Token），剩余预算再排 Prefill，这样长 Prompt 不会阻塞已在跑的请求。running 队列用 **round-robin（轮询）** 防止饥饿；显存不够时**抢占（Preemption）** 部分请求（释放 KV），靠 Prefix Cache 或重算恢复。

## 8.3 Chunked Prefill


**速查**：把长 Prompt 的 Prefill 拆成多个 **Chunk（块）** 分批执行，避免一次性占满显存（OOM），同时可以和 Decode 混在同一 Batch 里，降低 TBT 抖动。

![图 5-2 Chunked Prefill](figures/fig_05_chunked_prefill.png)
*图 5-2 · 长 Prompt 拆块后与 Decode 混批，显存峰值可控、TBT 平滑*

**术语解释**：

- **Chunked Prefill（分块预填充）**：把一次大 Prefill 切成多个小 Chunk 分轮执行。
- **Head-of-Line Blocking（队首阻塞）**：队列头部的大请求阻塞了后面小请求的处理。

```text
不切分：8192 Token Prompt → 一次 Prefill
  → Activation + KV 峰值高，容易 OOM
  → 这一轮耗时极长，其他请求的 Decode 被卡住（TBT 抖动）

Chunked：8192 拆成 16 个 512 Token Chunk
  → 每个 Chunk 和 Decode 混批执行
  → 显存峰值可控，TBT 平滑
```

**代价**：

- 中间 Chunk 不应采样；实现可跳过其中的 LM Head/logits。代价主要是更多调度与 Kernel 边界、可能的算子效率变化。
- 调度更复杂，需要处理「未完成的 Prefill 请求」在队列中的位置。

**码 5-2 · Chunked Prefill 的切块与「中间块不采样」**

```python
CHUNK = 512
for start in range(0, len(prompt), CHUNK):
    chunk = prompt[start:start + CHUNK]
    is_last = start + CHUNK >= len(prompt)
    logits = model(prefill_chunk(chunk, kv_cache), return_logits=is_last)
    if is_last:
        next_token = sample(logits)
```

> 中间 Chunk 只推进 KV，不做采样；是否执行 LM Head 取决于实现。只有最后一个 Chunk 的末位置 logits 用于首次采样。

**追问：Chunked Prefill 和 Continuous Batching 什么关系？**
> 是互补的。Continuous Batching 解决「请求动态进出」，Chunked Prefill 解决「单个长请求内部怎么切分以平滑调度」。两者一起用才能既高吞吐又低抖动（代表实现：Sarathi-Serve）。

## 8.4 调度器设计要点


面试如果让你「设计一个推理调度器」，按这个框架答：

1. **预算（Budget）**：Token Budget 和 Sequence Slot 共享，Decode 优先、Prefill 吃剩余。
2. **公平性（Fairness）**：Round-robin 防饥饿；被抢占请求插队首（比新请求优先恢复）。
3. **内存准入（Admission）**：`can_allocate` 试探 → 逐步缩小 Chunk 直到可行 → 增量分配 KV（避免按最大长度预留）。
4. **抢占（Preemption）**：只在必要时触发；只抢占本轮尚未执行的 victim，避免释放正在使用的 Block Table。
5. **缓存（Cache）**：Prefix Cache 优先，命中即省 Prefill。
6. **流式与 SLO**：Chunked Prefill 平滑 TBT，满足延迟约束。

**术语解释**：

- **Admission（准入）**：判断一个请求当前是否有足够资源可以进入执行。
- **Victim（牺牲者）**：被选中执行抢占、需要释放资源的请求。
- **Sequence Slot（序列槽位）**：一轮调度中允许容纳的最大序列数。

---


> **接下来用在哪里**：下一章减少每步的字节、算力与前向轮数。

---

# 第 9 章 量化与投机解码

> **承上**：调度决定何时运行请求；量化与投机决定每轮运行的成本与产出。

## 9.1 量化与投机的工程验证顺序

量化从 `FP16/BF16 reference → 权重/激活统计 → scale 与打包 → kernel 解包/计算 → 逐层误差 → 任务指标` 前进。INT4 weight-only 省权重字节，但解包和反量化开销可能吞掉小 Batch 收益；W8A8/FP8 的吞吐优势依赖对应 Tensor Core 路径。KV 量化还要验证长上下文、不同位置和 Prefix 复用。

```python
def quant_check(ref, test, atol, rtol):
    diff = (ref.float() - test.float()).abs()
    ok = torch.allclose(ref.float(), test.float(), atol=atol, rtol=rtol)
    return {'ok': ok, 'max_abs': diff.max().item(),
            'mean_abs': diff.mean().item()}
```

投机解码把 Draft 的多个候选交给 Target **一次验证前向**，随后按顺序接受或拒绝。MTP 用模型额外预测头给候选，EAGLE 从特征空间构造 Draft，Medusa 用多个解码头与候选树；验证、回退和 KV 提交是系统实现的关键。看吞吐时同时统计 draft 长度、平均接受 token、Target 验证耗时、Draft 耗时、KV 回滚次数；接受率高但 Draft 太慢仍可能负收益。采样分布正确性不能用贪心验证逻辑替代随机采样的接受/修正规则。

## 9.2 量化总论


**速查**：量化是用更低精度（FP8/INT8/INT4）表示权重/激活，换取更小显存和更高吞吐，代价是精度损失。分两大类：**权重量化（Weight-only）** 和 **权重+激活量化（W8A8）**。

![图 2-1 量化的基本流程](figures/fig_02_quant_flow.png)
*图 2-1 · 对称线性量化的完整流程，以及融合反量化的重要性*

**术语解释**：

- **量化（Quantization）**：把高精度浮点数映射到低精度表示的过程；**反量化（Dequantization）** 是其逆运算，把低精度值还原成浮点参与计算。
- **PTQ（Post-Training Quantization，训练后量化）**：模型训练完成后直接做量化，无需重训，工业界推理的主流。
- **QAT（Quantization-Aware Training，量化感知训练）**：在训练时模拟量化误差，精度更好但成本高。
- **Weight-only（仅权重量化）**：只量化权重、激活仍为浮点，主要省显存，但用不上 INT8 Tensor Core。
- **W8A8（Weight 8-bit / Activation 8-bit，权重与激活均为 8 位）**：权重和激活都量化到 8 位，才能用上整数 Tensor Core 获得算力收益。
- **Scale（缩放因子）**：量化时用来把浮点范围映射到整数范围的系数，反量化时需要它还原。
- **粒度（Granularity）**：scale 的统计范围——**per-tensor**（整个张量一个 scale）/ **per-channel**（每个通道一个）/ **per-token**（每个 Token 一个）/ **per-group**（每小组一个）。粒度越细精度越好，scale 开销越大。
- **对称量化 / 非对称量化**：对称指量化范围关于 0 对称（如 ±127），实现简单；非对称引入 **zero-point（零点）** 偏移，对非零中心分布更友好。

| 分类维度 | 选项 | 说明 |
|---|---|---|
| 时机 | PTQ / QAT | 工业界推理以 PTQ 为主，QAT 精度更好但成本高 |
| 对象 | Weight-only / W8A8 / KV Cache 量化 | Weight-only 只省显存；W8A8 才能用上 INT8 Tensor Core |
| 粒度 | per-tensor / per-channel / per-token / per-group | 粒度越细精度越好，scale 开销越大 |
| 对称性 | 对称（±127） / 非对称（zero-point） | 对称实现简单，非对称对非零中心分布更好 |
| 映射 | 线性量化 / 非线性（如 NF4） | 低比特（≤4bit）常用非线性 |

**量化基本公式**（对称线性量化）：

```text
q = round(x / scale)          # 量化
x̂ = q × scale                 # 反量化
scale = max(|x|) / (2^(b-1) - 1)   # 对称，b 为位宽
```

**码 2-1 · 对称线性量化与反量化**

```python
def quantize_sym(x, bits=8, dim=None):
    """对称线性量化。dim=None → per-tensor；dim=-1 → per-token（每行一个 scale）。"""
    qmax = 2 ** (bits - 1) - 1                    # INT8 → 127
    if dim is None:
        scale = x.abs().max()                     # 整个张量共用一个 scale
    else:
        scale = x.abs().amax(dim=dim, keepdim=True)   # 每个 Token / 每个通道一个 scale
    scale = torch.clamp(scale, min=1e-8) / qmax   # 防除零，再归一化到 qmax
    q = torch.round(x / scale).clamp(-qmax, qmax).to(torch.int8)
    return q, scale

def dequantize(q, scale):
    return q.to(torch.float32) * scale            # 反量化：多一次乘法，省一半带宽
```

> **两处细节值得记住**：① `dim` 的选择就是「粒度」——`dim=None` 是 per-tensor，`dim=-1` 是 per-token，粒度越细精度越好、scale 开销越大；② `clamp` 是防御性边界保护；Kernel 是否能省略需在 scale 与舍入规则固定后逐项证明，并覆盖零张量与非有限输入。
>
> **数值校验**：per-token 量化在 4×128 的随机张量上，反量化最大误差 0.046，不超过半个量化 step，与「算子层误差 ≤ 半个 step」的结论一致。

**追问：量化会掉精度吗？**
> 会，但程度取决于粒度、方法和模型。关键是把「精度损失」分三层说：① **算子层**——反量化与浮点参考实现（reference）的误差通常 ≤ 半个量化 step；② **Attention 输出层**——**余弦相似度（Cosine Similarity，衡量两个向量方向一致性的指标）** 通常在 0.999+；③ **端到端层**——极小的 Logits 差异在接近 tie（分数接近）时会翻转 argmax（取最大值的下标），自回归会把分歧放大。**诚实的态度是：算子数值接近 ≠ 下游质量无损，生产前必须测 PPL（Perplexity，困惑度，衡量语言模型预测能力的经典指标）/ 任务指标。**

## 9.3 权重量化三剑客：GPTQ / AWQ / SmoothQuant


**速查**：GPTQ 用二阶信息逐层最小化量化误差（经典 PTQ）；AWQ 保护对激活重要的少量权重通道（激活感知）；SmoothQuant 把激活的量化难度「迁移」到权重上，让 W8A8 可行。

| 方法 | 一句话 | 核心机制 | 典型配置 |
|---|---|---|---|
| **GPTQ** | 根据量化误差逐层优化权重 | 逐层用 **Hessian（二阶偏导矩阵，此处用于估计量化误差的敏感度）** 做补偿，量化一列后更新剩余列 | INT4/INT3 weight-only |
| **AWQ** | 找重要权重，少量重点保护，其余压成 INT4 | 按激活幅值识别 **salient channel（显著通道）**（~0.1–1%），per-channel 缩放后再量化 | INT4 weight-only，快、鲁棒 |
| **SmoothQuant** | 解决 Activation Outlier，让 W8A8 更容易实现 | per-channel 缩放：`Y=(X/s)(sW)`，把激活的离群值转移到权重 | W8A8 |
| **FP8** | 用浮点指数位保留更大动态范围 | E4M3/E5M2 + per-tensor/block scale | H100/B200 高吞吐 |

**术语解释**：

- **GPTQ（Generative Pre-trained Transformer Quantization）**：一种基于二阶误差补偿的 PTQ 权重量化算法，名字来源于其针对 GPT 类模型提出。
- **AWQ（Activation-aware Weight Quantization，激活感知权重量化）**：认为「对激活影响大的通道才重要」，只保护极少数显著通道。
- **SmoothQuant**：通过数学等价变换平滑激活分布，是「Smooth（平滑）+ Quant（量化）」的合成词。
- **Activation Outlier（激活离群值）**：激活张量中少数幅值异常大的通道，会把 per-tensor 的 scale 撑大，导致其余通道量化分辨率极差。
- **NF4（4-bit NormalFloat）**：一种针对正态分布设计的非线性 4-bit 量化格式。

**追问：AWQ 和 GPTQ 的区别？**
> GPTQ 是**误差最小化**视角，用二阶信息做逐列补偿，精度好但校准慢；AWQ 是**激活感知**视角，认为只有少数通道重要，保护它们即可，实现更简单、对超参不敏感、在指令微调模型上更稳。工程上常两个都试。

**追问：为什么 SmoothQuant 能帮到 W8A8？**
> LLM 激活里存在**离群值（outlier）**，少数通道幅值极大，per-tensor 量化会被这些离群值把 scale 撑大，导致其余通道分辨率极差。SmoothQuant 用数学等价变换把难度从激活转移到权重（权重分布更平滑，好量化），从而让激活也能安全地量化到 INT8。

## 9.4 FP8


**速查**：FP8 是 8-bit 浮点，比 INT8 多了指数位，动态范围大得多，适合大模型权重/激活，且能被 Hopper/Blackwell 的 Tensor Core 原生加速。

**术语解释**：

- **FP8（8-bit Floating Point，8 位浮点数）**：总位宽 8 bit 的浮点格式。
- **E4M3 / E5M2**：FP8 的两种具体格式，`E` 后面的数字是**指数位**数量，`M` 后面是**尾数位**数量。E4M3 = 1 符号 + 4 指数 + 3 尾数（范围约 ±448，精度较高）；E5M2 = 1 + 5 + 2（范围约 ±57344，精度较低）。
- **Hopper / Blackwell**：NVIDIA 的两代 GPU 架构（H100/H200 属 Hopper，B200 属 Blackwell），原生支持 FP8 Tensor Core。
- **Transformer Engine**：NVIDIA 提供的 FP8 自动转换与缩放管理库。

**原理**：

- 因为动态范围大，通常只需 **per-tensor 或粗粒度 block scale**（如 DeepSeek-V3 用 128×128 的 block-wise 缩放），比 INT8 的 per-token 更省开销。
- 累加仍需 FP32，以保证数值精度。

**追问：FP8 和 INT8 怎么选？**
> INT8 的 scale 与分布强相关，遇到离群值需要细粒度（per-token/per-channel）；FP8 自带指数位，动态范围宽，scale 粒度可以更粗，且 Hopper 之后有原生 FP8 Tensor Core。所以新卡上 FP8 通常更省心、吞吐更高；老卡（如 Ampere 架构）只能用 INT8。

## 9.5 量化编码格式对比表（必背）


**术语解释**：下表的格式记法为「1 符号位 + E 指数位 + M 尾数位」。**指数位决定动态范围（能表示多大/多小）**，**尾数位决定精度（能表示多细）**。

| 类型 | 位宽构成 | 特点 | 用途 |
|---|---|---|---|
| **FP32** | 1+8+23 | 精度最高（单精度浮点） | 普通计算、参考实现 |
| **TF32** | 1+8+10（19 bit 存储） | 与 FP32 同范围，尾数精度降低 | Ampere Tensor Core 默认 |
| **FP16** | 1+5+10 | 精度高于 FP8，但范围小（易溢出） | 推理/训练 |
| **BF16** | 1+8+7 | 与 FP32 相同指数位，范围大 | LLM 训练/推理主流 |
| **FP8** | 1+4+3 / 1+5+2 | 更低存储、更高 Tensor Core 吞吐 | H100/B200 推理 |
| **INT8** | 1 符号 + 7 数值 | 定点表示，需 scale | W8A8、KV Cache 量化 |
| **INT4** | 1 符号 + 3 数值 | 极致压缩，精度风险高 | Weight-only（AWQ/GPTQ） |

> **速记**：**BF16 换范围，FP16 换精度，FP8 换吞吐，INT 换密度。** LLM 领域 BF16 是默认基线。
>
> 补充：**FP16（半精度）** 与 **BF16（BFloat16，Brain Floating Point 16）** 都是 16 bit，区别在指数/尾数分配——FP16 精度高但范围小，BF16 范围大但精度低。LLM 因数值范围大，通常选 BF16。

## 9.6 其他模型层手段


| 技术 | 速查 | 备注 |
|---|---|---|
| **剪枝（Pruning）** | 移除冗余权重/结构 | **非结构化剪枝**省显存但需稀疏算子支持；**结构化剪枝**直接变小模型 |
| **蒸馏（Distillation）** | 小模型学大模型分布 | 常与量化/剪枝组合，产出更小的可部署模型 |
| **稀疏 Attention（Sparse Attention）** | 只算部分 Attention 位置（滑窗/块稀疏） | Longformer、StreamingLLM、滑动窗口 |
| **MTP** | 训练时预测多个未来 Token | 见本章投机解码 |

**术语解释**：

- **知识蒸馏（Knowledge Distillation）**：让一个小模型（学生）去拟合大模型（教师）的输出分布，从而以更小体量获得接近的能力。
- **滑动窗口注意力（Sliding Window Attention）**：每个 Token 只与最近 N 个 Token 做 Attention，把 KV Cache 长度限制在常数。

---

## 9.7 核心思想


**速查**：先用成本更低的方法生成多个候选 Token（Draft），再由主模型**一次性并行验证**，从而减少 Decode 阶段的串行 Forward 次数。

![图 8-1 投机解码流程](figures/fig_08_speculative.png)
*图 8-1 · Draft 快速生成候选，Target 一次 Forward 并行验证并接受前 k 个*

**术语解释**：

- **投机解码（Speculative Decoding）**：用低成本模型生成候选、由目标模型并行验证的加速范式。
- **Draft Model（草稿模型）**：生成候选 Token 的轻量模型。
- **Target Model（目标模型）**：真正要部署的大模型，负责验证候选。
- **Forward（前向）**：一次完整的前向计算。
- **Accept / Reject（接受 / 拒绝）**：验证阶段判定候选 Token 是否被采纳。
- **Acceptance Rate（接受率，α）**：候选 Token 被接受的比例，是决定加速比的关键。
- **modified rejection sampling（改进拒绝采样）**：一种采样修正方法，保证接受结果的分布与目标模型完全一致。

```text
普通 Decode（串行）:
  Forward → t1 → Forward → t2 → Forward → t3 → ...   （N 次 Forward 出 N 个 Token）

投机解码:
  Draft 快速生成 t1,t2,t3,t4（低成本）
  Target 一次 Forward 并行验证这 4 个候选
  → 接受前 k 个（如接受 t1,t2），拒绝后续
  → 1 次 Target Forward 出了 2 个 Token
```

**码 8-1 · 投机解码的验证与接受 / 拒绝**

```python
# ① Draft 低成本生成 γ 个候选
draft = [draft_model.step(ctx) for _ in range(gamma)]

# ② Target 一次 Forward，并行得到每个位置的概率分布（唯一的「贵」操作）
p_target = target_model(ctx + draft)        # [γ, vocab]

# ③ 逐个验证：modified rejection sampling
accepted = 0
for i, tok in enumerate(draft):
    if torch.rand(()) < min(1.0, p_target[i, tok] / p_draft[i, tok]):
        accepted += 1                       # 接受，继续验证下一个
    else:
        break                               # 拒绝，后面的候选一并作废

# ④ 第一个被拒的位置，从「修正分布」重采样
if accepted < gamma:
    p_corrected = torch.clamp(p_target[accepted] - p_draft[accepted], min=0)
    next_token = sample(p_corrected / p_corrected.sum())
```

> **第 ④ 步才是「无损」的来源。** 只做 ①②③ 的话，输出分布会偏向 Draft（因为 Draft 说「是」的 token 更容易留下），必须补上「从 `max(p_target − p_draft, 0)` 归一化后的修正分布重采样」，接受的 Token 边缘分布才与目标模型完全一致。
>
> **期望输出数**（含首个拒绝后的修正 Token，或全接受后的额外 Token）`E = (1 − α^(γ+1)) / (1 − α)`：α=0.8、γ=4 → **3.36**；α=0.6、γ=4 → 2.31；α=0.5、γ=8 → **2.00**。单算被接受的 Draft Token 数，应从这里减去 1。最后一组说明：接受率只有 0.5 时，把 Draft 长度从 4 加到 8 也几乎不涨收益，反而增加 Draft 成本。

**为什么正确？** 若验证与修正采用目标模型实际生效的采样分布，并执行 **modified rejection sampling**，输出分布与直接从目标模型采样一致；近似验收模式不在此保证之内。

## 9.8 五类 Draft 来源


| 类别 | 代表 | Draft 从哪来 | 特点 |
|---|---|---|---|
| **① Draft Model** | 经典 Speculative Decoding | 一个更小的独立模型自回归生成 | 简单通用，需额外部署小模型 |
| **② MTP** | DeepSeek MTP | 主模型内置的多 Token 预测模块 | 不需额外模型，复用主模型表示 |
| **③ Hidden Feature** | EAGLE 系列 | 用 Target 的 hidden state 训练轻量 Draft 网络 | 接受率通常更高，Draft 递归但很小 |
| **④ 多预测 Head** | Medusa | 主模型后挂多个预测 Head + Tree Attention | 无需 Draft 模型，Head 很轻 |
| **⑤ 文本匹配** | Prompt Lookup / N-gram | 从上下文/历史中匹配重复片段 | 几乎零额外开销，但只对重复内容有效 |

**术语解释**：

- **MTP（Multi-Token Prediction，多 Token 预测）**：训练时让模型同时预测未来多个位置，推理时可当 Draft 模块用。
- **EAGLE**：一类用目标模型隐藏特征训练轻量 Draft 网络的方法，全称为「Extrapolation Algorithm for Greater Language-model Efficiency」。
- **Medusa（美杜莎）**：在主模型后挂多个预测头并行预测不同未来位置的方法。
- **N-gram（N 元组）**：连续 N 个 Token 构成的片段；N-gram 投机即匹配历史中已出现的片段作为候选。
- **Tree Attention（树形注意力）**：用一棵候选树一次性验证多条候选路径，共享前缀计算。
- **hidden state / hidden feature（隐藏状态 / 隐藏特征）**：Transformer 中间层的向量表示。

**补充**：还有 Self-Speculative（跳层）、LayerSkip、Lookahead / Jacobi Decoding 等不依赖独立 Draft Model 的方法。

## 9.9 统一流程


这些方法在抽象上都遵循下面的流程；单路径与候选树的具体验收规则不同：

```text
Draft → Verify → Accept / Reject
```

**逐个拆解**：

1. **Draft Model + Target Model**
   Draft Model 自回归生成 `x_{t+1}, x_{t+2}, x_{t+3}`；Target Model 把整段候选一次性 Forward，并行计算各位置概率，逐个 Accept/Reject。只要 Draft 足够轻且与 Target 分布接近，一次 Target Forward 就能接受多个 Token。

2. **MTP（Multi-Token Prediction）**
   不一定需要独立小模型，而是在主模型训练时增加 MTP 模块，让模型除了预测 t+1 还学习预测更远的未来 Token。推理时作为轻量 Draft 递归产生候选，再由主模型验证。

3. **Medusa Head**
   挂在 LLM 最后一层 hidden state 上的轻量预测头，每个 Head = `MLP/Linear+SiLU`（带残差）+ `hidden→vocab` 的 Linear。不同 Head 分别预测未来第 1、2、3… 个 Token，每位置取 Top-K 构造**候选树**，通过 Tree Attention 让原模型一次并行验证多个候选。

4. **EAGLE**
   不用多个独立 Head 直接预测，而是训练一个轻量**自回归 Draft 模型**：根据目标模型的 hidden feature 和 token embedding 预测下一步 feature，再得到 draft token，递归生成多个候选，最后 Target 一次并行验证。Draft 仍串行但很小，且能利用前一个 Token 的信息，所以接受率通常高于独立多 Token Head。

5. **N-gram / Prompt Lookup**
   直接从 Prompt 或历史生成内容中匹配重复片段作为候选，几乎没有额外模型开销。适合代码补全、摘要、RAG 等有大量重复片段的场景。

## 9.10 收益分析


**决定加速比的三要素**：

```text
① Draft 成本 c         —— 越低越好（Draft 越轻越好）
② 接受率 α             —— 越高越好（Draft 与 Target 分布越接近越好）
③ 一次 Target Forward 平均接受 Token 数
```

**期望输出 Token 数**（含修正或全接受后的额外 Token；假设每个位置的条件接受率均为 α、Draft 长度为 γ）：

```text
E[tokens] ≈ (1 - α^(γ+1)) / (1 - α)
```

**结论**：

- α 越高，收益越大；α 很低时（如 < 0.5）投机解码可能反而变慢。
- γ 不是越大越好——γ 太大时后面的候选接受率会下降，还要付出更多 Draft 成本。
- **加速比上限受限于 Target Forward 与 Draft 成本之比**。

## 9.11 各方法对比表


| 方法 | 额外模型 | 接受率 | Draft 成本 | 实现复杂度 | 典型场景 |
|---|---|---|---|---|---|
| Draft Model | 需要（独立小模型） | 中 | 中 | 低 | 通用 |
| MTP | 不需要（内置） | 中高 | 低 | 中（需模型支持） | DeepSeek 系列 |
| Medusa | 不需要（多个 Head） | 中 | 低 | 中（需训练 Head） | 通用，易集成 |
| EAGLE | 需要（轻量 Draft 网络） | **高** | 低 | 高（需训练） | 追求高接受率 |
| N-gram / Lookup | 不需要 | 场景相关 | **极低** | **最低** | 代码补全、RAG |

**追问：投机解码是「无损」的吗？**
> 在采样时使用 modified rejection sampling，并以实际生效的目标分布（含温度、top-k/top-p 等处理）验收与修正，才保证输出分布一致；贪心解码可逐位置核对 Target 的最优 Token。部分方法另有近似验收模式，例如 Medusa 的 typical acceptance，不能统称严格无损。

---


> **接下来用在哪里**：下一章处理发射、捕获和异步重叠。

---

# 第 10 章 运行时：Stream、Graph 与异步流水线

> **承上**：算子与请求队列的开销已明确；此处减少 CPU/GPU 提交与同步成本。

## 10.1 Stream、Event、Graph 的先后关系

```text
H2D stream:  拷贝 batch metadata ─ record(event)
compute:                       wait(event) ─ model kernels ─ record(done)
CPU:       准备下一批                          回收已完成请求
```

```python
copy_stream = torch.cuda.Stream()
ready = torch.cuda.Event()
with torch.cuda.stream(copy_stream):
    gpu_input.copy_(cpu_input, non_blocking=True)  # cpu_input 应为 pinned memory
    ready.record()
torch.cuda.current_stream().wait_event(ready)
out = model(gpu_input)
```

不同 stream 没有天然顺序；异步拷贝源内存过早释放或复用会造成偶发错误。CUDA Graph 将稳定的 Kernel 提交序列捕获后重放，适合 Shape/地址可固定的 Decode 桶；动态分配、CPU 数据依赖与 Batch 变化需通过 padding/bucketing 或 fallback 处理。Profiler 中如果每个小 Kernel 前都有 CPU 空隙，先看 launch 和 Graph；若 GPU Kernel 本身长，Graph 通常不是第一优先级。

## 10.2 CUDA Graph


**速查**：CUDA Graph 把一系列 Kernel 预先 **Capture（捕获）** 成一张执行图，之后通过 **Replay（重放）** 一次性提交，降低 CPU 调度、Driver 调用和 Kernel Launch 开销。**它优化的是执行调度开销，不是 Attention 的数学计算本身。**

![图 4-1 CUDA Graph 的执行方式对比](figures/fig_04_cuda_graph.png)
*图 4-1 · 普通执行逐个 launch Kernel；CUDA Graph 一次提交整张图*

**术语解释**：
**码 4-1 · CUDA Graph 的捕获与重放**

```python
g = torch.cuda.CUDAGraph()

# ① 预热：捕获前相关算子必须已在真实 stream 上跑过，否则会把初始化逻辑录进图里
static_in = torch.randn(batch, hidden, device="cuda")
for _ in range(3):
    model(static_in)

# ② 捕获：把整条 kernel 序列及其依赖关系录成一张图
with torch.cuda.graph(g):
    static_out = model(static_in)

# ③ 重放：数据必须写进同一块 static_in —— 不能换张量、不能换地址
static_in.copy_(next_batch)      # 正确：原地写入
g.replay()
out = static_out.clone()

# static_in = next_batch         # 错误：静默失效——图里绑定的仍是旧地址
```

> 两块固定地址的 `static_in` / `static_out` 就是「静态形状 + 静态地址」约束的来源，也是 Graph 额外占显存的原因。**第 ③ 步是实践中最容易踩的坑**：`copy_` 是原地写，`=` 是重新绑定变量名，后者不会报错但结果全错。

- **CUDA Graph（CUDA 执行图）**：把一串 Kernel 及其依赖关系记录成有向图，之后可整体重放。
- **Capture / Replay（捕获 / 重放）**：Capture 是把 Kernel 序列录制成图的过程；Replay 是按图一次性提交执行。
- **Kernel Launch Overhead（核函数启动开销）**：每次启动 Kernel 时 CPU 构造参数、调用驱动、提交到 GPU 的固定成本，Kernel 越短这个开销占比越高。
- **Driver（驱动）**：操作系统与 GPU 之间的中间层，CUDA 调用最终都要经过它。
- **静态 Buffer（静态缓冲区）**：预先分配、地址固定的输入/输出显存，Graph 重放要求地址稳定。

```text
普通 CUDA 执行（每步都要）：
  CPU: launch k1 → launch k2 → ... → launch kn
       ↑ 每次 launch 都有 CPU→Driver→GPU 的固定开销
  GPU: 执行 k1 → 执行 k2 → ... （CPU 可能还没提交完，GPU 在等）

CUDA Graph：
  一次性 Capture 整条 Kernel 序列 → 生成 Graph
  Replay 时一条指令提交整张图 → GPU 连续执行
```

**收益**：

- 主要省 **CPU Kernel Launch Overhead**，对「大量小 Kernel、重复执行、计算图固定」的场景收益最大。
- 最适合 **LLM Decode**：每步计算图重复、Batch 较小、单 Kernel 短，容易受 Launch 开销影响，能显著降低 TPOT。

**代价**：

- 初始化 Capture 耗时；
- Graph 占用额外显存（静态输入/输出 Buffer）；
- **要求 Tensor Shape 和显存地址尽量稳定** → 框架需预分配静态 Buffer，并按不同 Batch Size 提前 Capture 多套 Graph。

**追问：为什么长 Prompt Prefill 用 CUDA Graph 收益不明显？**
> 因为 Prefill 计算量大，计算本身占主要时间，Kernel Launch 占比很低，属于「计算受限」；CUDA Graph 省的那点 Launch 开销被淹没在计算时间里。**收益与「Launch 开销占比」成正比。**

**追问：CUDA Graph 和动态 Kernel 选择为什么不兼容？**
> Graph 在 Capture 时就固定了 Kernel 序列，Replay 时无法更换。如果框架想根据 **Context（上下文，此处指历史序列长度）** 长度动态选 Kernel（如两种 Paged Attention 实现），Graph 里就只能捕获其中一种。解决思路是做 `(batch, context_bucket)` 的图池，但显存和捕获成本会上升。

## 10.3 torch.compile


**速查**：torch.compile 是 PyTorch 2.0 的编译器，通过 `TorchDynamo → FX Graph → AOTAutograd → TorchInductor` 把模型编译成融合的 Triton/CUDA Kernel，实现算子融合、图优化和减少 Python 开销。

![图 4-2 torch.compile 编译流水线](figures/fig_04_torch_compile.png)
*图 4-2 · 编译流水线的四个阶段，以及主要优化与常见坑*

**术语解释**：

- **torch.compile**：PyTorch 2.0 引入的即时编译器（JIT Compiler）接口，一行代码即可编译模型。
- **TorchDynamo**：捕获 Python 字节码并生成计算图的组件；遇到不支持的算子会 **Graph Break（图中断）**，退回逐算子 eager 执行。
- **FX Graph（Functional eXchange Graph）**：PyTorch 的图中间表示（IR），由节点和边组成的计算图。
- **AOTAutograd（Ahead-Of-Time Autograd，提前自动微分）**：提前生成前向/反向计算图并做图级优化。
- **TorchInductor**：后端编译器，负责算子融合与代码生成（默认生成 Triton Kernel）。
- **Recompilation（重编译）**：输入 Shape 变化导致需要重新编译，带来额外开销。
- **Guard（守卫）**：运行时检查输入假设（如 Shape、dtype）是否仍然成立，不成立则重编译。

```text
PyTorch Model
    ↓
TorchDynamo        # 捕获 Python 字节码，生成 FX Graph
    ↓
FX Graph
    ↓
AOTAutograd        # 提前生成前向/反向图，做图级优化
    ↓
TorchInductor      # 后端编译器，做算子融合 + 生成代码
    ↓
Triton / CUDA Kernel
```

**主要优化**：

- **Operator Fusion**：融合逐元素算子、减少中间张量；
- **Graph Optimization**：常量折叠、死代码消除、布局优化；
- **Kernel Generation**：自动生成 Triton Kernel；
- **减少 Python Overhead**：把 Python 逐算子调用变成一次编译后的执行。

**码 4-2 · torch.compile 的用法与 Graph Break 诱因**

```python
# mode: "default" | "reduce-overhead"（含 CUDA Graph） | "max-autotune"（自动调 Triton 参数）
model = torch.compile(model, mode="max-autotune", dynamic=False)

# 反例：会触发 Graph Break —— Python 控制流依赖张量值
if x.sum() > 0:                      # 张量参与 Python 分支 → 图被切断
    x = x * 2

# 正例：改成张量运算，可以留在图内
x = torch.where(x.sum() > 0, x * 2, x)
```

> `dynamic=False` 意味着输入 shape 一变就重编译。推理服务里 batch 大小和序列长度天然波动，所以只有两条路：开 `dynamic=True`（编译期变长、运行期略慢），或按 shape 桶预编译——后者与 CUDA Graph 按 batch size 桶捕获是同一个思路。

**追问：torch.compile 的坑？**
> ① **Graph Break**——动态控制流、不支持的算子会打断图；② **Recompilation**——输入 Shape 变化会触发重新编译（动态 Shape 用 `dynamic=True` 缓解）；③ **编译耗时**——首次运行慢，不适合冷启动敏感场景；④ **Guards**——需要检查输入假设是否成立，有额外开销。

## 10.4 异步 Pipeline / Overlap


**速查**：让 CPU 调度、GPU 计算、数据传输并行执行，用「重叠（**Overlap**）」隐藏等待时间。

**术语解释**：

- **Pipeline（流水线）**：把任务拆成多个阶段，让不同阶段并行处理不同数据。
- **Overlap（重叠）**：让计算与通信/传输同时进行，从而隐藏其中一方的时间。
- **Pinned Memory（锁页内存）**：被锁定在物理内存中的主机内存，支持直接 DMA 与异步拷贝。
- **DMA（Direct Memory Access，直接内存访问）**：不经过 CPU 的数据搬运方式。
- **H2D（Host to Device，主机到设备）**：从 CPU 内存到 GPU 显存的拷贝。
- **CUDA Stream（流）**：GPU 上的任务队列，不同 Stream 之间的操作可以并发。

| 重叠对 | 做法 | 收益 |
|---|---|---|
| CPU 调度 ↔ GPU 计算 | CPU 提前准备下一个 Batch 的元数据，GPU 同时 Decode 当前 Batch | 隐藏 CPU 开销 |
| 计算 ↔ H2D 拷贝 | 用 pinned memory + 异步拷贝 | 隐藏传输延迟 |
| 计算 ↔ 通信（多卡） | 计算与 All-Reduce / All-to-All 重叠 | 隐藏通信开销 |
| Prefill ↔ Decode | 同一 Batch 混合调度 | 提升利用率、平滑 TBT |

**追问：为什么要用 pinned memory？**
> 普通内存（pageable）的 DMA 拷贝需要驱动先锁页，是同步的；pinned（page-locked）内存可以直接 DMA，支持异步拷贝，配合 CUDA Stream 就能和计算重叠。

---


> **接下来用在哪里**：下一章跨卡部署模型和缓存，观察通信代价。

---

# 第 11 章 分布式推理、MoE 与 PD 分离

> **承上**：单卡链路已完整；模型放不下或专家跨卡时需要分布式执行。

## 11.1 通信进入每层时间账本

以两卡 Tensor Parallel 的线性层为例，Column Parallel 把输出通道切分，后续 Row Parallel 把输入通道切分并在输出处归约。理想计算时间下降时，每层 All-Reduce/All-Gather 的启动延迟与传输字节会成为新瓶颈。Pipeline Parallel 按层切分，需要微批填满阶段；Expert Parallel 把 token 经 All-to-All 送到专家所在设备，再逆路由回来。

```text
TP: X → [XW0 | XW1] → 局部 Attention/MLP → All-Reduce
PP: Stage0 → 激活 P2P → Stage1（微批之间交错）
EP: Router → 按 expert 分桶 → All-to-All → Expert GEMM → All-to-All
PD: Prefill Pool → KV transfer（含 block layout/位置）→ Decode Pool
```

**NCCL 排障基线**：各 rank 必须按同样顺序进入 collective，shape/dtype/count 必须相容，所有 rank 都要报告前后日志；超时日志里的“最后一个 collective”不一定是最早出错点。先用两 rank、最小张量复现，再加 PP/TP/EP 和动态请求。检查 topology、网络、KV 传输量与通信计算重叠。[NCCL 用户指南](https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/) 与 [TensorRT-LLM 的 KV transfer 工作流](https://github.com/NVIDIA/TensorRT-LLM/blob/main/docs/source/developer-guide/kv-transfer.md) 可作为实现入口。

## 11.2 并行策略：DP / TP / PP / EP / SP


**速查**：单卡放不下或要提吞吐时，把模型/数据切到多卡。DP 复制模型切数据；TP 层内切权重（通信量大，需 NVLink）；PP 按层切（有 Bubble，需 Micro-batch）；EP 切专家（MoE 专用）；SP 切序列维度。

![图 5-4 并行策略 DP / TP / PP / EP](figures/fig_05_parallelism.png)
*图 5-4 · 四种主要并行策略的切分方式、通信模式与适用场景*

**术语解释**：

- **DP（Data Parallel，数据并行）**：每张卡放一份完整模型，把数据切分到各卡。
- **TP（Tensor Parallel，张量并行）**：把同一层的权重矩阵按维度切到多卡，层内需要频繁通信。
- **PP（Pipeline Parallel，流水线并行）**：按层切分到不同卡，形成流水线。
- **EP（Expert Parallel，专家并行）**：MoE 场景下把不同 Expert 放到不同卡。
- **SP（Sequence Parallel，序列并行）**：沿序列维度切分，常用于降低激活显存。
- **Micro-batch（微批次）**：PP 中把一个大 Batch 拆成多个小批，填充流水线以减少气泡。
- **Pipeline Bubble（流水线气泡）**：流水线启动/排空阶段部分阶段空闲造成的浪费。
- **All-Reduce（全归约）**：所有卡的数据求和后广播回每张卡。
- **All-to-All（全交换）**：每张卡把自己的一部分数据发给所有其他卡，是 EP 的核心通信模式。
- **All-Gather / Reduce-Scatter（全收集 / 归约散射）**：All-Reduce 的两个半边操作，分别用于聚合与分散。
- **P2P（Point-to-Point，点对点）**：相邻卡之间直接传输，PP 使用。
- **NVLink**：NVIDIA 的 GPU 间高速互联，带宽远高于 PCIe，适合放通信量大的 TP。
- **RDMA（Remote Direct Memory Access，远程直接内存访问）**：跨机器的高性能网络传输技术。

| 策略 | 切什么 | 通信模式 | 通信量 | 适用 |
|---|---|---|---|---|
| **DP** | 数据（模型复制） | All-Reduce（梯度） | 训练时大 | 训练 / 推理多副本 |
| **TP** | 层内权重矩阵 | All-Reduce（每层） | **很大** | 单机内（NVLink） |
| **PP** | 按层切分 | P2P（相邻 stage） | 小 | 跨机 |
| **EP** | MoE 专家 | **All-to-All** | 大 | MoE 模型 |
| **SP** | 序列维度 | All-Gather / Reduce-Scatter | 中 | 长序列 |

**追问：TP 和 PP 怎么选？**
> TP 通信量大但延迟低，适合放在**同一台机器内**（NVLink 带宽高）；PP 通信量小但会产生 Pipeline Bubble（需要 Micro-batch 填充），适合**跨机器**。实践中常用 **TP + PP + DP** 的组合（3D 并行）。

**追问：为什么 MoE 要用 EP？**
> MoE 的专家数量多（DeepSeek-V3 有 256 个），全放一张卡显存不够。EP 把不同专家放到不同卡上，Token 通过 All-to-All 路由到对应专家所在卡。代价是通信成为新瓶颈。

## 11.3 PD 分离（Prefill-Decode Disaggregation）


**速查**：把 Prefill 和 Decode 部署到不同的 GPU/机器上，各自按自己的瓶颈做优化（Prefill 配高算力卡、Decode 配高带宽卡），互不干扰。

![图 5-5 PD 分离架构](figures/fig_05_pd_disagg.png)
*图 5-5 · Prefill Pool 与 Decode Pool 分离部署，通过 KV Cache 传输衔接*

**术语解释**：

- **PD 分离（Prefill-Decode Disaggregation，预填充-解码分离）**：把两个阶段部署到独立资源池，代表工作有 DistServe、Splitwise。
- **KV Transfer（KV 传输）**：Prefill 算好的 KV Cache 传给 Decode 侧，是 PD 分离的核心开销来源。

```text
合在一起的问题：
  Prefill 是计算密集、耗时波动大
  Decode 是访存密集、要求 TBT 稳定
  两者混跑 → Prefill 会打断 Decode，TBT 抖动；资源也难各自最优

PD 分离：
  ┌─────────────┐        KV Transfer       ┌─────────────┐
  │ Prefill Pool│ ───────────────────────► │ Decode Pool │
  │ 高算力 GPU  │   (NVLink/RDMA/网络)     │ 高带宽 GPU  │
  └─────────────┘                          └─────────────┘
```

**收益**：TTFT 和 TPOT 分别优化、资源配比可按负载比例调整、故障隔离。
**代价**：KV Cache 需要跨节点传输（大 Prompt 的 KV 可能几百 MB），网络带宽成为新瓶颈；系统复杂度上升。

## 11.4 原理


**速查**：MoE 在 Transformer 的 FFN 中引入多个 **Expert（专家，指独立的 FFN 子网络）**，**Router（路由器，负责给 Token 打分并选择专家的模块）** 为每个 Token 稀疏选择 **Top-K（分数最高的 K 个）** Expert，从而用较小的 Activated FLOPs 获得更大的模型容量。

![图 7-1 MoE 的 Router 与 Top-K 选择](figures/fig_07_moe.png)
*图 7-1 · Router 为每个 Token 选择 Top-K Expert，加权合并后输出*

**术语解释**：

- **MoE（Mixture of Experts，混合专家）**：用多个专家网络 + 稀疏路由替代单一 FFN 的模型结构。
- **Router / Gate（路由器 / 门控）**：计算每个 Token 对各 Expert 的权重并选出 Top-K 的模块。
- **Top-K**：取分数最高的 K 个；MoE 中 K 通常为 1~8。
- **Activated FLOPs（激活计算量）**：单个 Token 实际触发的计算量，MoE 中只算被选中的 Expert。
- **共享专家（Shared Expert）**：所有 Token 都会经过的专家，用于提升泛化并稳定路由（DeepSeek-V3 采用）。
- **Auxiliary Loss（辅助损失）**：训练时用于约束专家负载均衡的额外损失项。

```text
Dense FFN:  x → FFN → y                    （每个 Token 都过全部参数）
MoE FFN:    x → Router → Top-K Expert → 加权求和 → y
                 ↑
        每个 Token 只激活 K 个 Expert（如 8 选 2、256 选 8）
```

**代表模型**：

| 模型 | 总参数 | 激活参数 | 专家配置 |
|---|---|---|---|
| Mixtral 8×7B | 47B | 13B | 8 Expert，Top-2 |
| DeepSeek-V3 | 671B | 37B | 256 路由 Expert + 1 共享，Top-8 |

## 11.5 MoE 与 Dense 的区别（面试高频对比）


**术语解释**：**Dense（稠密模型）** 指每个 Token 都经过全部参数的常规模型，与 MoE 的稀疏激活相对。

| 维度 | Dense | MoE |
|---|---|---|
| 每 Token 经过的参数 | 全部 | Top-K Expert |
| 总参数 vs 激活参数 | 接近 | 差距巨大 |
| 单 Token 计算量 | 与总参数成正比 | 与激活参数成正比 |
| 显存 | 模型大小决定 | **总参数决定（全部 Expert 都要装）** |
| 系统难点 | 大 GEMM | Routing、Dispatch、Grouped GEMM、负载均衡、All-to-All |
| 推理复杂度 | 相对简单 | 复杂很多 |

> **关键认知**：MoE **省的是计算，不是显存**。所有 Expert 都要加载到显存，所以 MoE 对显存容量要求反而更高（这也是 EP 存在的原因）。

## 11.6 系统难点


| 难点 | 说明 |
|---|---|
| **Routing（路由）** | Router 计算开销小但引入额外依赖，需与 Attention 重叠 |
| **Token Dispatch / Permute（Token 分发 / 重排）** | 把 Token 按 Expert 分组重排，产生大量不规则访存 |
| **Grouped GEMM（分组矩阵乘）** | 每个 Expert 的 Batch 大小不同，需要专门的分组 GEMM Kernel |
| **负载均衡（Load Balancing）** | 某些 Expert 过热会导致个别 GPU 成为瓶颈（需 aux loss / 容量因子） |
| **All-to-All 通信** | EP 下 Token 要跨卡路由到专家，通信量巨大 |

## 11.7 EP 与通信优化


- **EP（Expert Parallel）**：把不同 Expert 放到不同 GPU，Token 通过 All-to-All 送到目标卡。
- **通信优化**：DeepEP（DeepSeek 开源）做了大量 All-to-All 优化；通信与计算重叠（把 Dispatch/Combine 与 Attention/GEMM 并行）。
- **共享专家**：DeepSeek-V3 保留 1 个共享 Expert，所有 Token 都过，提升泛化并稳定路由。

**追问：MoE 推理为什么比 Dense 难做？**
> Dense 的难点是「大 GEMM」，是规整的计算问题；MoE 的难点是「路由 + 不规则访存 + 负载不均 + 跨卡通信」，是系统问题。尤其 EP 下 All-to-All 通信很容易成为瓶颈，且 Expert 负载不均衡会导致部分 GPU 空转。

---


> **接下来用在哪里**：下一章到真实框架源码里找到这些组件。

---

# 第 12 章 主流框架源码导览

> **承上**：已能从概念定位各组件；本章把它们映射到项目入口、类和函数。

## 12.1 源码读法：入口 → 状态 → 一轮执行

先固定 Git commit/版本，再按 `API/Engine → Scheduler → KV Manager → ModelRunner → Attention Backend → Sampler → Output` 追一条 request_id。只摘核心控制流，类与函数名以链接版本为准。下面是**从真实接口抽象的 16 行导读伪代码**，用于标注数据流，不是项目逐字源码：

```python
request = engine.add_request(prompt, sampling_params)
while not request.finished:
    plan = scheduler.schedule()       # token 预算、KV 块与抢占
    inputs = runner.prepare(plan)     # packed tokens、position、slot mapping
    result = runner.execute(inputs)   # embedding → layers → attention backend
    outputs = sampler(result.logits, plan.sampling_metadata)
    scheduler.update_from_output(plan, outputs)
    engine.publish(outputs)           # streaming token / finish reason
```

| 项目 | 架构/关键类 | 首先跟的函数与问题 |
|---|---|
| [vLLM V1](https://github.com/vllm-project/vllm/tree/main/vllm/v1) | `Scheduler`、`KVCacheManager`、`GPUModelRunner` | `schedule` 如何确定 token 数；`execute_model` 如何把调度输出变 GPU 输入 |
| [SGLang](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt) | Scheduler、RadixCache、ModelRunner | 前缀匹配结果如何进入新请求；树节点何时引用/淘汰 |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | Executor、BatchManager、KV Cache Manager | in-flight batching 怎样混合 context/generation；KV transfer 何时提交 |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | `llama_context`、`ggml` graph、KV cache | token batch 如何构图和执行；GGUF 量化张量如何参与计算 |
| [PyTorch](https://github.com/pytorch/pytorch) / [Triton](https://github.com/triton-lang/triton) / [CUTLASS](https://github.com/NVIDIA/cutlass) | Dispatcher / compiler / GEMM tile | 算子从 Python API 到 GPU Kernel 的边界在哪里 |

**vLLM V1 的 14 行控制流缩写**（按当前 [Scheduler.schedule](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py)、[KVCacheManager](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_manager.py)、[GPUModelRunner](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu_model_runner.py) 改写，保留关键状态而省略分支）：

```python
token_budget = scheduler.max_num_scheduled_tokens
for request in running_then_waiting:
    cached_blocks, cached_tokens = kv.get_computed_blocks(request)
    missing = request.num_tokens - request.num_computed_tokens - cached_tokens
    scheduled = min(missing, token_budget)
    if scheduled <= 0:
        continue
    new_blocks = kv.allocate_slots(request, scheduled,
                                   new_computed_blocks=cached_blocks)
    if new_blocks is None:
        handle_preemption_or_wait(request)
        continue
    plan[request.request_id] = (scheduled, new_blocks)
    token_budget -= scheduled
runner_output = model_runner.execute_model(plan)
scheduler.update_from_output(plan, runner_output)
```

阅读真实源码时，逐行把 `num_computed_tokens`、`num_tokens_with_spec`、Prefix 命中长度、`num_scheduled_tokens` 写进一个 3 请求表格；这能看出普通 Decode、Chunked Prefill、投机 token 如何由同一进度差表达。上面的 `running_then_waiting` 和 `handle_preemption_or_wait` 是教学抽象，不是实际标识符。

**源码记录模板**：记 `输入 shape/stride → 状态变更 → 分配/释放 → Kernel launch → 输出`。每次只读一条路径并做断点/日志验证；遇到版本差异以源码为准，不把教学伪代码当成正式实现。

## 12.2 推理框架对比


| 框架 | 定位 | 核心特性 | 适合场景 |
|---|---|---|---|
| **vLLM** | 通用 LLM Serving 标杆 | PagedAttention + Continuous Batching + Scheduler | 通用在线服务、研究基线 |
| **SGLang** | 强调前缀复用与 Agent | RadixAttention、Prefix 复用、结构化输出 | Agent、多轮对话、高前缀复用 |
| **TensorRT-LLM** | NVIDIA 官方极致性能 | TensorRT、CUDA Graph、**In-flight Batching**、低精度 Kernel | 追求极致 GPU 性能、N 卡生产部署 |
| **llama.cpp** | 本地 / 端侧 | **GGUF（GPT-Generated Unified Format，一种含量化权重的单文件模型格式）**、低比特量化、CPU/GPU/Metal 跨平台 | 本地推理、端侧、Mac |

**术语解释**：

- **In-flight Batching（飞行中批处理）**：TensorRT-LLM 对 Continuous Batching 的叫法。
- **Agent（智能体）**：能自主调用工具、多轮规划与执行的 LLM 应用形态，其请求通常有很长且高度重复的上下文。
- **结构化输出（Structured Output）**：用正则或 JSON Schema 约束解码过程，保证输出符合指定格式。

**追问：vLLM 和 SGLang 的区别？**
> 两者都基于 PagedAttention + Continuous Batching。SGLang 的差异点在于 **RadixAttention**（用基数树做更灵活的前缀复用，覆盖多轮/Agent 场景）和**结构化输出**（约束解码），在 Agent 与多轮对话负载下往往更有优势。


> **接下来用在哪里**：下一章用指标判断真实系统慢在何处。

---

# 第 13 章 Performance Analysis：从慢现象到瓶颈证据

> **承上**：读懂调用链后，要用测量判断哪层实际限制吞吐和延迟。

## 13.1 先分类慢，再看指标

同一模型至少做三组实验：固定 Batch 改序列长度、固定长度改 Batch、固定总 token 改请求数。先把总时间拆成排队、CPU 准备、H2D、GPU Kernel、通信、采样、回传。Nsight Systems 看时间线和空隙；Nsight Compute 深挖一个确定的 Kernel；`torch.profiler` 看 PyTorch op 与 Kernel 对应关系。`nvprof` 属旧工具，现代 GPU 优先用 Nsight；CUPTI 适合构建自动采集器。

**Roofline**：`Arithmetic Intensity = FLOPs / DRAM bytes`，理论上界 `min(峰值 FLOP/s, 强度 × 峰值带宽)`。测量时用实际 DRAM 字节和实际耗时，得到 achieved FLOP/s 与 achieved bandwidth；不能拿模型参数量直接代替所有 Kernel 的实际流量。低强度且 DRAM 接近带宽上限是 memory-bound 的证据；强度高但 Tensor Core 没工作，可能是 dtype/layout/alignment 或 fallback；两者都低，要检查 launch、同步、访存延迟或并行度。

| 指标 | 看见什么 | 下一步 |
|---|---|---|
| SM active / GPU busy | 有多少时间在执行 GPU 工作 | 低则先看 CPU 空隙、短 Kernel、batch 太小 |
| Tensor Core utilization / 指令数 | 是否走 MMA 路径 | 查 dtype、维度对齐、layout、编译产物 |
| DRAM throughput、L2 hit | 流量与缓存复用 | 区分带宽饱和和高延迟散读 |
| Occupancy、register/thread、shared/CTA | 可驻留 Warp 与资源限制 | 查 spill、tile 大小、Block 数 |
| Eligible warps / issue active | 调度器是否有可发射工作 | 再看 stall；不要只按单一 stall 百分比下结论 |

## 13.2 Warp Stall 速查与动作

| Stall | 通常代表 | 定位与优化方向 |
|---|---|---|
| Long Scoreboard | 等待 L1TEX/global/local load 的依赖 | 找产生依赖的 load；看 coalescing、L2/DRAM、预取、增加独立工作或减少 spill |
| Short Scoreboard | 等待 shared/MIO 类操作的依赖 | 看 shared 访问、bank conflict、依赖链、特殊函数 |
| MIO Throttle | MIO 指令队列拥塞 | 查 shared 操作数、散乱地址、指令宽度与局部性 |
| Barrier | 等待 CTA 屏障及线程进度 | 减少不必要屏障、平衡 Warp 工作量 |
| Not Selected | Warp 已准备好但本周期未被选中 | 这是调度状态，通常说明可发射 Warp 足够；无需直接“消除” |
| Instruction Fetch / No Instructions | 指令未就绪或 I-cache/短 Grid 等因素 | 看代码体积、分支、Grid 是否足够大 |

Stall 名称和计数器会随架构/工具版本变化；优先读 [Nsight Compute Profiling Guide 的 Warp Stall 解释](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)。`Long Scoreboard` 高并不能单独证明 DRAM 带宽打满；它也可能是访问延迟或 local memory spill。

## 13.3 案例：自定义 GEMM 只有 cuBLAS 的 72%

```text
同 shape/dtype、预热和重复计时，确认 72% 可复现
  ↓ Nsight Systems：Kernel 占主要时间，CPU launch 空隙不显著
  ↓ Nsight Compute：先确认 Tensor Core 指令实际发射
  ↓ 查 DRAM throughput 与 L2 hit；未打满带宽
  ↓ Long Scoreboard 高 + global load 事务分散
  ↓ 检查 A/B tile 地址、stride、向量化与预取
  ↓ 修改一处，重新测耗时、误差、寄存器与 occupancy
```

**命令示例**：`nsys profile -o trace python bench.py` 看时间线；`ncu --set full --kernel-name regex:gemm python bench.py` 分析目标 Kernel（详细采集开销大，勿用该次耗时当正式性能数据）；`torch.profiler.profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA])` 建立 op→Kernel 映射。报告必须写明 GPU、驱动/CUDA、shape、dtype、stride、warmup、迭代次数、统计量和参考实现。测得的 72% 是**案例输入**，不是通用性能断言。

```python
import torch
from torch.profiler import profile, ProfilerActivity, record_function

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
             record_shapes=True) as prof:
    for _ in range(10):
        with record_function("model_step"):
            output = model(input_ids)
        torch.cuda.synchronize()
print(prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=15))
prof.export_chrome_trace("trace.json")
```

这段用于定位 op 与 Kernel，不用 Profiler 包裹正式 Benchmark。Serving 若“GPU busy 很低但 TPOT 高”，先对齐请求时间线与 GPU 时间线：若 `schedule → execute` 之间有长空隙，查 CPU 元数据准备与同步；若长 Prefill Kernel 插入 Decode，查 token 预算与 Chunked Prefill；若 GPU 一直忙且 KV/DRAM 流量高，再考虑 GQA、KV 量化或缩短上下文。


> **接下来用在哪里**：下一章检查性能优化是否破坏数值正确性。

---

# 第 14 章 Model Debugging：从输出错误到第一处分歧

> **承上**：性能优化可能改变布局和精度；本章建立第一处分歧的定位流程。

## 14.1 从最终输出回溯到第一处分歧

先确认 tokenizer、权重、位置、mask、采样和模型模式一致。固定输入做 deterministic forward，比较 logits；采样 token 不同可能只是两个几乎相等的 logits 排序翻转。不能只用 `torch.allclose` 的布尔值：同时记录最大/平均绝对误差、相对误差、NaN/Inf 个数、argmax 差异与误差分布，并按 dtype/量化粒度设置容差。

```python
import torch

def compare(name, ref, test, atol=1e-3, rtol=1e-3):
    assert ref.shape == test.shape, (name, ref.shape, test.shape)
    a, b = ref.detach().float(), test.detach().float()
    diff = (a - b).abs()
    denom = a.abs().clamp_min(1e-8)
    print(name, 'allclose=', torch.allclose(a, b, atol=atol, rtol=rtol),
          'max_abs=', diff.max().item(), 'max_rel=', (diff / denom).max().item(),
          'nan=', torch.isnan(b).sum().item(), 'inf=', torch.isinf(b).sum().item())

for layer_id, (ref, test) in enumerate(zip(ref_layers, candidate_layers)):
    compare(f'layer_{layer_id}', ref, test)
```

```text
最终 logits 不一致 → Token-by-Token 比较，找第一枚不同 token
  → 该 token 的 Layer-by-Layer 比较，发现 Layer 17 开始发散
  → 比较 attention input/output 与 MLP input/output
  → Attention 异常 → Q/K/V、position_id、RoPE、mask、KV slot
  → 修 position_id → 重跑短序列/长序列/Prefix/抢占回归
```

**典型分歧**：全 NaN 先看除零、`exp` 溢出、全 mask 行、未初始化读取；FP16/BF16 差异先看累加 dtype 与运算顺序；量化后退化先看 scale 轴、zero-point、打包顺序、校准样本，再看任务指标；KV 错误先比较每步新写入 K/V 与无缓存 reference。Layer 17 是诊断示例，不意味着固定故障层。做随机测试时覆盖 `1、block_size-1、block_size、block_size+1` 的长度、非连续输入、GQA head 比例及不同 batch 顺序。

**追问**：为什么只比较最终生成文本不足以验收 Kernel？采样放大微小 logit 差异，同时也可能在错误尚未影响 argmax 时掩盖问题；应先比较中间张量和 logits，再测任务表现。


> **接下来用在哪里**：下一章把典型错误抽象成可复用的故障模式。

---

# 第 15 章 AI Infra Bug Analysis：故障模式与修复

> **承上**：已经会逐层找数值错误；本章扩大到 CUDA、缓存、调度和 NCCL 故障。

## 15.1 按“现象 → 假设 → 证据 → 修复 → 回归”记录 Bug

| 模式 | 现象 | 首查证据 | 修复与回归 |
|---|---|---|---|
| CUDA 越界 / Illegal Access | 后续 API 才报错、输出随机 | `compute-sanitizer memcheck`、索引与尾块 mask | 修边界；测最小/非整块长度 |
| Race / 同步错误 | 偶发不一致、调试时消失 | racecheck、stream/event 依赖、写后读 | 补正确同步；多次并发重复跑 |
| Deadlock / Barrier | Kernel 卡住 | 检查分支内 `__syncthreads`，各线程到达路径 | 将屏障移到所有线程可达位置 |
| Bank Conflict | 正确但 shared throughput 低 | NCU shared 事务、bank 地址算术 | 改布局/padding，再测性能 |
| Kernel index/layout/stride | 某些 shape 错误 | 打印逻辑坐标→物理 offset；与 contiguous reference 比 | 修 stride、转置、padding/mask；随机 shape 回归 |
| KV Block 泄漏 / 双重释放 | 显存持续增长或读到别人的 KV | free/used/cached/refcount 守恒、请求生命周期日志 | 原子化状态转移；取消/异常/共享回归 |
| Prefix hash 错 | 不同 prompt 命中同块，logits 错 | token 序列、模型上下文、hash key、碰撞策略 | 扩大 key 或强哈希/校验；跨租户隔离测试 |
| Preemption / Batch 错位 | 只在高并发错 token | request_id→row、position、block table 每轮快照 | 更新状态与 GPU 输入的同一版本；重排/恢复测试 |
| NCCL Hang / TP mismatch | 某 rank 挂住或 shape 报错 | 每 rank collective 序号、shape/dtype、网络日志 | 统一调用次序和分片尺寸；2 rank 最小复现 |

**完整案例：KV 泄漏**。现象是每轮完成请求后 free blocks 只降不升；先按 request_id 记录 `allocate/share/release`，发现异常取消路径没有 `release`；修复为请求完成/取消/超时均进入同一幂等清理函数；回归用“Prefix 共享 + 中途取消 + 抢占”三种交错，并断言最后 `free + cached = total` 且没有正 refcount 的孤儿块。

**完整案例：NCCL Hang**。现象是 rank 1 在 All-Reduce 等待；按每 rank 的调用序号定位 rank 0 更早因 OOM 跳过 collective；修复为所有 rank 一致传播失败状态后退出或重试，不让部分 rank 继续；用注入单 rank OOM 验证不会无限等待。调试环境可启用 `TORCH_DISTRIBUTED_DEBUG=DETAIL` 与 NCCL 日志，但要控制日志量和敏感请求信息。


> **接下来用在哪里**：下一章从 API 请求到最终 Token 做完整复盘。

---

# 第 16 章 完整推理系统分析与工程实战

> **承上**：所有组件和排障方法已齐；本章用一条真实请求链路整合。

## 16.1 一条请求的源码与数据流复盘

假设系统收到 1024 token prompt、最多生成 32 token、当前有其他运行请求。追踪时不要先问“用了哪些优化”，先在每一步记录状态：

| 阶段 | 输入/输出 | 状态改变 | 应测量/验证 |
|---|---|---|---|
| API/tokenizer | 文本→token_ids | 建 Request | token 数、模板和特殊 token |
| Admission/Scheduler | waiting→本轮计划 | 占 token 预算与 KV 块 | 排队时间、block free、prefix hit |
| Prefill | packed tokens→最后 logits | 写 1024 份 KV（可分 chunk） | position、mask、TTFT、GEMM/Attention 时间 |
| Sampler | logits→首 token | 更新输出与停止条件 | seed、top-p/top-k、logit 差异 |
| Decode loop | 1 token→下一个 logits | 每步追加 KV、动态混批 | TPOT、KV 带宽、Graph 命中 |
| 完成/取消 | 输出→流式响应 | 释放或缓存 KV | 请求生命周期、块守恒、尾延迟 |

**工程练习 A：实现**。在一张 GPU 上跑 `RMSNorm → RoPE → Attention → logits` 的最小 PyTorch reference；分别替换一个 CUDA/Triton Kernel，每次用固定输入检查 `allclose` 与边界 shape。**工程练习 B：服务**。运行 [CPU-only 的 MiniEngine 示例](examples/mini_serving.py)，以两条不同长度请求模拟连续批处理，打印每轮 `request_id / computed / scheduled / physical blocks / free blocks`；再自行扩展到三条请求与 Prefix 共享。**工程练习 C：诊断**。人为注入错误的 RoPE position 与缺少 release，分别使用逐层比较和块守恒定位；再给一个 GEMM 做 Roofline/NCU 记录。

**验收标准**：能画出 API→Scheduler→KV Manager→ModelRunner→Attention Backend→Sampler 调用链；能解释 vLLM 的分页缓存与迭代调度、FlashAttention 的 IO tiling、PagedAttention 的逻辑到物理映射；能用一个实测指标和一个反证排除错误瓶颈假设。源码导读从上一章的链接进入，按固定版本记录，不把“框架一定如此”写成跨版本定论。


> **接下来用在哪里**：下一章用问题与工程交付物检验掌握程度。

---

# 第 17 章 面试速查与能力验收

> **承上**：完成端到端追踪后，用简答与代码任务检查是否能解释、实现和调试。

## 17.1 三种能力的自测

1. **能解释**：画 Q/K/V tile 的 online softmax 更新、PagedAttention 的 block table 寻址和 Scheduler 的 token/KV 双预算，分别指出它们优化的资源。
2. **能实现**：独立实现 Reduction、Scan、Softmax、GEMM、RMSNorm、RoPE、Attention、KV BlockPool 和简化 Continuous Batching，给出边界测试与基线。
3. **能调试**：拿到“Kernel 慢、NaN、KV 错、TPOT 抖动、NCCL Hang”时，先提出可证伪假设，选择一项指标或中间张量定位，再给修复与回归。

## 17.2 高频问题 30 秒速答


| 问题 | 30 秒回答骨架 |
|---|---|
| **推理加速有哪些方法？** | 分四层：模型层（量化、GQA/MQA、MLA、投机解码）、算子层（FlashAttention、PagedAttention、算子融合、FP8）、执行层（CUDA Graph、torch.compile、异步 Pipeline）、系统层（Continuous Batching、Chunked Prefill、Prefix Cache、KV 管理、TP/PP/EP、PD 分离） |
| **Prefill 和 Decode 的区别？** | Prefill 并行处理整个 Prompt，计算密集，决定 TTFT；Decode 每步 1 Token，访存密集，决定 TPOT。前者优化算力，后者优化带宽。 |
| **KV Cache 为什么是瓶颈？** | 容量上随并发线性增长常超权重；带宽上 Decode 每步都要全量读。对策见第 6 章六类手段。 |
| **FlashAttention 省的是什么？** | 省 HBM IO，不是省 FLOPs（反向还会增加）。把 Attention 从访存受限推向计算受限。 |
| **PagedAttention 解决什么？** | 显存碎片（浪费从 60–80% 降到 <4%），并支持前缀共享和 Copy-on-Write。 |
| **CUDA Graph 的好处？** | 减少 Decode 每步的 CPU 调度和 Kernel Launch 开销；不减少 GPU 计算。适合小 Kernel、重复执行、图固定的场景。 |
| **Continuous Batching 原理？** | 每个 iteration 重新组 Batch，请求动态进出，把 Decode 的权重读取分摊到更多请求。 |
| **Chunked Prefill 的代价？** | 中间 Chunk 的采样结果被丢弃（浪费一个 Token），调度更复杂；换来的是 TBT 平滑和防 OOM。 |
| **Prefix Cache 怎么保证正确？** | 缓存 key 覆盖模型/租户/位置等上下文 + 抗碰撞策略 + 明确完整命中时 logits 的取得方式。 |
| **量化会掉精度吗？** | 分三层说：算子层（≤半个 step）、Attention 输出层（cosine 0.999+）、端到端层（接近 tie 会翻转 argmax 并被自回归放大）。必须测 PPL。 |
| **MoE 省什么？** | 省计算（激活参数少），**不省显存**（所有 Expert 都要装）。难点是路由、负载均衡和 All-to-All 通信。 |
| **投机解码何时无损？** | 用目标模型的实际采样分布做精确验收与修正时，输出分布与直接采样一致。 |
| **PD 分离的动机？** | Prefill 计算密集、Decode 访存密集，混跑互相干扰；分离后各自最优，代价是 KV 跨节点传输。 |
| **TP 和 PP 怎么选？** | TP 通信量大、延迟低，放机内（NVLink）；PP 通信量小、有 Bubble，跨机。常用 3D 并行组合。 |

## 17.3 必背对比表


**① 四种 Attention 变体**

| 变体 | KV Head 数 | KV Cache | 质量 | 代表 |
|---|---|---|---|---|
| MHA | = Q Head | 1.0× | 最好 | 早期模型 |
| GQA | < Q Head（分组） | g/n× | 接近 MHA | Llama-2 70B / Llama-3 |
| MQA | 1 | 1/n× | 有损失 | PaLM、Falcon |
| MLA | latent 压缩 | ~0.1× | 好 | DeepSeek-V2/V3 |

**② 量化方法**

| 方法 | 类型 | 粒度 | 特点 |
|---|---|---|---|
| GPTQ | PTQ weight-only | per-group | 二阶误差补偿，经典 |
| AWQ | PTQ weight-only | per-channel | 激活感知，保护 salient 通道 |
| SmoothQuant | W8A8 | per-channel | 迁移激活离群值到权重 |
| FP8 | W8A8 | per-tensor/block | 动态范围大，新卡首选 |

**③ 四层加速技术一览**（见 0.2 总览表）

## 17.4 数字速记


| 数字 | 含义 |
|---|---|
| **2 × L × H_kv × D × S × B × dtype** | KV Cache 大小公式 |
| **< 4%** | PagedAttention 显存浪费 |
| **2–24×** | vLLM 相比 HF 的吞吐提升（论文） |
| **~90%** | MLA 相比 MHA 的 KV Cache 减少量 |
| **1/2、1/4** | INT8、INT4 KV 量化的理论压缩比 |
| **2×** | FP8 相比 FP16 的 Tensor Core 峰值吞吐 |
| **1.5–2×** | FA3 相比 FA2 在 H100 上的提升 |
| **~0.999+** | 量化后 Attention 输出的 cosine similarity 量级 |

## 17.5 常见易错点 / 陷阱


| 易错点 | 正确认知 |
|---|---|
| ❌ FlashAttention 减少 FLOPs | ✅ 减少 HBM IO；反向反而增加 FLOPs |
| ❌ CUDA Graph 提升 GPU 计算速度 | ✅ 只减少 CPU Launch/调度开销 |
| ❌ KV 量化省显存 | ✅ 显存按剩余定容，涨的是**可用块数/上下文容量** |
| ❌ MoE 省显存 | ✅ 省计算；所有 Expert 都要驻留显存 |
| ❌ 投机解码必然有损 | ✅ 精确验收可保持目标分布；近似验收另当别论 |
| ❌ Prefix Cache 只需比较 token IDs | ✅ 还要包含模型/LoRA/位置/隔离上下文；完整命中时另行处理 logits |
| ❌ 量化后 attention 数值接近就代表质量无损 | ✅ 需测 PPL / 任务指标，exact token agreement 不是语言质量指标 |
| ❌ 静态 Batching 就够了 | ✅ 静态 Batching 会因请求长度不一浪费大量算力 |
| ❌ Chunked Prefill 没有代价 | ✅ 更多调度边界与可能的算子效率变化，尾延迟需实测 |
| ❌ 带宽减半 Kernel 就快一倍 | ✅ 受寄存器压力、ALU、非连续访存等影响，实际收益打折 |

## 17.6 反问面试官的问题（加分项）


- 团队目前推理服务的主要瓶颈是 TTFT 还是 TPOT？有没有用 PD 分离？
- 线上主要用哪套框架（vLLM / SGLang / TensorRT-LLM）？自研 Kernel 的比重有多大？
- 模型侧有没有考虑 MLA / MoE 这类结构优化，还是以量化为主？
- 有没有多卡 / 多机部署，通信是不是瓶颈？
- 对精度（PPL / 任务指标）有没有明确的验收标准？

---


> **接下来用在哪里**：下一章把前面的 Kernel、Serving、Profiling 和 Debug 方法用于跨设备模型适配。

---

# 第 18 章 GPU/NPU 模型适配：能跑、跑对、跑快、跑稳

> **二面主线**：模型适配不是把 `.cuda()` 改成 `.npu()`。面试回答先交代目标模型、目标硬件和服务负载，再按**环境与框架 → 算子与结构 → 数值精度 → 性能 → 分布式 → 稳定性**给出证据。遇到问题先分类，再找到第一处失败或分歧。

## 18.1 先明确适配边界与验收基线

适配前冻结模型权重与配置、tokenizer/chat template、输入及采样参数、dtype/量化方式、目标设备型号、框架与运行时版本。至少保存一份原平台 reference：固定输入的中间张量、logits 和任务指标，以及 Prefill/Decode 的延迟与显存数据。不同后端的浮点计算顺序可能不同，因此先约定逐算子容差与模型级质量门槛，而不是要求所有生成文本逐字一致。

| 维度 | 最小验收证据 | 典型漏项 |
|---|---|---|
| 能跑 | Load、Forward、Prefill、Decode、Generate 都成功；明确设备与 dtype | 只跑 `batch=1, seq=128` |
| 跑对 | 分层/分算子误差、logits、Top-K、PPL 或任务指标 | 只看是否生成一句通顺的话 |
| 跑快 | 同负载下 TTFT、TPOT、输出 tokens/s、吞吐、显存峰值 | 只报一个 Kernel 的加速比 |
| 跑稳 | 长时运行、高并发、长上下文、取消/抢占/恢复 | 没有验证 KV 块回收与内存趋势 |
| 覆盖率 | Batch、序列长度、动态 Shape、边界长度、并发矩阵 | 只验证单一静态 Shape |

建议把验收矩阵写成 `batch × prompt_len × output_len × dtype × 并发数 × 并行度`，优先覆盖块边界（`block_size−1 / block_size / block_size+1`）、最长上下文、空或短输入、Prefix 命中、请求取消与资源紧张。每个失败样例保留输入、版本矩阵、首次异常位置和 trace，方便复现。

## 18.2 NPU 软件栈：从设备到模型

NPU 是面向神经网络张量运算的加速器；CPU 更适合控制与通用逻辑，GPU 提供大规模并行计算，NPU 的算子、内存与编译执行路径则由具体平台决定。以 NVIDIA 与昇腾为例：

```text
NVIDIA: PyTorch/推理框架 → CUDA → cuBLAS/cuDNN/NCCL 等 → GPU
昇腾:  PyTorch/推理框架 → TorchNPU (torch_npu) → CANN/HCCL → Ascend NPU
```

CUDA 自定义 Kernel、CUDA 版 FlashAttention 或 PagedAttention **不能直接在昇腾上执行**。需要先核对目标版本是否已有等价 NPU 算子，再考虑用已有算子组合、替换 attention backend、修改布局，最后才实现自定义算子。已有算子“名称相同”也不能跳过 shape、mask、精度与性能验证。昇腾的 [TorchNPU 项目](https://github.com/Ascend/pytorch)提供 PyTorch 设备适配；其[版本配套表](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md)要求检查 PyTorch、TorchNPU、CANN、Python、驱动与固件组合；[HCCL 文档](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/commlib/hcclug/docs/en/user_guide/hccl_intro.md)说明集合通信支持。具体 API 与工具名称随版本变动，应以当前部署版本文档为准。

**启动顺序**：设备可见与健康 → 驱动/固件 → CANN Runtime → PyTorch/TorchNPU → 最小张量 MatMul → 模型加载 → 首个 Forward → Prefill/Decode。若最小 MatMul 都失败，先修环境；不要直接改模型代码。记录版本、设备型号、容器镜像和环境变量，使成功运行可复现。

## 18.3 算子与模型结构适配：先做清单再改代码

从计算图列出 Linear/MatMul、RMSNorm、RoPE、Attention、Softmax、SiLU、TopK/Sampling、KV 读写和集合通信；逐项记录输入输出 `shape/dtype/stride/layout`、是否支持动态 Shape、目标设备实现和回退路径。对每个不支持的算子按“现成 NPU 实现 → 算子拆解 → 自定义 Kernel”选择，并记录拆解后是否引入额外搬运、同步或精度误差。

| 模型特性 | 适配时要核对的语义 | 常见故障表现 |
|---|---|---|
| GQA/MQA | Q Head 与 KV Head 映射、KV 缓存布局 | 某些 Head 的 logits 错、长上下文才错 |
| MLA / 特殊 Attention | 压缩与解压路径、mask、位置编码、缓存格式 | Prefill 对而 Decode 错 |
| RoPE / MRoPE | 配对布局、position_id、多轴位置 | 首轮正常，Prefix/分块/多模态错误 |
| MoE | Router Top-K、Expert 排布、Token 重排与还原 | 并发或多卡时输出错/通信慢 |
| MTP / 特殊解码 | 额外预测头、候选 token 验证、状态推进 | 生成位置错位或接受率异常 |
| 动态 Shape | 编译缓存、padding、尾块 mask、layout 转换 | 某些 batch/长度才报错或重编译 |

**适配顺序**：先保证 eager、单卡、FP32/BF16 reference 路径正确；再接 KV Cache 与分块/并发；最后启用融合、量化、Graph 与多卡。这样每一步都能知道是哪项变化引入分歧。特殊结构（如 Gated DeltaNet）应先对照模型定义核对状态更新，再判断现有 Attention Kernel 是否适用，不能按普通 Transformer 路径硬套。

## 18.4 跑不起来：找第一个真实错误

```text
设备未识别 → 驱动/固件/容器设备映射
环境初始化失败 → Runtime、框架、插件版本配套
权重加载失败 → checkpoint key、模型配置、dtype、内存
首次 Forward 失败 → 第一个 Unsupported Op、shape、layout、device
仅某些 Shape 失败 → 动态 Shape、mask、编译缓存、尾块
仅多卡失败 → rank、通信组、collective 次序和分片尺寸
```

异步设备错误可能在后续同步点才浮现。保留**第一条异常之前**的算子与输入记录；必要时用同步执行或最小算子复现定位最早失败点，再逐层恢复异步与并发。不要把最后一个笼统 Runtime Error 当成根因。`device`、`dtype`、`shape`、`stride`、position、mask 和当前 rank 是最基本的现场信息。

## 18.5 跑得起来但结果错：Reference 与第一处分歧

先固定输入、权重、tokenizer、位置、mask、随机种子与采样方式，比较**同一次 Forward 的 logits**。如果不同，按 Layer 输出二分定位第一个异常层，再拆成 RMSNorm → QKV → RoPE → Attention（score、softmax、输出）→ MLP → residual；Decode 还要比对每一步新写入的 K/V、block table 和历史长度。这里的“二分”是减少人工检查范围；若误差缓慢累积，还需看每层误差曲线与容差，而不能只选一个布尔失败层。

| 指标 | 计算/解释 | 注意点 |
|---|---|---|
| 最大/平均绝对误差 | `abs(ref−test)` | 适合观察整体规模与极值 |
| 相对误差 | `abs(ref−test)/(abs(ref)+ε)` | reference 接近零时单独看会夸大 |
| Cosine Similarity | 向量方向一致性 | 高相似度仍可能掩盖局部大错 |
| `allclose` | 按 `atol + rtol × abs(ref)` 判定 | 容差须按 dtype/算子设定 |
| Logits/Top-K/PPL | 模型级与任务级表现 | 采样微扰会放大为不同生成文本 |

先查系统性错误：权重转置、维度广播、RoPE 排列、mask 方向、scale 轴、量化 zero-point、累计精度；再查偶发错误：未初始化读、越界、Stream 竞争、KV 共享/释放。对 FP16/BF16，记录累加 dtype 与融合前后运算顺序。**验收不能仅看最终 token 是否完全一致**：接近并列的 logits 可能因微小浮点差异改变采样路径，但任务质量仍可接受；反过来，错误也可能暂未改变 argmax。

## 18.6 跑得对但慢：跨设备性能定位

先用相同模型、负载和统计口径测端到端 TTFT、TPOT、吞吐与内存；再拆 Queue → Scheduler → Prefill → Decode → Sampling → Response。GPU 上用 Nsight Systems 看 CPU/GPU 时间线和空隙，用 Nsight Compute 看已确认的热点 Kernel；NPU 上用目标 CANN 版本的 Profiling 工具查看算子耗时、设备时间线与通信，不要把 CUDA 专属计数器名称直接套到 NPU。若阶段占比为 Attention 40%、MatMul 30%、Transpose 15%，先验证 Top-K 热点能否解释总耗时，再做一个改动并重测端到端结果。

```text
设备时间线有大片空洞？ → CPU 准备、调度、同步、H2D、短 Kernel/小 Batch、通信等待
设备持续忙但单算子慢？ → dtype/layout/对齐、访存、算术强度、并行度、融合与编译结果
算子在 CPU 执行？       → 检查不支持算子或显式搬运导致的 CPU fallback
仅并发时慢？           → KV 容量、抢占、调度公平性、通信与尾延迟
```

**CPU fallback** 的典型路径是 NPU → CPU 算子 → NPU，功能与精度可能都对，但传输和同步造成明显延迟。用 op→device 映射、设备 trace 和拷贝事件证明它是否存在；不能仅凭“性能差”断言发生了 fallback。类似地，Transpose/Format Convert 可能不是计算热点，却在每层重复出现，累计成本很高。迁移后应优先消除不必要布局转换与同步，再评估算子融合和专用 Kernel。

**指标、日志、Trace 各回答一个问题**：Metrics（QPS、TTFT/TPOT 的 P50/P95/P99、token/s、设备/KV 利用率）说明当前症状；Logs（加载失败、OOM、抢占、算子异常）说明发生的事件；Trace 用同一 request_id 串起 HTTP、Scheduler、Prefill、Decode、Sampling、响应，定位实际耗时点。记录时间戳与请求、rank、设备关联，避免把不同请求的事件误拼成一条链。

## 18.7 多卡适配：计算分片与通信一致性

TP 中，Column Parallel 把 `W=[W₁,W₂]` 按输出维切分，各设备先算本地输出；后续需要完整输出时才做 AllGather，也可能由下一层直接消费分片。Row Parallel 把输入/权重按归约维切分，局部部分和通常通过 AllReduce 或 ReduceScatter 合并。通信要按实际计算图决定，不能说“Column Parallel 一定 AllGather”。PP 还要测流水线空泡，DP 要看实例/批次分配，MoE 的 EP 重点关注 Token 路由、AllToAll、Expert 负载偏斜与还原顺序。

调试多卡时给每个 rank 的 collective 编号、shape、dtype、group、输入 token 数和错误状态打点。某 rank 先 OOM 或跳过分支，其他 rank 可能表现为 HCCL/NCCL Hang；根因可能在更早的单卡错误。验证单卡 → 双卡 → 目标并行度，比较单卡与分布式 logits、吞吐、P99 和通信占比，再判断 TP/EP 的容量收益能否覆盖通信成本。

## 18.8 稳定性、OOM 与工程环境

推理内存按**权重 + KV Cache + 激活 + Graph/编译缓存 + 临时 Workspace + 分配器碎片**列账。并发升高才 OOM，先看 `max_num_seqs`、上下文上限、KV 块总数、Prefix Cache 与抢占；显存持续单向增长，检查张量/Block 生命周期和取消、异常路径。抢占可选择释放后重算、利用仍在缓存中的前缀，或把状态换出到主机内存；后者节省设备内存但增加传输与状态管理。每轮可核对 `free + occupied + cached = total`（按实现定义避免重复计数），结束后检查没有孤儿引用计数。

长时压测同时覆盖请求取消、超时、Prefix 共享、块边界、动态 Batch、长上下文和多卡异常传播。监控 NaN/Inf、OOM、crash、内存峰值/趋势、抢占次数、恢复成本、P99 TTFT/TPOT；对可复现故障保存最小输入与版本矩阵。

Linux/Docker 是复现环境的基本工具：记录镜像、容器启动参数、Volume、设备映射、端口和环境变量；用 `docker ps/logs/inspect/exec` 对照进程、日志与设备可见性。容器共享 Host Kernel，设备驱动通常依赖宿主机；镜像内的软件栈仍要与宿主驱动/固件和目标设备匹配。先在相同容器里跑最小 MatMul，再比较模型差异。

## 18.9 面试回答与实操交付

**什么是模型适配？**“把模型从原软硬件栈迁到目标设备，覆盖框架、算子/模型结构、精度、性能、分布式和稳定性；我用能跑、跑对、跑快、跑稳验收，而不是只看设备 API 改动。”

**输出错误怎么定位？**“先固定模型、输入、dtype 和版本，保存 reference；比较 logits，再逐层和逐算子找第一处分歧；记录误差指标、shape/layout、位置和 KV 状态，修复后回归边界与并发场景。”

**GPU 利用率低怎么定位？**“先看系统时间线区分设备是否无任务可做；若有空洞，查 CPU 准备、launch、同步、传输和通信；若持续有工作，再对热点 Kernel 看 Roofline、吞吐、缓存、Warp Stall、寄存器与 occupancy。NPU 用对应平台工具和计数器走同样的分层思路。”

**怎么证明适配成功？**“交付版本/算子兼容矩阵、逐层误差表、端到端性能报告、Shape 覆盖矩阵和长时压力记录；每一项有可复现输入与通过阈值。”

二面准备按“是什么 → 为什么 → 怎么实现 → 出问题怎么定位 → 怎样证明收益 → Trade-off”组织回答。优先把本章与第 13–15 章连读，再对照第 7–8 章的 Prefill/Decode、KV 与调度账本，用同一个真实项目案例说明每一步的证据。

---
# 附录

## 附录 A · 术语表

按首次出现顺序排列，标注「缩写 · 英文全称 · 中文名」。

### A.1 推理与指标

| 缩写 | 英文全称 | 中文名 |
|---|---|---|
| LLM | Large Language Model | 大语言模型 |
| Token | — | 词元（最小文本单位） |
| TTFT | Time To First Token | 首 Token 延迟 |
| TPOT | Time Per Output Token | 每输出 Token 耗时 |
| TBT | Time Between Tokens | Token 间隔 |
| E2E | End-to-End | 端到端 |
| MFU | Model FLOPs Utilization | 模型算力利用率 |
| MBU | Memory Bandwidth Utilization | 显存带宽利用率 |
| Goodput | — | 有效吞吐量 |
| SLO | Service Level Objective | 服务等级目标 |
| FLOPs | Floating-point Operations | 浮点运算次数 |
| PPL | Perplexity | 困惑度 |
| EOS | End Of Sequence | 序列结束符 |

### A.2 模型结构与训练

| 缩写 | 英文全称 | 中文名 |
|---|---|---|
| Prefill | — | 预填充阶段 |
| Decode | — | 解码阶段 |
| KV Cache | Key-Value Cache | 键值缓存 |
| Attention | — | 注意力机制 |
| MHA | Multi-Head Attention | 多头注意力 |
| MQA | Multi-Query Attention | 多查询注意力 |
| GQA | Grouped-Query Attention | 分组查询注意力 |
| MLA | Multi-head Latent Attention | 多头潜在注意力 |
| RoPE | Rotary Position Embedding | 旋转位置编码 |
| FFN | Feed-Forward Network | 前馈网络 |
| MLP | Multi-Layer Perceptron | 多层感知机 |
| RMSNorm | Root Mean Square Layer Normalization | 均方根层归一化 |
| SiLU | Sigmoid Linear Unit | SiLU（Swish）激活函数 |
| LM Head | Language Model Head | 语言模型输出头 |
| MoE | Mixture of Experts | 混合专家 |
| MTP | Multi-Token Prediction | 多 Token 预测 |
| AR | Autoregressive | 自回归 |
| Causal Mask | — | 因果掩码 |
| latent | — | 潜在向量 |
| hidden state | — | 隐藏状态 |

### A.3 量化

| 缩写 | 英文全称 | 中文名 |
|---|---|---|
| FP32 / FP16 / BF16 | 32 / 16-bit Floating Point (BFloat16) | 单精度 / 半精度 / 脑浮点 16 |
| FP8 | 8-bit Floating Point | 8 位浮点数 |
| TF32 | TensorFloat-32 | 张量浮点 32 |
| INT8 / INT4 | 8 / 4-bit Integer | 8 位 / 4 位整型 |
| PTQ | Post-Training Quantization | 训练后量化 |
| QAT | Quantization-Aware Training | 量化感知训练 |
| W8A8 | Weight 8-bit / Activation 8-bit | 权重与激活均 8 位 |
| AWQ | Activation-aware Weight Quantization | 激活感知权重量化 |
| GPTQ | Generative Pre-trained Transformer Quantization | GPT 类模型量化算法 |
| NF4 | 4-bit NormalFloat | 4 位正态浮点 |

### A.4 算子与硬件

| 缩写 | 英文全称 | 中文名 |
|---|---|---|
| HBM | High Bandwidth Memory | 高带宽显存 |
| SRAM | Static Random Access Memory | 静态随机存储器（片上共享内存） |
| SM | Streaming Multiprocessor | 流多处理器 |
| GEMM | General Matrix Multiply | 通用矩阵乘法 |
| GEMV | General Matrix-Vector multiply | 通用矩阵向量乘 |
| IO | Input/Output | 输入/输出（此处指显存读写） |
| TMA | Tensor Memory Accelerator | 张量内存加速器 |
| WGMMA | Warpgroup Matrix Multiply-Accumulate | 线程组级矩阵乘累加指令 |
| ALU | Arithmetic Logic Unit | 算术逻辑单元 |
| Warp | — | 线程束（32 线程） |
| Bank | — | 存储体 |
| Bank Conflict | — | 存储体冲突 |
| Padding | — | 填充 |
| Swizzle | — | 地址重排 |
| Tensor Core | — | 张量核心 |
| SDPA | Scaled Dot-Product Attention | 缩放点积注意力算子 |
| Tiling | — | 分块 |
| Online Softmax | — | 在线 Softmax |
| Recomputation | — | 重计算 |

### A.5 执行与系统

| 缩写 | 英文全称 | 中文名 |
|---|---|---|
| CUDA Graph | — | CUDA 执行图 |
| Capture / Replay | — | 捕获 / 重放 |
| Kernel Launch | — | 核函数启动 |
| Graph Break | — | 图中断 |
| FX Graph | Functional eXchange Graph | PyTorch 图中间表示 |
| AOTAutograd | Ahead-Of-Time Autograd | 提前自动微分 |
| DMA | Direct Memory Access | 直接内存访问 |
| H2D | Host to Device | 主机到设备 |
| Pinned Memory | — | 锁页内存 |
| Continuous Batching | — | 连续批处理 |
| Chunked Prefill | — | 分块预填充 |
| Prefix Cache | — | 前缀缓存 |
| RadixAttention | Radix Attention | 基数树注意力 |
| Radix Tree | — | 基数树 |
| LRU | Least Recently Used | 最近最少使用 |
| Block / Block Table | — | 块 / 块表 |
| Copy-on-Write | — | 写时复制 |
| Preemption | — | 抢占 |
| Admission | — | 准入 |
| Victim | — | 牺牲者 |
| Offload / Swap | — | 卸载 / 换出 |
| Head-of-Line Blocking | — | 队首阻塞 |
| PD 分离 | Prefill-Decode Disaggregation | 预填充-解码分离 |

### A.6 并行与通信

| 缩写 | 英文全称 | 中文名 |
|---|---|---|
| DP | Data Parallel | 数据并行 |
| TP | Tensor Parallel | 张量并行 |
| PP | Pipeline Parallel | 流水线并行 |
| EP | Expert Parallel | 专家并行 |
| SP | Sequence Parallel | 序列并行 |
| All-Reduce | — | 全归约 |
| All-to-All | — | 全交换 |
| All-Gather | — | 全收集 |
| Reduce-Scatter | — | 归约散射 |
| P2P | Point-to-Point | 点对点 |
| Pipeline Bubble | — | 流水线气泡 |
| Micro-batch | — | 微批次 |
| NVLink | — | NVIDIA GPU 间高速互联 |
| PCIe | Peripheral Component Interconnect Express | 外设互联总线 |
| RDMA | Remote Direct Memory Access | 远程直接内存访问 |
| NVMe | Non-Volatile Memory express | 非易失性内存标准 |

### A.7 投机解码与框架

| 缩写 | 英文全称 | 中文名 |
|---|---|---|
| Speculative Decoding | — | 投机解码 |
| Draft Model | — | 草稿模型 |
| Target Model | — | 目标模型 |
| Acceptance Rate | — | 接受率 |
| modified rejection sampling | — | 改进拒绝采样 |
| Medusa | — | 美杜莎（多头预测） |
| EAGLE | Extrapolation Algorithm for Greater Language-model Efficiency | EAGLE 投机解码 |
| N-gram | — | N 元组 |
| Tree Attention | — | 树形注意力 |
| GGUF | GPT-Generated Unified Format | 单文件量化模型格式 |
| RAG | Retrieval-Augmented Generation | 检索增强生成 |
| In-flight Batching | — | 飞行中批处理 |

## 附录 B · 论文与资料清单

**Attention 与 Kernel**
- FlashAttention (2022) / FlashAttention-2 (2023) / FlashAttention-3 (2024)
- PagedAttention · vLLM (SOSP 2023)
- Triton: An Intermediate Language and Compiler (2019)

**调度与 Serving**
- Orca: Continuous Batching (OSDI 2022)
- Sarathi-Serve: Chunked Prefill (OSDI 2024)
- SGLang / RadixAttention (2024)
- DistServe / Splitwise: PD 分离 (2024)

**量化**
- GPTQ (2023) / AWQ (MLSys 2024) / SmoothQuant (ICML 2023)
- FP8 Formats for Deep Learning (NVIDIA, 2022)

**模型结构**
- GQA (EMNLP 2023)
- DeepSeek-V2 (MLA) / DeepSeek-V3 (MTP + MoE)
- Mixtral of Experts (2024)

**投机解码**
- Fast Speculative Decoding (Leviathan et al., 2023)
- Medusa (2023) / EAGLE (2024)

**并行**
- Megatron-LM: TP (2019) / GPipe、PipeDream: PP

## 附录 C · 面试前自查清单

- [ ] 能用一句话说清四层框架，并各举 2–3 个技术
- [ ] 能画图解释 Prefill / Decode 的区别与各自瓶颈
- [ ] 能默写 KV Cache 大小公式并举例
- [ ] 能说清 FlashAttention 省的是 IO 不是 FLOPs
- [ ] 能说清 PagedAttention 解决的碎片问题与 Block Table 机制
- [ ] 能解释 CUDA Graph 的收益与代价（静态形状、额外显存）
- [ ] 能说出 Continuous Batching / Chunked Prefill / Prefix Cache 各自解决什么、代价是什么
- [ ] 能对比 GPTQ / AWQ / SmoothQuant / FP8
- [ ] 能对比 MHA / GQA / MQA / MLA 的 KV Cache 与质量
- [ ] 能说清 MoE 省计算不省显存，以及 All-to-All 通信难点
- [ ] 能说出投机解码五类 Draft 来源与统一流程
- [ ] 能解释 PD 分离的动机与代价
- [ ] 能说出 TP / PP / EP 的通信模式与适用场景
- [ ] 能说明 GPU/NPU 软件栈、版本配套与 CUDA Kernel 迁移路径
- [ ] 能用 reference 和中间张量找到模型精度的第一处分歧
- [ ] 能分别解释跑不起来、结果错、性能差、偶发故障的定位步骤
- [ ] 能拿出功能、精度、性能、稳定性与 Shape 覆盖的适配验收证据
- [ ] 对每个技术都能主动说出「收益 + 代价 + 适用边界」

---

> **最后一句**：面试时最好的状态不是「我知道很多技术」，而是「**我能把问题定位到瓶颈，再选技术，并说清代价与边界**」。这份文档的所有表格与配图，本质都是在训练这套判断力。
