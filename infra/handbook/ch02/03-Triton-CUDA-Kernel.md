# 2.3 Triton / CUDA Kernel

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

---
