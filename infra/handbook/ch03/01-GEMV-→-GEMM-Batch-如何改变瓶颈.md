# 3.1 GEMV → GEMM：Batch 如何改变瓶颈

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
