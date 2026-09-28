# 3.2 最小 Triton 算子：行 Softmax

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

---
