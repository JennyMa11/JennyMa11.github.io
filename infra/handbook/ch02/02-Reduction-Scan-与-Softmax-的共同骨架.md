# 2.2 Reduction、Scan 与 Softmax 的共同骨架

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
