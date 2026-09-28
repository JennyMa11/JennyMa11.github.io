# 1.1 从 CPU 内存层次过渡到 GPU

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
