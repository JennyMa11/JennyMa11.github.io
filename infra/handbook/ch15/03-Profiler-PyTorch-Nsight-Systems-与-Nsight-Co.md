# 15.3 Profiler：PyTorch、Nsight Systems 与 Nsight Compute

### 先分类慢，再看指标

同一模型至少做三组实验：固定 Batch 改序列长度、固定长度改 Batch、固定总 token 改请求数。先把总时间拆成排队、CPU 准备、H2D、GPU Kernel、通信、采样、回传。Nsight Systems 看时间线和空隙；Nsight Compute 深挖一个确定的 Kernel；`torch.profiler` 看 PyTorch op 与 Kernel 对应关系。`nvprof` 属旧工具，现代 GPU 优先用 Nsight；CUPTI 适合构建自动采集器。

**Roofline**：`Arithmetic Intensity = FLOPs / DRAM bytes`，理论上界 `min(峰值 FLOP/s, 强度 × 峰值带宽)`。测量时用实际 DRAM 字节和实际耗时，得到 achieved FLOP/s 与 achieved bandwidth；不能拿模型参数量直接代替所有 Kernel 的实际流量。低强度且 DRAM 接近带宽上限是 memory-bound 的证据；强度高但 Tensor Core 没工作，可能是 dtype/layout/alignment 或 fallback；两者都低，要检查 launch、同步、访存延迟或并行度。

| 指标 | 看见什么 | 下一步 |
|---|---|---|
| SM active / GPU busy | 有多少时间在执行 GPU 工作 | 低则先看 CPU 空隙、短 Kernel、batch 太小 |
| Tensor Core utilization / 指令数 | 是否走 MMA 路径 | 查 dtype、维度对齐、layout、编译产物 |
| DRAM throughput、L2 hit | 流量与缓存复用 | 区分带宽饱和和高延迟散读 |
| Occupancy、register/thread、shared/CTA | 可驻留 Warp 与资源限制 | 查 spill、tile 大小、Block 数 |
| Eligible warps / issue active | 调度器是否有可发射工作 | 再看 stall；不要只按单一 stall 百分比下结论 |

### Warp Stall 速查与动作

| Stall | 通常代表 | 定位与优化方向 |
|---|---|---|
| Long Scoreboard | 等待 L1TEX/global/local load 的依赖 | 找产生依赖的 load；看 coalescing、L2/DRAM、预取、增加独立工作或减少 spill |
| Short Scoreboard | 等待 shared/MIO 类操作的依赖 | 看 shared 访问、bank conflict、依赖链、特殊函数 |
| MIO Throttle | MIO 指令队列拥塞 | 查 shared 操作数、散乱地址、指令宽度与局部性 |
| Memory Throttle（或具体的 LG Throttle 等） | 访存指令的某一级队列/通路不能继续接收请求；名称依工具和架构而变 | 对照该指标的官方定义，查每条访存指令的事务数、合并情况与对应管线压力；不等同于 DRAM 带宽已满 |
| Barrier | 等待 CTA 屏障及线程进度 | 减少不必要屏障、平衡 Warp 工作量 |
| Not Selected | Warp 已准备好但本周期未被选中 | 这是调度状态，通常说明可发射 Warp 足够；无需直接“消除” |
| Instruction Fetch / No Instructions | 指令未就绪或 I-cache/短 Grid 等因素 | 看代码体积、分支、Grid 是否足够大 |

Stall 名称和计数器会随架构/工具版本变化；优先读 [Nsight Compute Profiling Guide 的 Warp Stall 解释](https://docs.nvidia.com/nsight-compute/ProfilingGuide/)。`Long Scoreboard` 高并不能单独证明 DRAM 带宽打满；它也可能是访问延迟或 local memory spill。

### 案例：自定义 GEMM 只有 cuBLAS 的 72%

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
