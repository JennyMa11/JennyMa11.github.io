# 6.5 KV Cache 管理与 Offload

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
