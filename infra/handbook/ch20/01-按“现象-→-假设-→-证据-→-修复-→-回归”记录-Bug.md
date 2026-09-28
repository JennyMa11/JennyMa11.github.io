# 20.1 按“现象 → 假设 → 证据 → 修复 → 回归”记录 Bug

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

---
