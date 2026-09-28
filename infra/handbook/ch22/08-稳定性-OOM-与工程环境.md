# 22.8 稳定性、OOM 与工程环境

推理内存按**权重 + KV Cache + 激活 + Graph/编译缓存 + 临时 Workspace + 分配器碎片**列账。并发升高才 OOM，先看 `max_num_seqs`、上下文上限、KV 块总数、Prefix Cache 与抢占；显存持续单向增长，检查张量/Block 生命周期和取消、异常路径。抢占可选择释放后重算、利用仍在缓存中的前缀，或把状态换出到主机内存；后者节省设备内存但增加传输与状态管理。每轮可核对 `free + occupied + cached = total`（按实现定义避免重复计数），结束后检查没有孤儿引用计数。

长时压测同时覆盖请求取消、超时、Prefix 共享、块边界、动态 Batch、长上下文和多卡异常传播。监控 NaN/Inf、OOM、crash、内存峰值/趋势、抢占次数、恢复成本、P99 TTFT/TPOT；对可复现故障保存最小输入与版本矩阵。

**稳定性分诊**：分别记录进程 RSS/CPU 内存与设备已分配、已保留显存；持续增长要先区分真实张量或 KV 引用泄漏、分配器缓存与碎片化。CUDA OOM 看申请大小、剩余/保留量及失败前的请求生命周期；Illegal Memory Access 用最小输入和内存检查工具定位 Kernel；NCCL/HCCL Timeout 或死锁先比较各 rank 的最后成功 collective 与更早异常。Watchdog 报警、进程崩溃时保留首个错误日志、栈跟踪和可用的 core dump，并记录故障前后设备重置/进程退出事件。超时是症状，需进一步区分队列堆积、设备卡死和通信等待。

Linux/Docker 是复现环境的基本工具：记录镜像、容器启动参数、Volume、设备映射、端口和环境变量；用 `docker ps/logs/inspect/exec` 对照进程、日志与设备可见性。容器共享 Host Kernel，设备驱动通常依赖宿主机；镜像内的软件栈仍要与宿主驱动/固件和目标设备匹配。先在相同容器里跑最小 MatMul，再比较模型差异。
