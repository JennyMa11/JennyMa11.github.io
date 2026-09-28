# 12.3 Pipeline Parallel：层切分、微批与气泡

PP 把连续层放在不同 stage，传递 hidden states 而非每层集合通信。相同层数不代表相同工作量：Embedding / LM Head、MoE、设备速度和请求长度影响 stage 平衡。

在等耗时 p 个 stage、m 个微批的理想单次前向流水线中，利用率近似 `m/(m+p−1)`；这是教学上界模型，不是自回归服务的直接吞吐预测。下一 token 依赖上一 token 输出，需多个独立请求填充流水线。记录各 stage 空闲、P2P 时间和排队，若最慢 stage 主导，继续增加并发只会推高尾延迟。
