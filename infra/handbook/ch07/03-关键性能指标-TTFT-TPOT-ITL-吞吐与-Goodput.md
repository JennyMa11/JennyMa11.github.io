# 7.3 关键性能指标：TTFT、TPOT / ITL、吞吐与 Goodput

以客户端收到首个输出为 TTFT 终点；以相邻流式 token 到达时间衡量 ITL，TPOT 是首 token 后的平均间隔。网络 chunk 未必一一对应 token，统计时说明观测点。吞吐分别记录 request/s、input token/s、output token/s；Goodput 只计满足约定 SLO 的成功请求。

```text
对输出 N ≥ 2 个 token 的请求：
TPOT = (最后 token 时间 − 首 token 时间) / (N − 1)
生成延迟 ≈ TTFT + (N − 1) × TPOT
Goodput = 同时满足 TTFT / TPOT 等 SLO 的成功请求数 / 测量时间
```

单 token 输出不定义上述 TPOT。报告 P50 / P95 / P99 与输入输出长度分布；平均 TPOT 可能掩盖单次长停顿。完整指标、混合负载和压测方法见[第 15 章](ch15.html)。
