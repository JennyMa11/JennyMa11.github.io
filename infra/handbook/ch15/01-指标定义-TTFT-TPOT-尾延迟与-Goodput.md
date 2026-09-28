# 15.1 指标定义：TTFT、TPOT、尾延迟与 Goodput

流式请求设发送时间 `t0`、首个有效输出 token 时间 `t1`、结束时间 `tN`、输出 token 数 N：

```text
TTFT = t1 − t0                         # 包含网络、排队、预处理与首段计算
TPOT = (tN − t1)/(N−1), N>1           # N=1 时未定义
TBT_i = t_i − t_(i−1)                  # 看最差停顿与分布
E2E ≈ TTFT + (N−1)×TPOT
```

网络 chunk 不一定等于单个模型 token；一次 chunk 含多 token 或空增量时，不可直接把 chunk 间隔称为真实 token 间隔。报告测量层（引擎 / 客户端）、token 计数器及是否包含 detokenization。

**Goodput**：在观测窗口内，成功完成且满足指定 SLO 的请求数 / 窗口秒数。若 SLO 同时要求 TTFT 与 TPOT，则同一请求必须同时满足；分别合格的 P95 TTFT / P95 TPOT 不能证明联合合格。超时、拒绝、取消和失败数量要单独报告，不能从总请求分母悄悄删除。

教学例：60 秒发来 600 个请求，540 个成功、420 个同时满足 SLO，则完成 QPS 为 9，Goodput 为 7，SLO 合格占全部到达请求的 70%。这些指标刻画不同事实；稳定负载可用 `平均在途数≈到达率×平均驻留时间` 检查统计一致性。
