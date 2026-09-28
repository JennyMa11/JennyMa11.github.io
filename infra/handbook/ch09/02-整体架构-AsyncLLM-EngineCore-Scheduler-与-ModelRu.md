# 9.2 整体架构：AsyncLLM、EngineCore、Scheduler 与 ModelRunner

```text
HTTP / chat template / tokenize → AsyncLLM 前端
  → EngineCore → Scheduler ↔ KVCacheManager
  → Worker → ModelRunner → Attention / Linear backend → Sampler
  → EngineCore 更新进度 → 前端 detokenize / stream
```

| 组件 | 责任 | 追踪的记录 |
|---|---|---|
| AsyncLLM / API 前端 | 输入输出与异步请求生命周期 | request ID、取消、流式响应 |
| EngineCore | 驱动调度和执行循环 | 轮次、消息传输、结束原因 |
| Scheduler | 生成本轮 token 计划 | scheduled tokens、抢占和等待 |
| KVCacheManager | 分配、命中、引用和回收 | block IDs、缓存 token、失败分配 |
| Worker / ModelRunner | rank 执行与设备输入组织 | positions、slot mapping、persistent batch |

这些是角色划分，不是保证各版本类名完全相同。固定 commit 后记录入口位置；CPU 队列延迟、GPU 执行时间、网络发送时间分开计时。
