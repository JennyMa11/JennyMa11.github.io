# 15.2 Benchmark 工具：vllm bench、GenAI-Perf 与实验配置

| 工具 / 模式 | 回答的问题 | 不能替代什么 |
|---|---|---|
| `vllm bench latency` | 固定 batch 的模型路径延迟 | 在线排队与用户尾延迟 |
| `vllm bench throughput` | 离线处理能力 | 给定到达率下的 Goodput |
| `vllm bench serve` | 在线流式延迟、吞吐与负载 | 正式业务质量验收 |
| Nsight / profiler | CPU、GPU、通信热点归因 | 无采集开销的正式计时 |

**开环负载**按预定到达率发请求，能暴露队列增长；**闭环负载**维持固定并发，完成后才补请求，响应变慢时到达率也下降，容易掩盖过载。若工具同时限制 QPS 与 max concurrency，实际到达率可能低于设定值。[vLLM bench serve 参数](https://docs.vllm.ai/en/stable/cli/bench/serve/)

```bash
# 服务按 9.1 启动；先核对安装版本支持的参数。
vllm bench serve --help
vllm bench serve --backend vllm \
  --base-url http://127.0.0.1:8000 --endpoint /v1/completions \
  --model infra-demo --tokenizer /models/model \
  --dataset-name random --random-input-len 1024 --random-output-len 128 \
  --num-prompts 200 --request-rate 4 --save-result --save-detailed
```

这是形状基准的模板，随机 token 不代表业务语义或量化质量。分别做业务请求回放；将 input/output 长度、到达率、Prefix 冷 / 热状态与取消行为固定。保存错误记录、原始每请求时间、实际输出长度、模型 / tokenizer revision、引擎配置与环境。

更换压测工具时核对请求协议、chat template、token 计数、流式响应与速率模型。GenAI-Perf 的统计口径也应写进实验配置。查阅时 NVIDIA 官方已说明 GenAI-Perf 逐步退出并推荐新需求使用 AIPerf；保留它用于理解参考大纲及历史报告，新实验先核对当前工具路线。[官方工具说明](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/perf_analyzer/genai-perf/README.html)

多个工具得到不同数字时，先比较负载和时间边界，再比较引擎。
