# 9.4 调度器源码：schedule → execute → update

先固定 Git commit/版本，再按 `API/Engine → Scheduler → KV Manager → ModelRunner → Attention Backend → Sampler → Output` 追一条 request_id。只摘核心控制流，类与函数名以链接版本为准。下面是**从真实接口抽象的 16 行导读伪代码**，用于标注数据流，不是项目逐字源码：

```python
request = engine.add_request(prompt, sampling_params)
while not request.finished:
    plan = scheduler.schedule()       # token 预算、KV 块与抢占
    inputs = runner.prepare(plan)     # packed tokens、position、slot mapping
    result = runner.execute(inputs)   # embedding → layers → attention backend
    outputs = sampler(result.logits, plan.sampling_metadata)
    scheduler.update_from_output(plan, outputs)
    engine.publish(outputs)           # streaming token / finish reason
```

| 项目 | 架构/关键类 | 首先跟的函数与问题 |
|---|---|
| [vLLM V1](https://github.com/vllm-project/vllm/tree/main/vllm/v1) | `Scheduler`、`KVCacheManager`、`GPUModelRunner` | `schedule` 如何确定 token 数；`execute_model` 如何把调度输出变 GPU 输入 |
| [SGLang](https://github.com/sgl-project/sglang/tree/main/python/sglang/srt) | Scheduler、RadixCache、ModelRunner | 前缀匹配结果如何进入新请求；树节点何时引用/淘汰 |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) | Executor、BatchManager、KV Cache Manager | in-flight batching 怎样混合 context/generation；KV transfer 何时提交 |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | `llama_context`、`ggml` graph、KV cache | token batch 如何构图和执行；GGUF 量化张量如何参与计算 |
| [PyTorch](https://github.com/pytorch/pytorch) / [Triton](https://github.com/triton-lang/triton) / [CUTLASS](https://github.com/NVIDIA/cutlass) | Dispatcher / compiler / GEMM tile | 算子从 Python API 到 GPU Kernel 的边界在哪里 |

**vLLM V1 的 14 行控制流缩写**（按当前 [Scheduler.schedule](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/sched/scheduler.py)、[KVCacheManager](https://github.com/vllm-project/vllm/blob/main/vllm/v1/core/kv_cache_manager.py)、[GPUModelRunner](https://github.com/vllm-project/vllm/blob/main/vllm/v1/worker/gpu_model_runner.py) 改写，保留关键状态而省略分支）：

```python
token_budget = scheduler.max_num_scheduled_tokens
for request in running_then_waiting:
    cached_blocks, cached_tokens = kv.get_computed_blocks(request)
    missing = request.num_tokens - request.num_computed_tokens - cached_tokens
    scheduled = min(missing, token_budget)
    if scheduled <= 0:
        continue
    new_blocks = kv.allocate_slots(request, scheduled,
                                   new_computed_blocks=cached_blocks)
    if new_blocks is None:
        handle_preemption_or_wait(request)
        continue
    plan[request.request_id] = (scheduled, new_blocks)
    token_budget -= scheduled
runner_output = model_runner.execute_model(plan)
scheduler.update_from_output(plan, runner_output)
```

阅读真实源码时，逐行把 `num_computed_tokens`、`num_tokens_with_spec`、Prefix 命中长度、`num_scheduled_tokens` 写进一个 3 请求表格；这能看出普通 Decode、Chunked Prefill、投机 token 如何由同一进度差表达。上面的 `running_then_waiting` 和 `handle_preemption_or_wait` 是教学抽象，不是实际标识符。

**源码记录模板**：记 `输入 shape/stride → 状态变更 → 分配/释放 → Kernel launch → 输出`。每次只读一条路径并做断点/日志验证；遇到版本差异以源码为准，不把教学伪代码当成正式实现。
