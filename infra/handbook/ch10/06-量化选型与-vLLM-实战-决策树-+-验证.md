# 10.6 量化选型与 vLLM 实战：决策树 + 验证

### 10.6.1 先问哪一部分在限制你

```text
建立 BF16/FP16 reference，固定 workload / SLO / 质量预算
│
├─ 权重装不下，或短 Decode 读权重占主导
│    → 先比较 GPTQ / AWQ W4A16 + 受支持的融合 Kernel
│
├─ 大 Prefill / 高并发 GEMM 计算占主导
│    → 比较 INT8 W8A8 / 原生 FP8
│    → Blackwell 上有匹配模型与后端时，再评估 NVFP4 / MXFP4
│
├─ 长上下文 / 高并发 KV 容量或读取占主导
│    → 独立启用受支持的 FP8 KV；更低 bit 需专用方案与长文本验证
│
└─ CPU 调度、通信、排队占主导
     → 量化未必改善主要瓶颈，先定位对应系统路径
```

多个分支可同时成立。权重量化腾出的显存可能用于更多 KV；增加 batch 又会改变 GEMM 的计算 / 带宽比例，需要重测。`4×` 权重压缩不是 `4×` 端到端加速；把转换、Attention、通信和调度都纳入计时。

### 10.6.2 vLLM 启动示例：先读 checkpoint，再选后端

以下是**命令模板**，本次文档更新没有在 GPU 上实跑。把路径替换为已准备且兼容的 checkpoint；GPU、vLLM 版本、模型 revision、量化配置与 Attention 后端要记入实验记录。支持信息核对日期为 **2026-09-27**，`stable` 文档会变化。[vLLM 量化入口](https://docs.vllm.ai/en/stable/features/quantization/)

```bash
# 记录实际环境，并查看本机版本可用的量化 / KV 参数。
python -c 'import vllm; print(vllm.__version__)'
nvidia-smi
vllm serve --help

# BF16 reference；按显存与模型能力调整长度。
vllm serve /models/model-bf16 --dtype bfloat16 --max-model-len 8192

# 已量化 checkpoint：由 quantization_config 自动识别适配的加载路径。
vllm serve /models/model-awq-int4 --dtype half --max-model-len 8192
vllm serve /models/model-gptq-int4 --dtype half --max-model-len 8192

# 独立启用 FP8 KV；checkpoint 应含匹配格式的校准 K/V scales。
vllm serve /models/model-with-fp8-kv-scales \
  --kv-cache-dtype fp8_e4m3 --max-model-len 8192

# 预量化 FP8 / NVFP4 / MXFP4 checkpoint：同样先由模型配置识别。
vllm serve /models/model-fp8 --max-model-len 8192
vllm serve /models/model-nvfp4 --max-model-len 8192
vllm serve /models/model-mxfp4 --max-model-len 8192
```

这些示例是分别启动的实验，不能同时抢占同一张卡 / 默认端口。`--dtype half` 配置浮点激活等部分，不会把 packed INT4 权重改回完整 FP16。对于传统 AWQ checkpoint，可按本机支持显式指定 `--quantization awq`；GPTQ、Marlin 或其他后端强制参数必须匹配 checkpoint，优先检查自动选择的日志。[AWQ 使用说明](https://docs.vllm.ai/en/stable/features/quantization/auto_awq/)、[GPTQModel 使用说明](https://docs.vllm.ai/en/stable/features/quantization/gptqmodel/)

**指定量化方法不等于完成离线量化**：GPTQ / AWQ checkpoint 需要相应校准工具生成。FP8 等在线量化能力要按具体版本与后端检查，不能把原始 BF16 路径加一个任意 `--quantization` 名称就视为准备好了模型。

KV scale 也不能忽略：优先用代表性数据离线校准并随 checkpoint 保存；缺失 scale 时，一些路径默认 `1.0`，必须验证是否合适。当前 stable 文档的校准示例使用 `llm-compressor`；旧教程中的 `--calculate-kv-scales` 在不同版本存在变更，不应直接照搬。[FP8 KV 配置与校准](https://docs.vllm.ai/en/stable/features/quantization/quantized_kvcache/)

### 10.6.3 启动后看什么，怎样证明有效

| 检查 | 记录 / 比较内容 | 解决的问题 |
|---|---|---|
| 加载 | bits、group size、zero-point、排除层、quantization_config | 是否真加载了目标量化权重 |
| 执行 | 日志选择的 Linear / MoE 与 Attention Kernel；必要时 profiler | 有没有回退、物化反量化或不支持的层 |
| 数值 | 逐层输出 / logits 的 max、mean error、NaN/Inf；scale 与统计轴 | 第一处分歧是否来自配置 / 实现错误 |
| 质量 | PPL、业务集；中文 / 代码 / 数学、长上下文检索 | 校准域之外是否退化 |
| 性能 | 相同输入 / 输出长度与负载下的 TTFT、TPOT、吞吐、P95/P99 | 对真实 SLO 是否有收益 |
| 容量 | 固定请求的权重 / KV 字节；固定显存预算的 KV token 容量 | 区分节省字节与增加预分配容量 |

实验先单独变权重精度，再单独变 KV，最后组合，保留相同 tokenizer、chat template、采样规则和最大输出长度。先固定并发比较单位请求，再扫描并发找容量收益；预热不计入正式测量。量化质量用任务指标判断，不能用某个固定 cosine 数值通吃所有模型。

```python
def quant_check(ref, test, atol, rtol):
    diff = (ref.float() - test.float()).abs()
    return {
        "ok": torch.allclose(ref.float(), test.float(), atol=atol, rtol=rtol),
        "max_abs": diff.max().item(),
        "mean_abs": diff.mean().item(),
        "finite": torch.isfinite(test).all().item(),
    }
```

**面试练习**：从 `W4A16 + FP8 KV` 出发，分别解释权重和 KV 的存储 dtype、激活 dtype、计算 / 累加 dtype、scale 粒度、反量化发生在哪、瓶颈变成什么。能说清这六项，才算把格式、算法和工程执行连起来。

---
