# 11.5 vLLM 投机解码实战：配置与接受率实测方法

### 11.5.1 统一用 speculative_config，先确认版本

下面命令按 **2026-09-27** 查阅的官方文档整理，是需要匹配 GPU / checkpoint 的部署模板，本次没有 GPU 实测数据。先记录 vLLM / CUDA / GPU、Target / Draft revision、采样、输入输出长度与并发，并检查本机 `vllm serve --help`。新配置入口是 JSON `--speculative-config`，旧的独立参数可能已弃用。[vLLM 配置入口](https://docs.vllm.ai/en/stable/features/speculative_decoding/)

```bash
python -c 'import vllm; print(vllm.__version__)'
vllm serve --help

# 基线：按本机显存调整模型、长度和并行度。
vllm serve /models/target --max-model-len 8192

# 无模型 N-gram：匹配窗口 2–5，最多提议 4 token。
vllm serve /models/target --max-model-len 8192 \
  --speculative-config '{"method":"ngram","num_speculative_tokens":4,"prompt_lookup_min":2,"prompt_lookup_max":5}' \
  --per-request-spec-decode-metrics detailed

# Suffix：此例关闭跨请求全局缓存，便于隔离实验。
vllm serve /models/target --max-model-len 8192 \
  --speculative-config '{"method":"suffix","num_speculative_tokens":4,"suffix_decoding_max_tree_depth":24,"suffix_decoding_max_cached_requests":0,"suffix_decoding_max_spec_factor":1.0,"suffix_decoding_min_token_prob":0.1}' \
  --per-request-spec-decode-metrics detailed

# 独立 Draft：替换为词表、上下文与后端都兼容的小模型。
vllm serve /models/target --max-model-len 8192 \
  --speculative-config '{"method":"draft_model","model":"/models/compatible-draft","num_speculative_tokens":4,"draft_tensor_parallel_size":1}' \
  --per-request-spec-decode-metrics detailed

# EAGLE-3：必须使用针对该 Target 训练的辅助 checkpoint。
vllm serve /models/target --max-model-len 8192 \
  --speculative-config '{"method":"eagle3","model":"/models/matched-eagle3","num_speculative_tokens":4}' \
  --per-request-spec-decode-metrics detailed
```

各方案分开启动，使用相同 Target 和 GPU 资源；辅助 checkpoint 的支持、可用 token 长度与 TP 方式需要检查日志。原版 EAGLE 的配置可用 `method="eagle"`，但这不证明任意动态树 / EAGLE-2 策略都已由该版本实现。Medusa 和其他方案也需核对当前支持，不能凭论文存在就假设有通用开关。[EAGLE 使用说明](https://docs.vllm.ai/en/stable/features/speculative_decoding/eagle/)

`temperature`、`top_p` 放在请求采样参数中；主模型 TP 用顶层参数，Draft TP 使用 `draft_tensor_parallel_size`。不要使用 synthetic acceptance 产生数据后把它当成真实接受率。

### 11.5.2 接受率至少需要三个统计口径

官方逐请求指标可由 `--per-request-spec-decode-metrics summary/detailed` 启用，单序列响应中读取 `metrics.speculative_decoding`；它是实验接口，旧版本可能没有。`detailed` 还提供每轮提议 / 接受数。服务聚合指标可查看 `/metrics`。[逐请求指标说明](https://docs.vllm.ai/en/stable/features/speculative_decoding/acceptance_metrics/)

```text
S = num_spec_steps                  # 有记录的验证步骤
D = num_draft_tokens                # 实际提议 token 数
A = num_accepted_draft_tokens       # 接受的 Draft token 数，不含 bonus

Draft 接受率 = A / D               # D=0 时未定义，不能填 100%
平均接受 Draft 数 = A / S
平均每轮输出（指标口径） = 1 + A / S
```

`mean_acceptance_length` 含修正 / bonus，而 `acceptance_histogram[j]` 统计接受 j 个 Draft 的轮数，不含 bonus。这个 `A/D` 也不等于 11.4 简化模型中的恒定条件接受率 α：拒绝后的候选没有走到同样的条件前缀，需结合位置存活概率理解。EOS / 输出截断下，实际交付 token 数还应以响应为准。

跨请求先求 `ΣA、ΣD、ΣS` 再计算总体比例，不要不加权地平均每请求接受率。Lookup 候选覆盖率、每个位置的 `Pr(K≥i)` 和接受长度直方图一起记录，才能区分“很少猜”与“经常猜错”。

### 11.5.3 可运行客户端：分别测代码与开放对话

仓库提供 [实测客户端](examples/speculative_benchmark.py)、[代码输入](examples/speculative_code.jsonl) 和 [对话输入](examples/speculative_chat.jsonl)。客户端只向已运行的本地 vLLM 发请求，保存原始计数与响应时间；小型示例用于检查流程，正式结论需要更大、覆盖业务的独立数据集。

```bash
# 在仓库根目录运行；每切换一个服务配置，重复相同数据集。
python3 infra/examples/speculative_benchmark.py \
  --dataset infra/examples/speculative_code.jsonl \
  --model /models/target --label ngram-code \
  --output /tmp/ngram-code.jsonl

python3 infra/examples/speculative_benchmark.py \
  --dataset infra/examples/speculative_chat.jsonl \
  --model /models/target --label ngram-chat \
  --output /tmp/ngram-chat.jsonl
```

先预热，再分别运行无投机、N-gram、Suffix、Draft、EAGLE；γ 扫描 `2/4/8`。先用相同 `temperature=0` 验证贪心行为，再在相同温度 / top-p 下评估随机采样。客户端是顺序、非流式测试，记录的是整请求时间和有效输出速率，**不把它称为 TTFT / TPOT 或高并发 serving 吞吐**；并发实验使用负载工具和服务日志单独记录 P95/P99。

| 数据集 | 方案 / γ | A/D | 1+A/S | 请求时间 / 输出速率 | TPOT / P99（另测） | 质量 |
|---|---|---|---|---|---|---|
| 代码 | 基线 / N-gram / Suffix / Draft / EAGLE | 待测 | 待测 | 待测 | 待测 | 单元测试 / pass rate |
| 开放对话 | 相同配置逐组比较 | 待测 | 待测 | 待测 | 待测 | 业务评价 / 盲评 |

这些格子保持“待测”，不是预设代码高于对话。固定 Prompt / chat template、长度分布、采样、设备与负载；Suffix 全局缓存如要启用，应按相同顺序预热。Draft / Verify 耗时需要服务端 tracing 或 profiler，客户端端到端时间不能直接拆出它们。

**验收问题**：输出分布正确吗；哪些任务有有效候选；每轮有效输出能否覆盖额外成本；高并发时 SLO 是否仍满足。用这四项决定是否部署。

---
