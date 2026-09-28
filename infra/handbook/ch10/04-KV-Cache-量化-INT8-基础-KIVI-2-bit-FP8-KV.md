# 10.4 KV Cache 量化：INT8 基础、KIVI 2-bit、FP8 KV

### 10.4.1 为什么长上下文需要单独压缩 KV

常规 Attention 每层有 `K,V ∈ R^[B,Hkv,S,dh]`，随着长度 `S` 增长，状态持续增加；Decode 每步还需要读取参与 Attention 的历史 KV。假设所有层 shape 相同：

```text
M_KV = 2 × L × B × Hkv × S × dh × bytes_per_element
# 2 表示 K 和 V；真实分页实现还包含 metadata、padding 和预留块。
```

教学例：`L=32、B=1、Hkv=8、S=32768、dh=128`，FP16/BF16 KV payload 为 `4 GiB`，INT8 / FP8 约 `2 GiB`，2 bit 理想 payload 约 `0.5 GiB`。实际还要加 scale、zero-point、残留高精度窗口。GQA 用的是 `Hkv`，不能误用 Query head 数；MLA 需按实际缓存表示另算。

### 10.4.2 写入、读取与融合反量化

以对称 INT8 为例，分别为 K、V 保存 scale：

```text
写入：Kq=clip(round(K/sK), -127,127)，Vq=clip(round(V/sV), -127,127)
缓存：Kq、Vq、sK、sV；非对称方案还需要 zK、zV
读取：K̂=sK×Kq，V̂=sV×Vq
Attention：P=softmax(QK̂ᵀ/√dh + mask)，O=PV̂
```

K 的误差改变 logits 和 softmax 权重，V 的误差改变加权和；不应只比较 K/V 重构 MSE 就宣布 Attention 质量足够。

```text
低效路径：读 INT8 KV → 独立 Dequant Kernel → 写 FP16 KV 到 HBM
                         → Attention 再读 FP16 KV
高效路径：读压缩 KV + metadata → 寄存器 / 片上 tile 中还原
                              → 直接参与 QK / PV、online softmax
```

这就是 **Fused Dequantization + Attention**。融合省去浮点 Cache 的物化，但仍有解包、scale 乘法和寄存器开销。Prefill、Decode、Prefix 命中后的路径都要检查，不能只证明某个 Decode Kernel 更快。

### 10.4.3 Scale 粒度与 Metadata

下表固定 `B、Hkv` 后看 `[S,dh]` 切片；Per-head scale 通常仍按层区分：

| 粒度 | scale 覆盖范围 | 在线写入的取舍 |
|---|---|---|
| Per-tensor | 整层 K 或 V 共一个 | 元数据最少；动态范围漂移要处理 |
| Per-head | 每个 KV head 一个 | 隔离 head 范围；可使用离线校准 scale |
| Per-token | 每个 token 沿 `dh` 一个 | 新 token 可独立量化并追加 |
| Per-channel | 每个 channel 沿 token 轴统计 | 跨 token 统计；不能无限改旧 scale 而不更新旧编码 |
| Group-wise | 沿 token 或 channel 轴固定长度分组 | 明确分组轴、尾组与分页 block 的对应关系 |

`dh=128` 且每个 `(token,head)` 的 K、V 各存一个 FP16 scale：FP16 payload 为 `512 bytes`，INT8 payload 为 `256 bytes`，scale 共 `4 bytes`，压缩约 `512/260≈1.97×`。粒度更细会增加 metadata；有 zero-point、padding 时需重新计算。

### 10.4.4 KIVI：K 与 V 使用不同的量化轴

KIVI 提出非对称低比特 KV 量化，并研究 2 bit 设置：**K 按 channel，V 按 token**。论文观察到部分模型 K 的 Outlier 集中在固定 channel；V 对输出的影响又与注意力选择的 token 相关，所以两者适合不同分组轴。[KIVI 论文](https://arxiv.org/abs/2402.02750)

```text
固定某个 KV head 的 [S,dh]：
K：每 G 个 token 为一组，对每个 channel 单独统计 / 量化
V：每个 token 内，沿 channel 分组统计 / 量化
```

K 的 channel 统计需要跨 token 收集，无法每来一个 token 就任意修改已有编码的 scale。KIVI 用分组量化与高精度 residual cache 处理流式追加，保留近期未量化状态，再批量压缩旧状态。它的收益包含这部分存储和操作成本，不能直接宣称真实缓存缩小到 FP16 的 `1/8`。[KIVI 的流式分组与 residual cache](https://arxiv.org/html/2402.02750v2#S3)

**2 bit 不是通用安全值**：长期检索、数学推理和不同模型的质量都要验证；论文方案有专用布局和 Kernel，不能凭一个 `--kv-cache-dtype` 参数假设任意 vLLM 版本已经实现 KIVI。

### 10.4.5 FP8 KV：存储类型与计算类型分别检查

FP8 KV 存储 8 bit 浮点编码，常见 E4M3 / E5M2，仍需要 scale 管理；不是 INT8 的 `round(K/s)` 整数网格。典型路径为 `cast_fp8(K/sK)` 写入，再 `cast_float(Kfp8)×sK` 参与计算。

在 vLLM 中，权重量化与 `kv_cache_dtype` 分别配置。官方文档区分 per-tensor 和 per-head scale；具体支持取决于 Attention 后端。有的路径片上还原，有的后端能直接在 FP8 域计算 Attention，例如文档描述的 FlashAttention 3 路径还会量化 Q。[vLLM FP8 KV 文档](https://docs.vllm.ai/en/stable/features/quantization/quantized_kvcache/)

**收益怎么报告**：固定请求集合时报告缓存字节下降；框架若用剩余显存预分配 KV pool，则 GPU 总占用可能接近原来，增加的是 token 容量 / 可并发请求。容量增加、Attention 延迟下降与端到端吞吐提升是三项不同指标。
