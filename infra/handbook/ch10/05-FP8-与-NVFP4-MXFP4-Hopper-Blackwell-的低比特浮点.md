# 10.5 FP8 与 NVFP4 / MXFP4：Hopper、Blackwell 的低比特浮点

### 10.5.1 FP8 不是不需要 scale 的 INT8 替代品

E4M3 使用 1 符号、4 指数、3 尾数位；E5M2 为 1+5+2。常见 E4M3 格式最大有限值约 `448`，E5M2 约 `57344`。前者精度更细，后者动态范围更大；具体特殊值规则要按格式定义检查。

```text
x_fp8 = cast_fp8(x / s)
x̂ = cast_float(x_fp8) × s
# 误差与值的指数区间有关，不是固定间距的整数网格。
```

FP8 仍会受 Outlier、上溢、下溢与 scale 粒度影响。Per-tensor、per-token 或 block scale 如何选，要结合输入分布与硬件路径。累加常用更高精度，但内部累加、分段提升和输出 dtype 都需检查 Kernel，不能只看输入格式。

Hopper 支持原生 FP8 Tensor Core，Ada 也有 FP8 支持；不能把「FP8 始于 Hopper」或「老 GPU 只支持 INT8」当成通用规则。没有原生算术支持的设备也可能读取 FP8 存储后转换计算，性能含义不同。[vLLM 硬件支持表](https://docs.vllm.ai/en/stable/features/quantization/)

### 10.5.2 MXFP4 与 NVFP4：同为 E2M1，Scale 设计不同

4 bit 浮点通常用 E2M1：1 符号、2 指数、1 尾数位，非负数值集合为 `{0,0.5,1,1.5,2,3,4,6}`。动态范围依赖共享 scale；它与 INT4 的均匀整数网格不同。

| 格式 | 元素编码 | Block 大小 | Block scale | 额外 scale |
|---|---|---|---|---|
| FP8 | E4M3 / E5M2 | 由方案决定 | 常用浮点 scale | 由方案决定 |
| MXFP4 | E2M1 | 32 个元素 | E8M0，表示 2 的幂 | 按具体实现检查 |
| NVFP4 | E2M1 | 16 个元素 | E4M3，可表示更细的 scale | FP32 per-tensor scale |

```text
MXFP4：x̂ = fp4_value × block_scale
NVFP4：x̂ = fp4_value × block_scale × tensor_scale
```

NVFP4 更小的 block 能限制 Outlier 对邻近值的影响，E4M3 scale 能比仅 2 的幂更细地拟合范围；代价是更多 metadata 与两级 scale。Blackwell 的原生低比特 Tensor Core 路径可消费相应 block scale；实际模型层、GPU SKU 和后端支持仍需核对。[NVIDIA NVFP4 格式说明](https://developer.nvidia.com/blog/introducing-nvfp4-for-efficient-and-accurate-low-precision-inference/)

只计元素和 block scale：MXFP4 有效 `4+8/32=4.25 bits/element`，NVFP4 为 `4+8/16=4.5`，后者还需摊销 tensor scale。更细 scale 可以提升拟合能力，却不保证任意模型 / 任务质量更高；应在相同 workload 下比较。

**关键区分**：W4A16 的 INT4 经常解包后做浮点计算；原生 FP4 的 W4A4 路径把两侧低比特操作数及 block scale 一起交给硬件。也有 FP4 Weight-only 或 emulation 后端，checkpoint 是 FP4 不代表激活一定 FP4，更不代表已经使用原生 FP4 MMA。

### 10.5.3 编码格式速查

| 格式 | 位宽与表示 | 典型作用 |
|---|---|---|
| FP32 | 1+8+23，32 bit | 参考计算、常见累加 |
| TF32 | 运算有效精度 1+8+10，张量通常仍按 FP32 存储 | Tensor Core 计算模式，不能按 19 bit 算存储压缩 |
| FP16 | 1+5+10，16 bit | 精度较细、范围较小 |
| BF16 | 1+8+7，16 bit | 指数范围大，常用 LLM 基线 |
| FP8 | E4M3 / E5M2，8 bit + scale | W/A 或 KV 压缩；算术路径独立检查 |
| INT8 / INT4 | 8 / 4 bit 整数编码 + scale / 可选 zero-point | W8A8、Weight-only 或 KV，需声明对象 |
| MXFP4 / NVFP4 | E2M1，4 bit + block scale | 微缩放浮点，需要匹配的布局与 Kernel |

NF4 是针对近似正态分布设计的非均匀 4 bit codebook，常见于 QLoRA；它与 INT4、E2M1 FP4 不是同一种格式。
