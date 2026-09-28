# 10.3 Weight-only INT4：GPTQ、AWQ、Marlin Kernel

GPTQ / AWQ 解决「怎样得到低误差权重」，Marlin 解决「怎样高效执行低比特权重」。算法、checkpoint 格式和执行 Kernel 必须分别检查。

### 10.3.1 GPTQ：二阶信息 + 顺序误差补偿

GPTQ 属于 **Weight-only PTQ**，并非独立地对每个元素做 `round(W/s)`；它用校准输入构造线性层重构目标，量化一列后补偿尚未量化的列，论文重点研究 3/4 bit 权重。[GPTQ 论文](https://arxiv.org/abs/2210.17323)

这里为对应实现改用 `A=W_lin ∈ R^[Cout,Cin]`，校准输入 `Z=Xᵀ ∈ R^[Cin,T]`，输出为 `AZ`：

```text
目标：min_Â ||AZ − ÂZ||_F²
H = 2ZZᵀ                         # 每个输出行共享的输入方向曲率
H_damped = H + λI                 # 校准不充分 / 相关输入时改善数值条件
```

这是**局部线性层重构损失的 Hessian**；相对于全模型任务损失，它提供近似二阶信息。不是去显式计算整个 LLM 参数空间的巨型 Hessian。

对某个输出行，当前还未量化的输入索引集合为 `F`，令 `G=(H_damped[F,F])^-1`。把第 `j` 个值从 `w_j` 固定到量化网格 `q_j`，约束二次目标的补偿解为：

```text
e_j = w_j − q_j
δ_F = − e_j × G[:,j] / G[j,j]
w_F ← w_F + δ_F
# δ_j = −e_j，当前值落到 q_j；其他尚未量化的值按相关性补偿。
```

`G` 的非对角项体现通道相关性：若两个输入方向高度相关，可以改另一个权重抵消当前误差；只用逐元素舍入看不到这种关系。每固定一个变量，下一步必须使用**剩余自由变量的逆 Hessian**，不能始终套用初始完整逆矩阵。[GPTQ 的二阶更新与分块实现](https://arxiv.org/html/2210.17323v2#S4)

```text
Calibration Data → 统计 H → damping / factorization
  → Quantize 第 1 列 → 计算 Error → 修改剩余列
  → Quantize 已被修改的第 2 列 → 再补偿 → … → 打包低比特权重
```

实际 GPTQ 使用 inverse Hessian 的 Cholesky 因子与分块延迟更新，提高稳定性和 GPU 效率；源码中变量可能仍叫 `Hinv`，但内容已是因子，不能把上面的概念公式直接当成该变量的更新式。`act-order` 等排序策略、group size、校准域和 damping 都会影响结果。

**追问：GPTQ 需要 Calibration Data，但为什么仍是 PTQ？** 校准只是收集输入统计并计算补偿，不等于梯度重训整个模型；少量样本的通道相关性也可能不代表真实业务，应独立评估代码、中文、数学和长上下文质量。

### 10.3.2 AWQ：Activation 判断重要通道 + 等价 Scaling

输出误差为 `ΔY=XΔW`，某个输入通道长期 `|x_i|` 很大时，相同权重误差 `ΔW[i,:]` 更容易影响输出。因此 AWQ 从激活统计判断 **salient input channels**，而不是只按权重绝对值挑选。[AWQ 论文](https://arxiv.org/abs/2306.00978)

AWQ 的最终部署方式并非简单把重要权重留在 FP16，而是沿输入通道搜索等价缩放：

```text
Y = (XS^-1)(SW)
Ŷ = (XS^-1) Q_dequant(SW)       # 激活保持浮点，权重仍统一低比特

原通道权重网格间距：Δ
放大 s_i 后网格间距：Δ′
折算回原坐标的有效间距：Δ′ / s_i
```

如果 group 最大值变化不大，`Δ′≈Δ`，重要通道的有效误差就减小；若过度放大把 group 的 `Δ′` 撑大，又会损害其他通道，所以 **scale 需要搜索，放大不是越多越好**。例如原网格间距 `0.1`，放大 4 倍且新间距仍是 `0.1`，折回原坐标的间距变成 `0.025`；若新间距也涨到 `0.4`，这项收益就消失。[AWQ 缩放与搜索](https://arxiv.org/html/2306.00978v4#S3)

论文通过保护约 1% 显著权重的实验展示重要性，并提出基于激活的 per-channel scale 搜索；该比例是实验观察，不是每个模型必须固定使用的部署参数。常见搜索以通道平均激活幅值的幂构造 scale 候选，比较层输出误差，也可配合 clipping。

**GPTQ 与 AWQ 的区别**：GPTQ 通过输入相关性决定逐列补偿；AWQ 通过激活幅值决定缩放保护。两者都用校准数据，AWQ 也会比较校准输出误差，不能把区别说成「一个用数据、另一个不用数据」。

### 10.3.3 Marlin：让 W4A16 的带宽收益落地

原始 Marlin 是 FP16×INT4 矩阵乘 Kernel，采用紧凑权重读取、片上反量化与浮点 Tensor Core 计算；它不是一种新的 PTQ 算法。优化包括离线重排权重 / group scale、异步 load、双缓冲、激活复用，以及解包 / 反量化指令与 MMA 的交错执行。[Marlin 官方实现](https://github.com/IST-DASLab/marlin)

```text
低效：INT4 HBM → 全量还原 FP16 → 写回 HBM → 另一个 GEMM 再读
高效：Packed INT4 + scales → 片上解包 / 反量化 → MMA → 输出
```

假设对称 INT4 每 `G=128` 个权重保存一个 FP16 scale，无 zero-point / padding：

```text
有效 bits/weight = 4 + 16/128 = 4.125
权重压缩比 = 16/4.125 ≈ 3.88×       # 不是完整模型的精确压缩比
```

短 Decode、小到中等 token batch 更可能受权重读取限制；大 Prefill / 大 batch 权重复用变高，会转向计算瓶颈。选 Kernel 时检查 GPU 架构、矩阵 shape、group size、zero-point 和排列支持；**checkpoint 写 AWQ/GPTQ 不等于运行时一定用了 Marlin**。原版 Kernel 与 vLLM 后续扩展的支持范围也不能混用。

### 10.3.4 三者一句话区别（必背）

| 对比项 | GPTQ | AWQ | SmoothQuant |
|---|---|---|---|
| 类型 | Weight-only | Weight-only | Weight + Activation |
| 常见形式 | W4A16，也研究 3 bit | W4A16 | INT8 W8A8 |
| 核心统计 | 输入相关性 / 二阶信息 | 通道激活幅值 | 激活与权重的通道范围 |
| 核心操作 | 逐列量化并补偿剩余权重 | 等价 Scaling 保护重要通道 | 等价 Scaling 迁移 Activation Outlier |
| 是否 PTQ | 是 | 是 | 是 |
| 工程重点 | 校准、factorization、group / 排列 | scale 搜索、clipping、融合逆缩放 | W/A 量化、INT8 GEMM、融合 |

> **必背**：GPTQ 看 Hessian，用二阶信息补偿量化误差；AWQ 看 Activation，识别重要 Weight Channel 并通过 Scaling 降低有效误差；SmoothQuant 主要解决 Activation Outlier，把量化难度从 Activation 部分转移到 Weight。Marlin 负责执行，决定低比特权重能否跑得快。
