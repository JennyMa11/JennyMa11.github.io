# 10.2 W8A8（SmoothQuant）：Activation Outlier 与等价变换

**问题**：在一些 LLM 中，激活的少数输入通道长期幅值很大，直接 INT8 量化会把其余通道的分辨率压低。SmoothQuant 是面向 W8A8 的 PTQ 方法，用可逆缩放在激活和权重之间重新分配动态范围。[SmoothQuant 论文](https://arxiv.org/abs/2211.10438)

### 10.2.1 为什么输出不变

沿用 `Y=XW`，令正对角矩阵 `S=diag(s₁,…,s_Cin)`：

```text
Y = XW = (XS^-1)(SW)
X′[:,i] = X[:,i] / s_i       # 缩小激活的大通道
W′[i,:] = W[i,:] × s_i       # 把对应权重行放大
```

例如一个通道的 Activation 最大值为 `100`，对应 Weight 最大值为 `1`，取 `s_i=10` 后，两边的最大值都变成 `10`；浮点乘积仍相同。接下来量化的是 `X′` 和 `W′`，**量化后的乘积仅近似相同**，等价变换自身不意味着量化无损。

### 10.2.2 Scale 怎么选

校准数据得到每个输入通道的激活最大幅值 `a_i`，权重对应行最大幅值为 `w_i`，常用形式：

```text
s_i = a_i^α / w_i^(1-α),     0 ≤ α ≤ 1
# 实现时给 a_i、w_i 加下界，避免零通道除零。
```

`α` 越大越向权重迁移难度；太大可能让权重量化变差。`α=0.5` 是平衡起点，最终用校准误差和独立验证集选取。**S 是离线 smoothing 参数，INT8 的量化 scale 是另一组参数**；后者可以是静态校准，也可以运行时按 token 动态统计。[公式与迁移强度](https://arxiv.org/html/2211.10438v6#S4)

### 10.2.3 从公式到 Kernel

```text
离线：校准 a_i → 选择 S → 重写 W′ → 量化并保存 Wq 与 scale
运行：生成 X′ → 量化 Xq → INT8 GEMM → 累加 / rescale → 输出
```

逆缩放能否折叠进前面的 Norm 或线性层，要检查共享消费者、残差支路和融合 QKV 的一致性。直接多发射一个 scale Kernel 会有额外流量。INT8 GEMM 常用 INT32 累加，输出缩放再与 bias / residual 等融合；不要把 FP8 的累加规则照搬过来。

**必背**：SmoothQuant 通过 `Activation / s、Weight × s` 把激活 Outlier 的量化难度部分转移到权重，目标是让两者都适合 INT8，支持高效 W8A8 计算。
