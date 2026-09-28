# 4.2 算子融合（RMSNorm / RoPE / SiLU）

**速查**：把多个连续的小算子合并成一个 Kernel，减少 Kernel Launch 次数和中间张量的 HBM 读写。

**术语解释**：

- **算子融合（Operator Fusion）**：把多个逐元素/归约算子合并进同一个 Kernel。
- **RMSNorm（Root Mean Square Layer Normalization，均方根层归一化）**：LLM 常用的归一化层，比 LayerNorm 更省计算。
- **SiLU（Sigmoid Linear Unit，也称 Swish 激活函数）**：`x · sigmoid(x)`，常用于 SwiGLU 门控结构。
- **SwiGLU**：用 SiLU 做门控的前馈网络结构，形式为 `SiLU(xW₁) ⊙ (xW₃)`。

| 融合模式 | 融合前 | 融合后收益 |
|---|---|---|
| **RMSNorm + Residual** | `add` 写 HBM → `rmsnorm` 读 HBM | 省一次读写 |
| **SiLU + Mul**（SwiGLU） | `silu` 写 HBM → `mul` 读 HBM | 省一次读写 |
| **RoPE** | 多次 reshape/mul/add | 合并为单 Kernel |
| **QKV Projection** | 三个 Linear | 合并为一个 GEMM |
| **整个 MLP** | Linear → SiLU → Mul → Linear | 减少中间激活 |

**码 3-3 · 融合前 vs 融合后**

```python
# 融合前：3 次 Kernel Launch，中间结果 3 次往返 HBM
h = x + residual                                              # add     → 写 HBM
h = h * torch.rsqrt(h.pow(2).mean(-1, keepdim=True) + eps)     # rmsnorm → 读 + 写 HBM
y = h * weight                                                # mul     → 读 + 写 HBM

# 融合后：1 次 Kernel Launch，h 只存在于寄存器 / SRAM
y = fused_add_rmsnorm(x, residual, weight, eps)
```

> `fused_add_rmsnorm` 是伪函数名，代表把三步塞进同一个 kernel。收益可以算得很清楚：**省 2 次 Launch 开销 + 省掉 `h` 的 3 次写、3 次读**。中间张量越大、算子越碎，这个收益越高——这正是 torch.compile 在 Decode 阶段（大量小算子、访存受限）特别有效的原因。

**追问：为什么融合能加速？**
> 因为 GPU 上「启动一个 Kernel」有固定开销，且每个 Kernel 都要把输入从 HBM 读到寄存器、把输出写回 HBM。融合后中间结果留在寄存器/SRAM 里，**既省 Launch 开销又省带宽**——对访存受限的 Decode 尤其有效。

---
