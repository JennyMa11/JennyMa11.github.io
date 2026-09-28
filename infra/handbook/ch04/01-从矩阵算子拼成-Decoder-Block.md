# 4.1 从矩阵算子拼成 Decoder Block

```text
x → RMSNorm → QKV Linear → RoPE(Q,K) → Attention(KV 写入/读取)
  → Output Linear → Residual → RMSNorm → Gate/Up Linear
  → SiLU(gate) * up → Down Linear → Residual
```

```python
import torch

def rmsnorm(x, weight, eps=1e-6):
    xf = x.float()
    inv = torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + eps)
    return (xf * inv).to(x.dtype) * weight

def apply_rope(x, cos, sin):
    # x: [..., head_dim]；约定相邻两维组成一对
    even, odd = x[..., 0::2], x[..., 1::2]
    y = torch.empty_like(x)
    y[..., 0::2] = even * cos - odd * sin
    y[..., 1::2] = even * sin + odd * cos
    return y
```

RoPE 必须明确布局约定：有的模型将前半维和后半维配对，不能把相邻维实现直接套用。`cos/sin` 的位置索引来自请求的**真实 position_id**，Prefix Cache、Chunked Prefill、抢占恢复、padding/packed batch 都可能改变“当前 token 在批中的列号”，但不能改变其在序列中的位置。**MRoPE** 对多模态位置使用多个坐标轴（如时间/高度/宽度）；引擎必须把每轴位置元数据传到模型，不能用一个递增标量代替。模型配置决定具体轴布局。

**优化**：RMSNorm 的平方和用 FP32 累加，输出转回模型 dtype；融合 residual/RMSNorm 减少一次中间张量 HBM 往返。若 fused Kernel 与 PyTorch 不一致，先分别比较归一化前的 residual、平方均值、rsqrt、最终乘 weight。追问：为什么 RoPE 应加在 Q/K 而不是缓存后的 K 再重复加？历史 K 只需按原位置旋转一次。
