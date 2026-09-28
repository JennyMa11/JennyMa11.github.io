# 5.1 从朴素 Attention 到分块执行

令 `Q:[B,Hq,Sq,D]`、`K/V:[B,Hkv,Sk,D]`，GQA 要求 `Hq % Hkv == 0`。数学参考实现先构造 `Sq×Sk` scores；这是正确性 oracle，却会把大矩阵写入显存。FlashAttention 通过 Q/K/V tile 与 online softmax 在片上更新状态。

```text
Q tile → 加载 K/V tile → S=QKᵀ×scale + mask
       → 行最大值 m_new → p=exp(S-m_new)
       → l_new=exp(m_old-m_new)×l_old+sum(p)
       → acc_new=exp(m_old-m_new)×acc_old+pV
       → 下一 tile；最终 O=acc/l
```

```python
import torch

def online_attention(q, k, v, block=64):
    # q:[Q,D], k/v:[K,D]；教学版，先用 float32 验证
    scale = q.shape[-1] ** -0.5
    out = []
    for qs in range(0, q.shape[0], block):
        qb = q[qs:qs + block].float()
        m = torch.full((qb.shape[0], 1), -float('inf'), device=q.device)
        l = torch.zeros_like(m)
        acc = torch.zeros((qb.shape[0], v.shape[-1]), device=q.device)
        for ks in range(0, k.shape[0], block):
            score = qb @ k[ks:ks + block].float().T * scale
            # causal: 全局 q 位置与 k 位置比较；空 tile 不参加更新
            qi = torch.arange(qs, qs + qb.shape[0], device=q.device)[:, None]
            ki = torch.arange(ks, ks + score.shape[1], device=q.device)[None, :]
            score = score.masked_fill(ki > qi, -float('inf'))
            new_m = torch.maximum(m, score.max(-1, keepdim=True).values)
            safe_m = torch.where(torch.isfinite(new_m), new_m, torch.zeros_like(new_m))
            alpha = torch.where(torch.isfinite(m), torch.exp(m - safe_m), 0)
            p = torch.exp(score - safe_m)
            acc = acc * alpha + p @ v[ks:ks + block].float()
            l = l * alpha + p.sum(-1, keepdim=True)
            m = new_m
        out.append(acc / l.clamp_min(1e-20))
    return torch.cat(out, dim=0)
```

上面针对整段 causal self-attention；Decode 时 `Q` 只有当前 token，但 `qi` 必须取其**绝对位置**，不能沿用局部行号。变长批还需 `cu_seqlens`、每请求长度和 mask；全 mask 行要定义输出，不可让 `-inf - -inf` 产生 NaN。工业实现可从 [FlashAttention 的 API 与 CUDA/Triton 实现](https://github.com/Dao-AILab/flash-attention) 先找入口和布局，再跟踪 tile 循环及 online softmax 状态。
