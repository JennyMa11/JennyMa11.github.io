# 7.4 瓶颈分析：算术强度与 Roofline

```text
# KV Cache 显存
KV_bytes = 2 × L × H_kv × D × S × B × dtype_bytes

# Decode 理论下界（访存受限）
TPOT_min ≈ (模型权重字节数 + KV Cache 字节数) / 峰值带宽

# 单 Token 计算量（Decode）
FLOPs ≈ 2 × N_active_params        # 仅激活参数（MoE 只算激活专家）

# 算术强度
Intensity = FLOPs / Bytes
# Intensity < 机器平衡点 → 访存受限（Decode）
# Intensity > 机器平衡点 → 计算受限（Prefill、大 Batch）

# 投机解码期望输出 Token 数（含修正或额外 Token；接受率 α、Draft 长度 γ）
E[tokens] ≈ (1 - α^(γ+1)) / (1 - α)
```

Roofline 的算术强度是 FLOPs / 实际搬运字节；必须注明计入的是权重、KV 还是全部数据。低 batch Decode 经常受权重 / KV 读取限制；高 batch、MoE 通信、长上下文 Attention 或 CPU 调度可能改变主瓶颈。Prefill / Decode 是初始假设，用时间线和计数器验证。
