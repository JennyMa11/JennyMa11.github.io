# 14.3 Multi-LoRA：共享底座不等于共享全部状态

LoRA 将权重更新表示为低秩矩阵，例如本章采用 `W′=W+αBA/r`，其中 `B∈R^[dout,r]`、`A∈R^[r,din]`。单矩阵 Adapter 参数约 `r(din+dout)`，小于完整 `din×dout` 更新；实际总量取决于被适配层。

多业务共享底座，仍需 Adapter 权重、额外计算、每请求 KV 与加载缓冲。不同 Adapter 的 hidden state 通常不同，Prefix KV key 必须区分 Adapter，不能把相同文本的 KV 无条件共享。

```bash
# 静态注册两个与底座匹配的 Adapter；未在本机实跑。
vllm serve /models/base --enable-lora \
  --lora-modules business-a=/models/lora-a business-b=/models/lora-b
```

请求按注册名称选择 Adapter；rank 上限、每批可用 Adapter 数、动态加载方式与后端支持按当前配置检查。[vLLM LoRA Adapters](https://docs.vllm.ai/en/stable/features/lora/)

调度时比较同 Adapter 集中请求与高混合度请求，记录加载命中率、底座 / Adapter / KV 显存、吞吐和 P99。动态卸载必须等活跃引用安全释放；模型 lineage、Adapter revision 与 base revision 一起保存。
