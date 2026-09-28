# 8.1 PagedAttention：分页寻址与块管理

逻辑 token 位置通过 block table 映射物理 KV 块。Kernel 必须同时接收序列长度、slot mapping 和有效位置，不能把物理分页读成连续张量。新增 token 前先预留槽位；已共享的部分尾块追加可能需要写时复制。

```text
logical_block = token_position // block_size
offset = token_position % block_size
physical_slot = block_table[logical_block] × block_size + offset
```

页表查找与非连续读取增加开销，收益来自减小碎片、支持动态批次和安全共享，而不是改变 Attention 数学。可运行 CPU 管理器、寻址图和释放规则见[6.1–6.3、6.10](ch06.html)。
