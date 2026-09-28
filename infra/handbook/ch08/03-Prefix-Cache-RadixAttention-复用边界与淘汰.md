# 8.3 Prefix Cache / RadixAttention：复用边界与淘汰

前缀相同是 token ID、位置语义及模型状态一致，不是文本看起来相同。模型 revision、Adapter、多模态内容和 KV dtype 都可能进入缓存身份。整块 hash 链和基数树是索引组织方式，底层仍需管理块引用和 GPU 容量。命中可减少重复 Prefill，不能跳过后续 Decode；前缀在不同副本上命中率还取决于路由。

实验固定请求分布，比较冷缓存、热缓存、低重复前缀，记录有效复用 token 数、TTFT、缓存占用和淘汰率。完整 Radix 图解与共享状态账本见[6.4、6.10](ch06.html)。[vLLM Prefix Caching 设计](https://docs.vllm.ai/en/stable/design/prefix_caching/)
