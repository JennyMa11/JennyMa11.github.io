# 14.4 多模态：视觉编码、语言 Token 与两类缓存

```text
图像 / 音视频 → 解码与预处理 → Encoder → 投影 / 融合
            → 语言模型 Prefill → KV → Decode
```

分辨率、图像裁块、帧数和音频时长影响预处理、Encoder 与语言 token 数。文本长度相同的两条请求，实际 GPU 工作量可能完全不同。测量要分别记录媒体处理、Encoder、Prefill 和输出时间。

**Encoder Cache**复用媒体编码结果；**Prefix KV Cache**复用语言模型已计算的前缀，两者处于不同阶段。缓存 key 应包含媒体内容、预处理参数、Encoder / 模型版本与 Adapter 等结果相关条件。只哈希图片 URL 不足以判断其内容是否变化。

服务限制应同时覆盖媒体数量 / 尺寸 / 时长、展开后的 token 与并发资源。扩容瓶颈也可能是 CPU 解码和主机内存，GPU 利用率低时先看预处理队列。[vLLM Multimodal Inputs](https://docs.vllm.ai/en/stable/features/multimodal_inputs/)

**实验**：同一图重复请求与不同图请求；固定文本、改变媒体复杂度；比较冷 / 热 Encoder 与 Prefix 命中。质量、内存和 TTFT 一起报告。
