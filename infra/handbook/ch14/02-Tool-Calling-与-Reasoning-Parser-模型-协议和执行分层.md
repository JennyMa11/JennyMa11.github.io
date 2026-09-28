# 14.2 Tool Calling 与 Reasoning Parser：模型、协议和执行分层

Tool Calling 是模型提出结构化工具名 / 参数，服务把生成文本解析成协议字段，应用执行工具，再把结果作为新的上下文交回模型。**vLLM 返回 tool call 不意味着已经执行了工具**。

```text
tools 描述 + messages → chat template → 模型生成
  → tool parser → 应用校验 / 执行 → tool result → 再次生成
```

模型是否学过工具协议、模板是否编码 tools、parser 是否匹配输出格式，是三个独立检查。流式参数可能跨多个 chunk，只能完整组装后执行；取消、重试、同一调用 ID 的幂等处理在应用层设计。

对于支持的模型，自动工具选择常使用 `--enable-auto-tool-choice`、匹配的 `--tool-call-parser`，必要时配置 tool-compatible chat template。Parser 名称不能从另一个模型的例子随意复制。[vLLM Tool Calling](https://docs.vllm.ai/en/stable/features/tool_calling/)

Reasoning Parser 把模型定义的推理段与可见回答分开，既不改变模型推理能力，也不自动缩短总输出。性能测试应明确是否计入 reasoning token、工具往返与重新 Prefill；仅展示最终回答时的“可见首字”也可能晚于引擎首 token。[vLLM Reasoning Outputs](https://docs.vllm.ai/en/stable/features/reasoning_outputs/)

**验收**：参数解析率、Schema 合规、工具选择正确率、执行超时与多轮总时延；不能只测单次模型 tokens/s。
