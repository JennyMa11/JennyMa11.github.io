# 14.1 结构化输出：约束合法 Token，不替代业务校验

受约束解码把 JSON Schema / 语法编译成状态机或等价结构，每个生成位置根据当前解析状态计算允许 token 集合，对不合法 logits 做 mask，然后采样并推进状态。

```text
Schema 编译 → 维护解析状态 → legal-token mask
           → logits 约束 → sample → 更新状态 → 继续
```

它可以提高格式与 Schema 合规性，但“合法 JSON”不代表事实正确、字段合理或工具参数安全。达到输出长度上限、请求取消、模型异常时仍可能返回不完整结果，业务必须检查 finish reason、完整解析和语义规则。

```bash
# 向 9.1 的本地服务请求 JSON Schema 输出。
curl http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"infra-demo","messages":[{"role":"user","content":"用一个字段简要解释 KV Cache。"}],"max_tokens":128,"response_format":{"type":"json_schema","json_schema":{"name":"explanation","schema":{"type":"object","properties":{"answer":{"type":"string"}},"required":["answer"],"additionalProperties":false}}}}'
```

后端的 Schema 子集与 tokenization 处理不同，先验证目标语法，再测编译冷启动、mask 开销、成功率和长尾。与投机组合时约束必须在正确候选前缀生效，不能沿用未约束 p/q 的证明。[vLLM Structured Outputs](https://docs.vllm.ai/en/stable/features/structured_outputs/)
