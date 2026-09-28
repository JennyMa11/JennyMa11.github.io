# 15.4 标准基准与可比性：MLPerf 等结果怎么读

[MLPerf Inference Datacenter](https://mlcommons.org/benchmarks/inference-datacenter/) 定义任务、质量要求及不同负载场景，用于规范评估。阅读结果先看模型 / 精度、场景、系统硬件、软件版本与提交规则；离线吞吐不能直接代表在线交互 SLO。

业务基准应另外固定长度分布、重复前缀比例、多模态 / LoRA 占比、到达过程和失败处理。标准结果用于外部参照，业务回放用于选型决策；任何“快多少倍”都要附匹配的质量与成本分母。
