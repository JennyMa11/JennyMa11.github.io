# 18.1 端侧推理基础：约束、硬件、Runtime、优化与 Benchmark

### 资源账本：权重能加载不等于应用能稳定运行

```text
应用峰值内存 ≈ 权重驻留 + KV + 激活 / Workspace + Runtime
             + 预处理 / UI 缓冲 + 后端转换 / 重复副本
```

手机或嵌入式设备的物理内存不是 App 可独占内存。3B 权重按 16/8/4 bit 的理论 payload 为 6/3/1.5 GB（十进制），实际还包含 scale、文件 / 缓冲对齐与运行状态。最大稳定上下文要从可用内存而非仅权重文件大小推导。

CPU、GPU、NPU 可能共享物理内存，但映射、layout 转换、Cache 一致性和同步仍有成本。零拷贝是一项需要证明的执行属性，不能从“共享内存”直接推断。

散热与电池决定持续性能；后台切换、低内存回收与模型更新决定生命周期。基准必须分别测冷启动、热状态和持续运行后的性能。

### 格式、Runtime 与后端：三者分别检查

| 层次 | 负责什么 | 示例 |
|---|---|---|
| 模型格式 / IR | 图、权重、形状与类型描述 | ExportedProgram、ONNX、GGUF |
| Runtime | 加载、内存、调度、后端调用 | ExecuTorch、ONNX Runtime、llama.cpp |
| Backend / Delegate | 执行受支持算子 / 子图 | XNNPACK、GPU provider、设备 NPU 后端 |

启用 NPU 后若多处不支持节点回退 CPU，会产生 `NPU→CPU→NPU` 搬运与同步。检查**按耗时 / FLOPs 加权的覆盖率**和切换次数，仅数“有多少节点被委托”容易被小算子误导。[ONNX Runtime Execution Providers](https://onnxruntime.ai/docs/execution-providers/)

| 工作流 | 适合优先验证的入口 | 需要确认 |
|---|---|---|
| PyTorch 导出 | ExecuTorch | export 能力、委托分区、shape 与 Runtime 版本 |
| 多框架图 | ONNX Runtime | operator set、provider 兼容、转换误差 |
| 本地 LLM | llama.cpp / GGUF | 量化类型、KV、CPU / GPU offload 与目标后端 |

这不是性能排名。先固定模型、输入、精度与目标设备，用 CPU 基线比较端到端延迟与能耗，再选后端。[ExecuTorch 导出流程](https://docs.pytorch.org/executorch/stable/using-executorch-export.html)、[llama.cpp 官方项目](https://github.com/ggml-org/llama.cpp)

### 导出、分区、内存规划：从静态小图开始

导出要暴露实际允许的输入 shape、状态与控制流。固定 shape 便于编译优化，动态 shape 需要范围约束和覆盖验证；不能只在一个样本导出成功就认为所有业务输入支持。

下面独立定义一个小模型，用于验证 PyTorch → ExecuTorch → XNNPACK 的导出接口；需要安装版本匹配的 PyTorch / ExecuTorch，本机没有运行此后端示例：

```python
from pathlib import Path
import torch
from executorch.backends.xnnpack.partition.xnnpack_partitioner import XnnpackPartitioner
from executorch.exir import to_edge_transform_and_lower

class TinyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.proj = torch.nn.Linear(16, 8)

    def forward(self, x):
        return torch.relu(self.proj(x))

torch.manual_seed(0)
model = TinyModel().eval()
inputs = (torch.randn(1, 16),)
with torch.no_grad():
    reference = model(*inputs)
torch.save({"inputs": inputs, "reference": reference}, "tiny_reference.pt")
ep = torch.export.export(model, inputs)
program = to_edge_transform_and_lower(
    ep, partitioner=[XnnpackPartitioner()]
).to_executorch()
Path("tiny.pte").write_bytes(program.buffer)
```

生成 `.pte` 只是导出环节，仍需在目标 Runtime 加载、执行同一输入、与 reference 比较，并查看分区日志。小线性图不能证明整个 LLM、KV 状态或任意动态长度已可委托。[官方导出 API](https://docs.pytorch.org/executorch/stable/using-executorch-export.html)

内存规划按张量生命周期复用缓冲，确认 inplace 不破坏仍有消费者的输入；预处理与 UI 避免重复复制大张量。量化 / layout 转换尽量融合到后端路径，但保留精度与输出校验。内存映射可以减少急切读取，仍应测首次触页、驻留内存与随机读开销。

### 可信基准：冷、热、稳态三组结果

| 组别 | 记录内容 | 目的 |
|---|---|---|
| 冷启动 | App、模型加载、编译 / 初始化、首次响应 | 用户首次体验 |
| 热状态 | 预热后 P50/P95/P99、内存、线程、输入 shape | 相同条件比较优化 |
| 稳态持续 | 温度、频率 / 功耗、性能曲线、内存趋势 | 发现降频、泄漏与能耗问题 |

LLM 额外记录 TTFT、TPOT、实际输出长度、最大稳定上下文与 KV dtype；视觉任务按实际媒体形状记录。功耗测量要说明工具、边界和基线，`平均功率×时间/有效 token` 可估算能耗，但不能把不同设备的测量口径混在一张榜单。

固定 SoC / OS / 驱动、模型 hash、Runtime、委托覆盖、线程数、冷 / 热状态与输入。持续时长由产品需求确定，记录原始曲线，避免只选温度尚未升高时的最佳一次。

### App 集成与排错：性能以完整用户路径验收

```text
导出失败 → unsupported op / 控制流 / shape 约束
输出不一致 → 预处理、量化轴、layout、第一处分歧
NPU 不如 CPU → 分区、搬运、同步、小算子 / 低覆盖
首次慢 → 加载、触页、编译与初始化
持续变慢 → 温度 / 降频、线程竞争、内存增长
后台恢复失败 → 状态恢复、资源释放与模型版本
```

模型加载 / 推理避免阻塞 UI；取消请求后停止不必要计算并回收状态；低内存时按产品策略缩短上下文、减少并发或切换小模型，质量与体验变化需明确。更新采用可校验版本包并保留回退方式，Runtime 和模型必须作为兼容组合管理。

**交付物**：一份模型导出与支持矩阵、一份实际分区记录、一份逐层 / 输出误差表、冷热稳态性能与内存报告、生命周期演练记录。能够把“指定了后端”解释成“哪些计算实际跑在哪里”，才完成端侧适配。

---
