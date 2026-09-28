# 22.2 NPU 软件栈：从设备到模型

NPU 是面向神经网络张量运算的加速器；CPU 更适合控制与通用逻辑，GPU 提供大规模并行计算，NPU 的算子、内存与编译执行路径则由具体平台决定。以 NVIDIA 与昇腾为例：

```text
NVIDIA: PyTorch/推理框架 → CUDA → cuBLAS/cuDNN/NCCL 等 → GPU
昇腾:  PyTorch/推理框架 → TorchNPU (torch_npu) → CANN/HCCL → Ascend NPU
```

CUDA 自定义 Kernel、CUDA 版 FlashAttention 或 PagedAttention **不能直接在昇腾上执行**。需要先核对目标版本是否已有等价 NPU 算子，再考虑用已有算子组合、替换 attention backend、修改布局，最后才实现自定义算子。已有算子“名称相同”也不能跳过 shape、mask、精度与性能验证。昇腾的 [TorchNPU 项目](https://github.com/Ascend/pytorch)提供 PyTorch 设备适配；其[版本配套表](https://github.com/Ascend/pytorch/blob/master/COMPATIBILITY.en.md)要求检查 PyTorch、TorchNPU、CANN、Python、驱动与固件组合；[HCCL 文档](https://www.hiascend.com/document/detail/en/CANNCommunityEdition/910/commlib/hcclug/docs/en/user_guide/hccl_intro.md)说明集合通信支持。具体 API 与工具名称随版本变动，应以当前部署版本文档为准。

**启动顺序**：设备可见与健康 → 驱动/固件 → CANN Runtime → PyTorch/TorchNPU → 最小张量 MatMul → 模型加载 → 首个 Forward → Prefill/Decode。若最小 MatMul 都失败，先修环境；不要直接改模型代码。记录版本、设备型号、容器镜像和环境变量，使成功运行可复现。
