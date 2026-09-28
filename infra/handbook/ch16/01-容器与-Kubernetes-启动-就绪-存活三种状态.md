# 16.1 容器与 Kubernetes：启动、就绪、存活三种状态

固定镜像 digest / tag、模型 revision / 文件校验、驱动与库版本。容器内模型目录、共享内存、CPU / 主机内存、GPU device plugin 和存储吞吐都属于运行条件；只申请 GPU 数量不能保证整个服务资源足够。[vLLM Kubernetes 部署](https://docs.vllm.ai/en/stable/deployment/k8s/)

[vLLM Production Stack](https://github.com/vllm-project/production-stack) 提供 Kubernetes 集群部署的参考系统；[KServe](https://kserve.github.io/website/docs/getting-started/genai-first-isvc) 用 InferenceService 等资源组织模型服务。选用这些方案时仍需检查镜像、模型挂载、路由、扩缩容和可观测性；平台不会自动确定本业务的 SLO 容量。先用单副本跑通请求与探针，再扩展副本和控制器。

| 探针 | 问题 | 避免的错误 |
|---|---|---|
| Startup | 下载、加载、编译 / 捕获完成了吗 | 正常慢启动被不断重启 |
| Readiness | 当前实例适合接收新请求吗 | 模型未就绪就分配流量 |
| Liveness | 进程是否失去恢复能力 | 将短时拥塞误判成死锁 |

配置 startup probe 后，Kubernetes 在启动成功前不会执行 readiness / liveness；失败阈值需覆盖正常启动分布，而非凭一个固定秒数猜测。[Kubernetes Probe 语义](https://kubernetes.io/docs/tasks/configure-pod-container/configure-liveness-readiness-startup-probes/)

```yaml
# Deployment 的 Pod template 片段；镜像、PVC、模型名需替换。
spec:
  terminationGracePeriodSeconds: 120
  containers:
    - name: inference
      image: "vllm/vllm-openai:<pinned-tag>"
      args: ["--model", "/models/model", "--host", "0.0.0.0", "--port", "8000"]
      ports:
        - containerPort: 8000
      resources:
        requests:
          cpu: "4"
          memory: "16Gi"
          nvidia.com/gpu: "1"
        limits:
          memory: "16Gi"
          nvidia.com/gpu: "1"
      volumeMounts:
        - name: models
          mountPath: /models
          readOnly: true
      startupProbe:
        httpGet: {path: /health, port: 8000}
        periodSeconds: 10
        failureThreshold: 60
      readinessProbe:
        httpGet: {path: /health, port: 8000}
        periodSeconds: 10
      livenessProbe:
        httpGet: {path: /health, port: 8000}
        periodSeconds: 30
        failureThreshold: 3
  volumes:
    - name: models
      persistentVolumeClaim: {claimName: model-weights}
```

资源值与时间阈值只是示例；模型、Graph、KV 与长请求可能需要更多 CPU / 内存。`/health` 反映进程健康，不一定保证有容量满足新请求 SLO，admission 与负载信号另外实现。上线先测正常慢启动、依赖失败和持续拥塞，确保探针不会导致重启风暴。

关闭时先停止接收新流量，再 drain 活跃请求，最后退出；终止宽限期、服务发现摘除延迟和流式请求最长持续时间一起设计。不能仅设置 grace period 就声称完成了优雅退出。
