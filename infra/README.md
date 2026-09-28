# Infra 学习笔记

这里是个人博客 `/infra/` 文档区的源文件。主手册已拆为 [handbook/](handbook/README.md) 中按主题分组的独立小文章：每章的 `index.md` 是目录，同目录下的编号文件是一篇文章。`build.py` 将它们、其他三份 Markdown 笔记、`figures/` 配图和 `examples/` 教学代码构建到 `public/infra/`；Astro 随后把它们纳入 GitHub Pages 站点。

## 本地构建

在仓库根目录运行：

```bash
python3 -m pip install -r infra/requirements.txt
npm ci
npm run build
npm run preview
```

访问 `/infra/`。`npm run dev` 也会先生成文档页面。修改 `infra/handbook/**/*.md`、`infra/*.md`、配图或 `infra/assets/` 后重新运行 `npm run build:infra`，再刷新页面。

CPU 调度示例可运行 `python3 infra/examples/mini_serving.py`，行为检查可运行 `python3 -m unittest discover -s infra/examples -p 'test_*.py'`。

`public/infra/` 是生成目录，已加入 `.gitignore`。GitHub Actions 每次部署都会重新生成，避免手工维护两份 HTML。引用了仓库内不存在的项目报告会显示为“未收录”，不会产生失效链接。

量化为第 10 章、投机解码为第 11 章。`examples/speculative_benchmark.py` 可向已运行的本地 vLLM 发起顺序请求并记录接受率；使用 `--help` 查看参数，示例数据集为 `speculative_code.jsonl` 与 `speculative_chat.jsonl`。客户端计时不代表 TTFT / TPOT，真实 GPU 性能需按章内实验流程测量。

## 推理优化模块结构

第 7–18 章按照 [AIInfraGuide 模块四](https://github.com/caomaolufei/AIInfraGuide/tree/main/docs/guides/%E6%A8%A1%E5%9D%97%E5%9B%9B-%E6%8E%A8%E7%90%86%E4%BC%98%E5%8C%96) 已有大纲排列，小节沿用对应主题。参考快照为 2026-09-27、commit `a3b63eeb81d6d36a3c42c8cfc5a1bdd96e36bab1`，详细内容采用本手册推导、已有实现和官方资料补充。

| 参考模块章号 | 本手册章号 | 主题 |
|---|---|---|
| 1 | 7 | LLM 推理基础 |
| 2 | 8 | 推理引擎核心技术 |
| 3 | 9 | 深入 vLLM 架构与源码 |
| 4 | 10 | 量化 |
| 5 | 11 | 投机解码 |
| 6 | 12 | 分布式推理 |
| 7 | 13 | PD 解耦 |
| 8 | 14 | 生产级服务特性 |
| 9 | 15 | 性能分析与 Benchmark |
| 10 | 16 | 生产部署与运维 |
| 11 | 17 | 选型与端到端实战 |
| 12 | 18 | 端侧推理 |

第 0–6 章保留硬件、算子与 KV 实现基础，第 19–22 章为调试、故障、面试验收及设备适配；第 23 章精读 DeepSeek-V3.2、V4、V4.1-Flash 与同期架构报告，资料核对至 2026-09-28。网页中的原 `/infra/chXX.html` 地址现在是章节目录，小文章使用 `/infra/chXX-YY.html`。GPU 配置、量化质量和跨节点性能尚需在相应硬件实测；文中的待测表格与命令模板不表示已测结果。
