# Agent Guide

这是博客 `/agent/` 的源文件。`guide_zh.md` 是手册正文，`make_figures.py` 生成可缩放的原创 SVG 原理图，`build.py` 将手册拆为章节页面并生成搜索索引。页面沿用 Infra Guide 的界面样式和交互。

在仓库根目录运行：

```bash
python3 -m pip install -r infra/requirements.txt
npm ci
npm run build
npm run preview
```

访问 `/agent/`。`npm run dev` 也会生成两套指南。修改 Markdown 或画图脚本后重新运行 `npm run build:agent`。`public/agent/` 是生成目录，GitHub Actions 部署时会重新构建。

可运行的最小工具循环：`python3 agent/examples/mini_agent.py`。这是离线教学示例，用确定性策略代替在线模型。

第 0–10、12 章已按教学流程重构，每章提供学习目标、贯穿案例、机制推演、失败分析与带参考检查的练习。两个主案例分别是运费函数修复与带来源的资料检索；新增 6 张图解释状态、规划、工具、协议、评测与信任边界。

离线实验使用 Python 标准库，无需 API Key。在仓库根目录运行：

```bash
python3 agent/examples/repair_lab.py --scenario happy
python3 agent/examples/repair_lab.py --suite
python3 agent/examples/retrieval_lab.py
python3 agent/examples/check_labs.py
```

修复实验在临时目录中修改代码并运行真实检查，演示超时、越界、假完成、过期证据和预算耗尽；修复答案由确定性策略预置，不能用于评价 LLM 编程能力。检索实验使用便于手算的 token 重叠评分，展示范围过滤、排序、Recall@k 与无答案问题，不能称为 BM25 或向量搜索。教学中的工具白名单不是 OS 沙箱。正文片段与实验建议使用 Python 3.9 或更新版本。

内容按传统 Agent、LLM Agent、系统工程、前沿与求职组织。事实和版本信息都链接至论文或官方规范；引用 [Hello-Agents](https://github.com/datawhalechina/hello-agents) 作为延伸阅读。该课程从 Agent 概念与经典方法讲到工具、记忆、协议和项目实现，配有教材与代码，适合系统学习。本站没有复制其正文或图片。

第 11 章展开 Harness Engineering、长任务检查点与恢复、Context Engineering、Agent Skills、测试时计算和 Agentic RL，提供同模型同预算的对照实验设计；新增资料于 2026-09-26 核对。

第 13 章拆解 OpenCode、OpenClaw、Hermes Agent、Claude Code 与 Codex，比较执行循环、上下文、记忆、Skills、权限、子任务与部署边界；包含固定 commit 的源码入口与公平评测设计。产品机制分析不代表已实测性能排名。

所有外部引用都有中文导读，包含主要内容、核心要点和阅读提示。在正文点击引用旁的「导读」，即可跳到本节直接展示的摘要；`/agent/sources.html` 按主题汇总全部来源，并提供综合解读。导读同样进入站内搜索。

维护时在 `source_summaries.json` 中更新来源摘要，在 `sources_overview_zh.md` 中更新综合解读。新增外部引用必须同时补充摘要，否则构建会报错；重复引用复用同一份内容。`sources.css` 仅影响 Agent 指南的导读样式。
