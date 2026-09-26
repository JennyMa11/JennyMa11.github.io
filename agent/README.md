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

内容按传统 Agent、LLM Agent、系统工程、前沿与求职组织。事实和版本信息都链接至论文或官方规范；引用 [Hello-Agents](https://github.com/datawhalechina/hello-agents) 作为延伸阅读。该课程从 Agent 概念与经典方法讲到工具、记忆、协议和项目实现，配有教材与代码，适合系统学习。本站没有复制其正文或图片。

第 11 章展开 Harness Engineering、长任务检查点与恢复、Context Engineering、Agent Skills、测试时计算和 Agentic RL，提供同模型同预算的对照实验设计；新增资料于 2026-09-26 核对。

第 13 章拆解 OpenCode、OpenClaw、Hermes Agent、Claude Code 与 Codex，比较执行循环、上下文、记忆、Skills、权限、子任务与部署边界；包含固定 commit 的源码入口与公平评测设计。产品机制分析不代表已实测性能排名。

所有外部引用都有中文导读，包含主要内容、核心要点和阅读提示。在正文点击引用旁的「导读」，即可跳到本节直接展示的摘要；`/agent/sources.html` 按主题汇总全部来源，并提供综合解读。导读同样进入站内搜索。

维护时在 `source_summaries.json` 中更新来源摘要，在 `sources_overview_zh.md` 中更新综合解读。新增外部引用必须同时补充摘要，否则构建会报错；重复引用复用同一份内容。`sources.css` 仅影响 Agent 指南的导读样式。
