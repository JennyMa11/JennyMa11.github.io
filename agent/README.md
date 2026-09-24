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

内容按传统 Agent、LLM Agent、系统工程、前沿与求职组织。事实和版本信息都链接至论文或官方规范；引用 [Hello-Agents](https://github.com/datawhalechina/hello-agents) 作为延伸阅读，没有复制其正文或图片。
