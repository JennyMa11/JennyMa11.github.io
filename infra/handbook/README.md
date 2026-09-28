# 手册编辑入口

手册按「主题 → 章节目录 → 小文章」组织。`ch00` 至 `ch23` 各有一个 `index.md`，用于写章节简介；同目录下编号开头的 Markdown 文件各生成一篇独立文章。`appendix/` 的组织方式相同。

例如：

```text
handbook/ch08/index.md                 → /infra/ch08.html
handbook/ch08/02-Continuous-Batching-每轮重组活跃请求.md
                                      → /infra/ch08-02.html
```

编辑小文章时直接修改对应文件，文件第一行 `#` 是页面标题。新增文章时按顺序使用下一个两位编号，文件名写清主题；构建脚本会自动把它加入章节目录、侧栏、上一篇/下一篇和站内搜索。原有 `/infra/chXX.html` 保留为目录页，旧的章节链接仍可访问。

在仓库根目录运行 `npm run build:infra`，或在当前目录运行 `../.venv/bin/python build.py`。构建结果在 `public/infra/`，不手工编辑生成的 HTML。
