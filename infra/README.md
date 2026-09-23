# Infra 学习笔记

这里是个人博客 `/infra/` 文档区的源文件。`build.py` 将四份 Markdown 笔记、`figures/` 配图和 `examples/` 教学代码构建到 `public/infra/`；Astro 随后把它们纳入 GitHub Pages 站点。主手册按 GPU/CUDA → 算子 → 推理服务 → 性能与调试组织为 18 章。

## 本地构建

在仓库根目录运行：

```bash
python3 -m pip install -r infra/requirements.txt
npm ci
npm run build
npm run preview
```

访问 `/infra/`。`npm run dev` 也会先生成文档页面。修改 `infra/*.md`、配图或 `infra/assets/` 后重新运行 `npm run build:infra`，再刷新页面。

CPU 调度示例可运行 `python3 infra/examples/mini_serving.py`，行为检查可运行 `python3 -m unittest discover -s infra/examples -p 'test_*.py'`。

`public/infra/` 是生成目录，已加入 `.gitignore`。GitHub Actions 每次部署都会重新生成，避免手工维护两份 HTML。引用了仓库内不存在的项目报告会显示为“未收录”，不会产生失效链接。
