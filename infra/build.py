#!/usr/bin/env python3
"""Build the Infra notes as a self-contained static documentation site."""

import html
import json
import re
import shutil
from pathlib import Path

import markdown
from markdown.extensions.toc import slugify_unicode
from pygments.formatters import HtmlFormatter


HERE = Path(__file__).resolve().parent
ROOT = HERE
OUT = HERE.parent / "public" / "infra"
EXTENSIONS = ["fenced_code", "tables", "codehilite", "toc"]
EXTENSION_CONFIG = {
    "codehilite": {"guess_lang": False, "css_class": "codehilite"},
    "toc": {"slugify": slugify_unicode, "permalink": False},
}

CHAPTER_DESCS = {
    0: "一条请求的时间、字节与状态账本",
    1: "SM、Warp、Block、存储层次与占用率",
    2: "CUDA 索引、访存、归约、扫描与 Softmax",
    3: "GEMV/GEMM、Tensor Core、Triton 与扩展",
    4: "RMSNorm、RoPE、MRoPE 与算子融合",
    5: "GQA/MLA、在线 Softmax 与 FlashAttention",
    6: "KV 块分配、分页寻址与前缀复用",
    7: "Prefill/Decode 单请求生成与显存账本",
    8: "Token/KV 双预算、分块预填充与状态机",
    9: "FP8/INT8/INT4、量化校验与投机验证",
    10: "Stream、Event、CUDA Graph 与执行重叠",
    11: "TP/PP/EP、NCCL、MoE 与 PD 分离",
    12: "vLLM、SGLang、TensorRT-LLM 与 llama.cpp",
    13: "Roofline、Nsight、Warp Stall 与定位流程",
    14: "逐层逐 Token 对比与第一处分歧定位",
    15: "CUDA、Kernel、Serving 与分布式故障模式",
    16: "从 API 到最终 Token 的完整工程复盘",
    17: "解释、实现和调试能力验收",
    18: "GPU/NPU 软件栈、算子迁移、精度定位与验收",
}


def read(path):
    return path.read_text(encoding="utf-8")


def md_to_html(source):
    return markdown.markdown(
        source, extensions=EXTENSIONS, extension_configs=EXTENSION_CONFIG
    )


def plain(source):
    source = re.sub(r"<(script|style)\b.*?</\1>", " ", source, flags=re.I | re.S)
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", source))).strip()


def sections(body, title):
    result = []
    for part in re.split(r'(?=<h2 id=")', body):
        heading = re.match(r'<h2 id="([^"]+)">(.*?)</h2>', part, flags=re.S)
        text = plain(part)
        if text:
            result.append({
                "anchor": heading.group(1) if heading else "",
                "heading": plain(heading.group(2)) if heading else title,
                "text": text[:4000],
            })
    return result


def make_page(url, group, title, desc, source, number=None):
    body = md_to_html(source)
    # Some project notes cite reports outside this repository. Keep their names
    # visible without publishing links that cannot resolve on the static site.
    def unavailable_link(match):
        filename = html.unescape(match.group(1))
        if (ROOT / filename).is_file():
            return match.group(0)
        return f'<span class="unavailable-link" title="原始资料未收录">{match.group(2)} <small>未收录</small></span>'

    body = re.sub(r'<a href="([^"/]+\.md)">(.*?)</a>', unavailable_link, body)
    headings = [
        (match.group(1), plain(match.group(2)))
        for match in re.finditer(r'<h2 id="([^"]+)">(.*?)</h2>', body, flags=re.S)
    ]
    return {
        "url": url, "group": group, "title": title, "desc": desc,
        "number": number, "body": body, "headings": headings,
        "sections": sections(body, title),
        "minutes": max(1, round(len(plain(body)) / 600)),
    }


def load_pages():
    source = read(ROOT / "inference_accel_handbook_zh.md")
    lines = source.splitlines()
    starts = [
        i for i, line in enumerate(lines)
        if re.match(r"^# (第 \d+ 章|附录)", line)
    ]
    if not starts:
        raise ValueError("Handbook has no chapter headings")
    preface = "\n".join(lines[:starts[0]])
    preface = re.sub(r"^# .+\n", "", preface, count=1)
    pages = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(lines)
        title = lines[start][2:].strip()
        match = re.match(r"第 (\d+) 章", title)
        number = int(match.group(1)) if match else None
        url = f"ch{number:02d}.html" if number is not None else "appendix.html"
        desc = CHAPTER_DESCS.get(number, "术语与补充资料")
        chapter_source = "\n".join(lines[start + 1:end])
        if number == 0 and preface.strip():
            chapter_source = preface + "\n\n" + chapter_source
        pages.append(make_page(url, "知识手册", title, desc, chapter_source, number))

    extras = [
        ("interview_qa_zh.md", "项目实测", "qa.html", "nano-vLLM 项目实测与面试问答"),
        ("paper_writing_framework_zh.md", "延伸笔记", "writing_framework.html", "技术论文与方案的写作方法"),
        ("paper_skeleton_zh.md", "延伸笔记", "writing_skeleton.html", "可直接填空的技术方案骨架"),
    ]
    for filename, group, url, desc in extras:
        source = read(ROOT / filename)
        title_match = re.search(r"^# (.+)$", source, flags=re.M)
        title = title_match.group(1).strip() if title_match else filename
        source = re.sub(r"^# .+\n", "", source, count=1)
        pages.append(make_page(url, group, title, desc, source))
    return pages


def e(value):
    return html.escape(str(value), quote=True)


def navigation(pages, active):
    groups = []
    for group in ("知识手册", "项目实测", "延伸笔记"):
        items = []
        for page in pages:
            if page["group"] != group:
                continue
            is_active = page["url"] == active
            current_attr = ' aria-current="page"' if is_active else ""
            label = (
                f'<span class="nav-number">{page["number"]:02d}</span>'
                if page["number"] is not None else '<span class="nav-number nav-symbol">↗</span>'
            )
            items.append(
                f'<a class="nav-link{" active" if is_active else ""}" '
                f'href="{e(page["url"])}"{current_attr}>'
                f'{label}<span>{e(page["title"])}</span></a>'
            )
        groups.append(f'<div class="nav-group"><p class="nav-heading">{group}</p>{"".join(items)}</div>')
    home_class = " active" if active == "index.html" else ""
    home_current_attr = ' aria-current="page"' if home_class else ""
    return (
        f'<a class="nav-link nav-home{home_class}" href="index.html"'
        f'{home_current_attr}>'
        '<span class="nav-number nav-symbol">⌂</span><span>首页 · 学习地图</span></a>'
        + "".join(groups)
    )


def shell(title, description, body, pages, active, toc="", page_class=""):
    return f'''<!doctype html>
<html lang="zh-CN" data-theme="light">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="{e(description)}">
  <meta name="theme-color" content="#f8faf8">
  <title>{e(title)} · Infra Notes</title>
  <link rel="stylesheet" href="assets/pygments.css">
  <link rel="stylesheet" href="assets/style.css">
  <script>try{{const saved=localStorage.getItem("theme");document.documentElement.dataset.theme=saved||(matchMedia("(prefers-color-scheme: dark)").matches?"dark":"light")}}catch(e){{}}</script>
</head>
<body class="{page_class}">
  <div id="progress" aria-hidden="true"></div>
  <header class="topbar">
    <button class="icon-button menu-button" id="menu-button" aria-label="打开目录" aria-expanded="false" aria-controls="sidebar">☰</button>
    <a class="brand" href="index.html"><span class="brand-mark">I<span>·</span></span><span>Infra<span class="brand-light"> Notes</span></span></a>
    <span class="topbar-divider"></span><span class="topbar-context">AI Infra 工程实践指南</span>
    <a class="back-link" href="/">← 返回主博客</a>
    <div class="topbar-actions">
      <div class="searchbox" id="searchbox">
        <label class="visually-hidden" for="search-input">搜索笔记</label>
        <input id="search-input" type="search" placeholder="搜索笔记..." autocomplete="off" aria-controls="search-results" aria-expanded="false">
        <kbd class="search-shortcut">⌘ K</kbd>
        <div id="search-results" class="search-results" role="listbox"></div>
      </div>
      <button class="icon-button theme-button" id="theme-toggle" aria-label="切换深色模式" title="切换深色模式">◐</button>
    </div>
  </header>
  <div class="sidebar-scrim" id="sidebar-scrim"></div>
  <aside class="sidebar" id="sidebar" aria-label="站点目录">
    <div class="sidebar-inner"><div class="sidebar-caption">EXPLORE / 探索</div>
      <nav>{navigation(pages, active)}</nav>
      <div class="sidebar-bottom"><a href="/">← 返回 Jenny Ma 主博客</a><br><span class="status-dot"></span>持续整理中的学习笔记</div>
    </div>
  </aside>
  <div class="site-layout">
    <main class="content" id="main-content">{body}
      <footer class="site-footer"><span>Infra Notes · 从原理走向实现</span><span>基于 Markdown 构建</span></footer>
    </main>
    {toc}
  </div>
  <script src="assets/app.js" defer></script>
</body>
</html>'''


def render_article(page, pages):
    toc = ""
    if page["headings"]:
        links = "".join(
            f'<a href="#{e(anchor)}">{e(heading)}</a>' for anchor, heading in page["headings"]
        )
        toc = f'<aside class="page-toc" aria-label="本页目录"><p>本页目录</p><nav>{links}</nav></aside>'
    index = pages.index(page)
    previous = pages[index - 1] if index else None
    following = pages[index + 1] if index + 1 < len(pages) else None
    pagination = '<nav class="pagination" aria-label="上一篇与下一篇">'
    for item, direction, label in ((previous, "prev", "上一篇"), (following, "next", "下一篇")):
        if item:
            pagination += (
                f'<a class="pagination-link {direction}" href="{e(item["url"])}">'
                f'<span>{label}</span><strong>{e(item["title"])}</strong></a>'
            )
    pagination += "</nav>"
    number = f' · 第 {page["number"]:02d} 章' if page["number"] is not None else ""
    body = f'''
      <div class="article-wrap">
        <div class="breadcrumb"><a href="index.html">首页</a><span>／</span>{e(page["group"])}{number}</div>
        <article class="article">
          <header class="article-header"><span class="eyebrow">{e(page["group"])} / READING NOTES</span>
            <h1>{e(page["title"])}</h1><p class="article-lead">{e(page["desc"])}</p>
            <div class="article-meta"><span>约 {page["minutes"]} 分钟阅读</span><span class="meta-dot">·</span><span>持续更新</span></div>
          </header>
          <div class="prose">{page["body"]}</div>
        </article>{pagination}
      </div>'''
    return shell(page["title"], page["desc"], body, pages, page["url"], toc, "article-page")


def render_home(pages):
    chapters = [page for page in pages if page["number"] is not None]
    extras = [page for page in pages if page["group"] != "知识手册"]
    routes = [
        ("01", "硬件与算子", "从 GPU 执行模型写到 GEMM、Softmax 和 Triton。", "ch01.html", "第 1–3 章", "route-map"),
        ("02", "模型与缓存", "实现 Transformer 算子、Attention 和分页 KV。", "ch04.html", "第 4–7 章", "route-kernel"),
        ("03", "服务与分布式", "追踪调度、量化、运行时和多卡通信。", "ch08.html", "第 8–12 章", "route-system"),
        ("04", "测量与排障", "从 Profiling、逐层比对到完整请求复盘。", "ch13.html", "第 13–17 章", "route-practice"),
    ]
    route_cards = "".join(
        f'<a class="route-card {css}" href="{url}"><span class="route-index">{number} / {label}</span>'
        f'<span class="route-glyph" aria-hidden="true">↗</span><h3>{title}</h3><p>{desc}</p>'
        '<span class="route-arrow">开始阅读 <span aria-hidden="true">→</span></span></a>'
        for number, title, desc, url, label, css in routes
    )
    chapter_cards = "".join(
        f'<a class="chapter-card" href="{e(page["url"])}">'
        f'<span class="chapter-number">{page["number"]:02d}</span>'
        f'<span class="chapter-copy"><strong>{e(page["title"])}</strong><small>{e(page["desc"])}</small></span>'
        '<span class="chapter-arrow" aria-hidden="true">↗</span></a>'
        for page in chapters
    )
    extra_cards = "".join(
        f'<a class="extra-card" href="{e(page["url"])}"><span>{e(page["group"])}</span>'
        f'<strong>{e(page["title"])}</strong><small>{e(page["desc"])}</small>'
        '<b aria-hidden="true">↗</b></a>'
        for page in extras
    )
    body = f'''
      <div class="home-wrap">
        <section class="home-hero">
          <div class="hero-copy"><span class="hero-eyebrow"><span class="eyebrow-line"></span> AI INFRA / LEARNING NOTES</span>
            <h1>从 GPU Kernel，<br><em>走到推理系统。</em></h1>
            <p>沿着「原理 → 图解 → 实现 → 源码 → Profiling → Debug」的路径，学习 AI Infra 的完整运行链路。</p>
            <div class="hero-actions"><a class="button button-primary" href="ch00.html">开始学习 <span aria-hidden="true">↗</span></a><a class="button button-secondary" href="#chapters">浏览章节 <span aria-hidden="true">↓</span></a></div>
          </div>
          <div class="hero-visual" aria-label="硬件层、算子层、服务层和诊断层学习地图">
            <div class="visual-header"><span>INFERENCE STACK</span><span class="visual-pulse"></span></div>
            <div class="visual-layers"><div><span>04</span><strong>诊断层</strong><small>Profiling · Debug · Bug</small></div>
              <div><span>03</span><strong>服务层</strong><small>调度 · 缓存 · 并行</small></div>
              <div><span>02</span><strong>算子层</strong><small>Attention · GEMM · Kernel</small></div>
              <div><span>01</span><strong>硬件层</strong><small>GPU · CUDA · Memory</small></div></div>
            <div class="visual-footer"><span>从原理到实现</span><span>↓</span></div>
          </div>
        </section>
        <div class="home-stats"><div><strong>{len(chapters):02d}</strong><span>系统章节</span></div><div><strong>{len(list((ROOT / "figures").glob("*.png"))):02d}</strong><span>原理配图</span></div><div><strong>04</strong><span>学习阶段</span></div></div>
        <section class="home-section" id="roadmap"><div class="section-heading"><span>01 / LEARNING PATH</span><h2>从哪里开始？</h2><p>按层次推进，也可以直接跳到你关心的主题。</p></div><div class="route-grid">{route_cards}</div></section>
        <section class="home-section" id="chapters"><div class="section-heading"><span>02 / HANDBOOK</span><h2>AI Infra 工程实践指南</h2><p>第 0–18 章沿同一条请求链路，连接硬件、算子、服务、性能、调试与模型适配。</p></div><div class="chapter-grid">{chapter_cards}</div></section>
        <section class="home-section" id="more"><div class="section-heading"><span>03 / MORE TO EXPLORE</span><h2>继续探索</h2><p>项目实践与技术写作笔记。</p></div><div class="extra-grid">{extra_cards}</div></section>
      </div>'''
    return shell("首页", "系统学习 LLM 推理加速的中文笔记", body, pages, "index.html", page_class="home-page")


def write_assets():
    shutil.copytree(HERE / "assets", OUT / "assets")
    light = HtmlFormatter(style="friendly").get_style_defs(".codehilite")
    dark = HtmlFormatter(style="monokai").get_style_defs('[data-theme="dark"] .codehilite')
    (OUT / "assets" / "pygments.css").write_text(light + "\n" + dark, encoding="utf-8")
    (OUT / ".nojekyll").touch()


def build():
    pages = load_pages()
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    write_assets()
    shutil.copytree(ROOT / "figures", OUT / "figures", ignore=shutil.ignore_patterns("*.svg"))
    shutil.copytree(
        ROOT / "examples", OUT / "examples",
        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"),
    )
    (OUT / "index.html").write_text(render_home(pages), encoding="utf-8")
    for page in pages:
        (OUT / page["url"]).write_text(render_article(page, pages), encoding="utf-8")
    index = [
        {"title": page["title"], "url": page["url"], **section}
        for page in pages for section in page["sections"]
    ]
    (OUT / "search-index.json").write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    print(f"Built {len(pages) + 1} pages in {OUT}")


if __name__ == "__main__":
    build()
