#!/usr/bin/env python3
"""Build the Infra notes as a self-contained static documentation site."""

import html
import hashlib
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
    7: "自回归生成、KV 生命周期、性能指标与 Roofline",
    8: "分页、连续批处理、前缀缓存、图与编译",
    9: "vLLM 快速入门、V1 架构、调度源码与调优",
    10: "量化原理、低比特 Kernel、KV 压缩与部署选型",
    11: "精确采样、Draft Tree、收益边界与 vLLM 实战",
    12: "TP、PP、DP、EP 与 Ray 多节点实践",
    13: "阶段互扰、KV 传输、Goodput 与资源池配比",
    14: "结构化输出、工具调用、LoRA、多模态与采样",
    15: "指标、Benchmark、Nsight 与性能回归门禁",
    16: "Kubernetes、监控、路由、扩缩容与容量成本",
    17: "选型决策、技术叠加与端到端交付",
    18: "端侧约束、Runtime、ExecuTorch 与稳态性能",
    19: "逐层逐 Token 对比与第一处分歧定位",
    20: "CUDA、Kernel、Serving 与分布式故障模式",
    21: "解释、实现和调试能力验收",
    22: "GPU/NPU 软件栈、算子迁移、精度定位与验收",
    23: "DeepSeek V3.2/V4/V4.1 技术报告、长上下文缓存与同期架构",
}

HANDBOOK = ROOT / "handbook"
STYLE_VERSION = hashlib.sha256((HERE / "assets" / "style.css").read_bytes()).hexdigest()[:8]
THEMES = [
    ("学习地图", range(0, 1)),
    ("GPU 与算子", range(1, 4)),
    ("模型与缓存", range(4, 7)),
    ("推理基础与引擎", range(7, 10)),
    ("加速与分布式", range(10, 14)),
    ("生产服务与端侧", range(14, 19)),
    ("排障、适配与前沿报告", range(19, 24)),
]


def theme_for(number):
    if number is None:
        return "资料附录"
    return next(name for name, numbers in THEMES if number in numbers)


def title_and_body(path):
    source = read(path)
    match = re.match(r"^# (.+)\n", source)
    if not match:
        raise ValueError(f"Expected a level-one title in {path}")
    return match.group(1).strip(), source[match.end():].strip()


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
    for part in re.split(r'(?=<h[23] id=")', body):
        heading = re.match(r'<h[23] id="([^"]+)">(.*?)</h[23]>', part, flags=re.S)
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
    pages = []
    chapter_dirs = sorted(HANDBOOK.glob("ch[0-9][0-9]")) + [HANDBOOK / "appendix"]
    for chapter_dir in chapter_dirs:
        title, source = title_and_body(chapter_dir / "index.md")
        match = re.match(r"第 (\d+) 章", title)
        number = int(match.group(1)) if match else None
        theme = theme_for(number)
        url = f"ch{number:02d}.html" if number is not None else "appendix.html"
        desc = CHAPTER_DESCS.get(number, "术语与补充资料")
        overview = make_page(url, theme, title, desc, source, number)
        overview.update(kind="overview", chapter=number, short_title=re.sub(r"^第 \d+ 章\s*", "", title))
        pages.append(overview)
        for path in sorted(chapter_dir.glob("[0-9][0-9]-*.md")):
            article_title, article_source = title_and_body(path)
            section = int(path.name[:2])
            article_url = f"ch{number:02d}-{section:02d}.html" if number is not None else f"appendix-{section:02d}.html"
            article = make_page(article_url, theme, article_title, desc, article_source, number)
            article.update(
                kind="article", chapter=number, short_title=re.sub(r"^(?:\d+(?:\.\d+)*|附录 [A-Z])\s*", "", article_title).strip(" ·"),
                old_anchor=slugify_unicode(article_title, "-"),
            )
            pages.append(article)

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
        page = make_page(url, group, title, desc, source)
        page.update(kind="extra", chapter=None, short_title=title)
        pages.append(page)
    return pages


def e(value):
    return html.escape(str(value), quote=True)


def navigation(pages, active):
    groups = []
    for group in [name for name, _ in THEMES] + ["资料附录"]:
        chapters = [page for page in pages if page["group"] == group and page["kind"] == "overview"]
        chapter_items = []
        for overview in chapters:
            children = [page for page in pages if page["kind"] == "article" and page["chapter"] == overview["chapter"]]
            if overview["chapter"] is None:
                children = [page for page in children if page["group"] == "资料附录"]
            is_open = active == overview["url"] or any(active == item["url"] for item in children)
            links = []
            for page in [overview] + children:
                is_active = page["url"] == active
                current_attr = ' aria-current="page"' if is_active else ""
                label = "目录" if page["kind"] == "overview" else page["title"].split(" ", 1)[0]
                links.append(
                    f'<a class="nav-link{" active" if is_active else ""}" href="{e(page["url"])}"{current_attr}>'
                    f'<span class="nav-number">{e(label)}</span><span>{e(page["short_title"])}</span></a>'
                )
            chapter_label = f'{overview["number"]:02d}' if overview["number"] is not None else '↗'
            chapter_items.append(
                f'<details class="nav-chapter"{" open" if is_open else ""}>'
                f'<summary><span class="nav-number">{chapter_label}</span><span>{e(overview["short_title"])}</span></summary>'
                f'<div class="nav-article-list">{"".join(links)}</div></details>'
            )
        groups.append(f'<div class="nav-group"><p class="nav-heading">{e(group)}</p>{"".join(chapter_items)}</div>')
    for group in ("项目实测", "延伸笔记"):
        links = []
        for page in pages:
            if page["group"] != group:
                continue
            is_active = page["url"] == active
            current_attr = ' aria-current="page"' if is_active else ""
            links.append(
                f'<a class="nav-link{" active" if is_active else ""}" href="{e(page["url"])}"{current_attr}>'
                f'<span class="nav-number nav-symbol">↗</span><span>{e(page["title"])}</span></a>'
            )
        groups.append(f'<div class="nav-group"><p class="nav-heading">{e(group)}</p>{"".join(links)}</div>')
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
  <link rel="stylesheet" href="assets/style.css?v={STYLE_VERSION}">
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
    article_list = ""
    if page["kind"] == "overview":
        children = [item for item in pages if item["kind"] == "article" and item["chapter"] == page["chapter"] and item["group"] == page["group"]]
        cards = "\n".join(
            f'<li><a class="article-list-card" id="{e(item["old_anchor"])}" href="{e(item["url"])}">'
            f'<strong>{e(item["title"])}</strong><br><small>约 {item["minutes"]} 分钟阅读</small>'
            '<span aria-hidden="true">↗</span></a></li>'
            for item in children
        )
        article_list = f'<section class="article-list" aria-label="本章小文章"><h2>本章小文章</h2><ul class="article-list-grid">{cards}</ul></section>'
    body = f'''
      <div class="article-wrap">
        <div class="breadcrumb"><a href="index.html">首页</a><span>／</span>{e(page["group"])}{number}</div>
        <article class="article">
          <header class="article-header"><span class="eyebrow">{e(page["group"])} / READING NOTES</span>
            <h1>{e(page["title"])}</h1><p class="article-lead">{e(page["desc"])}</p>
            <div class="article-meta"><span>约 {page["minutes"]} 分钟阅读</span><span class="meta-dot">·</span><span>持续更新</span></div>
          </header>
          <div class="prose">{page["body"]}</div>{article_list}
        </article>{pagination}
      </div>'''
    return shell(page["title"], page["desc"], body, pages, page["url"], toc, "article-page")


def render_home(pages):
    chapters = [page for page in pages if page["kind"] == "overview" and page["number"] is not None]
    articles = [page for page in pages if page["kind"] == "article"]
    extras = [page for page in pages if page["kind"] == "extra"]
    routes = [
        ("01", "硬件与算子", "从 GPU 执行模型写到 GEMM、Softmax 和 Triton。", "ch01.html", "第 1–3 章", "route-map"),
        ("02", "模型与缓存", "实现 Transformer 算子、Attention 和分页 KV。", "ch04.html", "第 4–6 章", "route-kernel"),
        ("03", "推理优化模块", "按 12 章大纲学习引擎、量化、投机、部署与端侧。", "ch07.html", "第 7–18 章", "route-system"),
        ("04", "排障、适配与前沿报告", "逐层定位、故障复盘、GPU/NPU 适配和 DeepSeek 报告精读。", "ch19.html", "第 19–23 章", "route-practice"),
    ]
    route_cards = "".join(
        f'<a class="route-card {css}" href="{url}"><span class="route-index">{number} / {label}</span>'
        f'<span class="route-glyph" aria-hidden="true">↗</span><h3>{title}</h3><p>{desc}</p>'
        '<span class="route-arrow">开始阅读 <span aria-hidden="true">→</span></span></a>'
        for number, title, desc, url, label, css in routes
    )
    chapter_groups = "".join(
        f'<div class="chapter-topic"><h3>{e(group)}</h3><div class="chapter-grid">' + "".join(
            f'<a class="chapter-card" href="{e(page["url"])}">'
            f'<span class="chapter-number">{page["number"]:02d}</span>'
            f'<span class="chapter-copy"><strong>{e(page["short_title"])}</strong><small>{e(page["desc"])}</small></span>'
            '<span class="chapter-arrow" aria-hidden="true">↗</span></a>'
            for page in chapters if page["group"] == group
        ) + '</div></div>'
        for group, _ in THEMES
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
        <div class="home-stats"><div><strong>{len(chapters):02d}</strong><span>章节目录</span></div><div><strong>{len(articles):02d}</strong><span>独立小文章</span></div><div><strong>{len(THEMES):02d}</strong><span>主题分组</span></div></div>
        <section class="home-section" id="roadmap"><div class="section-heading"><span>01 / LEARNING PATH</span><h2>从哪里开始？</h2><p>按层次推进，也可以直接跳到你关心的主题。</p></div><div class="route-grid">{route_cards}</div></section>
        <section class="home-section" id="chapters"><div class="section-heading"><span>02 / HANDBOOK</span><h2>按主题查阅</h2><p>先选大主题和章节，再打开独立小文章。第 7–18 章沿用 AIInfraGuide 模块四的推理优化大纲。</p></div>{chapter_groups}</section>
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
    urls = [page["url"] for page in pages]
    if len(urls) != len(set(urls)):
        duplicates = sorted({url for url in urls if urls.count(url) > 1})
        raise ValueError(f"Duplicate article numbers or chapter URLs: {', '.join(duplicates)}")
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
