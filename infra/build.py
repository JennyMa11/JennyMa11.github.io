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
    0: "四层技术地图、核心指标与推理瓶颈",
    1: "Prefill、Decode、KV Cache 与显存计算",
    2: "量化、注意力结构与模型压缩",
    3: "FlashAttention、PagedAttention 与 GPU 算子",
    4: "CUDA Graph、编译优化与异步执行",
    5: "连续批处理、缓存、并行与 PD 分离",
    6: "KV Cache 的开销、设计与缩减方法",
    7: "MoE Router、专家选择与系统挑战",
    8: "Draft-then-Verify 与接受采样",
    9: "分层结论与高频面试问题",
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
    pages = []
    for index, start in enumerate(starts):
        end = starts[index + 1] if index + 1 < len(starts) else len(lines)
        title = lines[start][2:].strip()
        match = re.match(r"第 (\d+) 章", title)
        number = int(match.group(1)) if match else None
        url = f"ch{number:02d}.html" if number is not None else "appendix.html"
        desc = CHAPTER_DESCS.get(number, "术语与补充资料")
        pages.append(make_page(
            url, "知识手册", title, desc, "\n".join(lines[start + 1:end]), number
        ))

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
  <script>try{{document.documentElement.dataset.theme=localStorage.getItem("infra-theme")||"light"}}catch(e){{}}</script>
</head>
<body class="{page_class}">
  <div id="progress" aria-hidden="true"></div>
  <header class="topbar">
    <button class="icon-button menu-button" id="menu-button" aria-label="打开目录" aria-expanded="false" aria-controls="sidebar">☰</button>
    <a class="brand" href="index.html"><span class="brand-mark">I<span>·</span></span><span>Infra<span class="brand-light"> Notes</span></span></a>
    <span class="topbar-divider"></span><span class="topbar-context">推理系统学习手册</span>
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
        ("01", "建立地图", "先看推理全局，再理解一次请求的生命周期。", "ch00.html", "第 0–1 章", "route-map"),
        ("02", "深入优化", "从量化、注意力算子到编译执行，逐层寻找瓶颈。", "ch02.html", "第 2–4 章", "route-kernel"),
        ("03", "进入系统", "理解调度、KV Cache、MoE 与投机解码。", "ch05.html", "第 5–8 章", "route-system"),
        ("04", "回顾与实战", "用速查表和项目问答巩固知识。", "ch09.html", "第 9 章 + 项目实测", "route-practice"),
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
            <h1>把复杂的推理系统，<br><em>讲清楚。</em></h1>
            <p>从 GPU 原理到服务架构，沿着「原理 → 实现 → 追问」的路径，系统理解 LLM 推理加速。</p>
            <div class="hero-actions"><a class="button button-primary" href="ch00.html">开始学习 <span aria-hidden="true">↗</span></a><a class="button button-secondary" href="#chapters">浏览章节 <span aria-hidden="true">↓</span></a></div>
          </div>
          <div class="hero-visual" aria-label="模型层、算子层、执行层和系统层学习地图">
            <div class="visual-header"><span>INFERENCE STACK</span><span class="visual-pulse"></span></div>
            <div class="visual-layers"><div><span>04</span><strong>系统层</strong><small>调度 · 缓存 · 并行</small></div>
              <div><span>03</span><strong>执行层</strong><small>编译 · Graph · Overlap</small></div>
              <div><span>02</span><strong>算子层</strong><small>Attention · Kernel</small></div>
              <div><span>01</span><strong>模型层</strong><small>量化 · 架构优化</small></div></div>
            <div class="visual-footer"><span>从原理到实现</span><span>↓</span></div>
          </div>
        </section>
        <div class="home-stats"><div><strong>{len(chapters):02d}</strong><span>系统章节</span></div><div><strong>{len(list((ROOT / "figures").glob("*.png"))):02d}</strong><span>原理配图</span></div><div><strong>04</strong><span>学习阶段</span></div></div>
        <section class="home-section" id="roadmap"><div class="section-heading"><span>01 / LEARNING PATH</span><h2>从哪里开始？</h2><p>按层次推进，也可以直接跳到你关心的主题。</p></div><div class="route-grid">{route_cards}</div></section>
        <section class="home-section" id="chapters"><div class="section-heading"><span>02 / HANDBOOK</span><h2>推理加速知识手册</h2><p>十个章节，串起模型、算子、执行与系统层的关键概念。</p></div><div class="chapter-grid">{chapter_cards}</div></section>
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
