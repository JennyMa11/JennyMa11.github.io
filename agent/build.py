#!/usr/bin/env python3
"""Build the Agent handbook into a self-contained static documentation site."""

import html
import json
import re
import shutil
from pathlib import Path

import markdown
from markdown.extensions.toc import slugify_unicode
from pygments.formatters import HtmlFormatter

ROOT = Path(__file__).resolve().parent
OUT = ROOT.parent / "public" / "agent"
INFRA_ASSETS = ROOT.parent / "infra" / "assets"
GROUPS = [
    ("基础与经典方法", range(0, 4)),
    ("核心能力", range(4, 7)),
    ("系统工程", range(7, 11)),
    ("前沿与求职", range(11, 14)),
]
DESCRIPTIONS = [
    "沿运费修复任务拆开策略、工具、环境与完成验收",
    "用两房间清扫案例理解反射、内部状态、搜索与效用",
    "运行真实修改与测试，逐步分析正常、超时和假完成轨迹",
    "从任务依赖到计划修订，用候选补丁理解反思与搜索",
    "设计工具契约，处理参数、资源范围、版本冲突与副作用",
    "手算 Recall@k，分析证据、引用、版本与长期记忆",
    "分配上下文预算，编写状态摘要并核对中断后的环境",
    "设计子任务契约、关键路径、隔离工作区与合并验收",
    "沿请求与任务生命周期理解 MCP、A2A 和应用责任",
    "建立可重置任务、可靠验收、重复试验与机制消融",
    "拆解注入路径，验证权限边界并用轨迹复盘失败",
    "Harness、Context Engineering、Skills、恢复与 Agentic RL",
    "从验收题到四个项目里程碑，整理复现、指标与面试讲述",
    "OpenCode、OpenClaw、Hermes、Claude Code 与 Codex 技术拆解和对比",
]


def esc(value):
    return html.escape(str(value), quote=True)


def strip_html(value):
    return re.sub(r"\s+", " ", html.unescape(re.sub(r"<[^>]+>", " ", value))).strip()


def md_to_html(source):
    return markdown.markdown(
        source,
        extensions=["fenced_code", "tables", "codehilite", "toc"],
        extension_configs={
            "codehilite": {"guess_lang": False, "css_class": "codehilite"},
            "toc": {"slugify": slugify_unicode, "permalink": False},
        },
    )


def get_sources():
    data = json.loads((ROOT / "source_summaries.json").read_text(encoding="utf-8"))
    sources = {}
    ids = set()
    for source in data["sources"]:
        for field in ("id", "url", "group", "title", "kind", "summary", "takeaway", "note"):
            if not isinstance(source.get(field), str) or not source[field].strip():
                raise ValueError(f"Missing source field {field}: {source.get('url')}")
        if source["url"] in sources or source["id"] in ids:
            raise ValueError(f"Duplicate source: {source['url']}")
        sources[source["url"]] = source
        ids.add(source["id"])
    return sources, data["reviewed_on"]


def source_card(source, anchor, heading="h4"):
    return (f'<blockquote class="source-note" id="{esc(anchor)}">'
            f'<{heading}>{esc(source["title"])}</{heading}>'
            f'<p class="source-meta">{esc(source["kind"])}</p>'
            f'<p><strong>主要内容：</strong>{esc(source["summary"])}</p>'
            f'<p><strong>核心要点：</strong>{esc(source["takeaway"])}</p>'
            f'<p><strong>阅读提示：</strong>{esc(source["note"])}</p>'
            f'<p><a href="{esc(source["url"])}">阅读原文 ↗</a></p></blockquote>')


def add_source_notes(body, sources):
    """Keep every external reference beside a visible, centrally maintained summary."""
    parts = []
    for index, part in enumerate(re.split(r'(?=<h2 id=")', body)):
        used = {}

        def annotate(match):
            url = html.unescape(match.group(1))
            if url not in sources:
                raise ValueError(f"External reference needs a summary: {url}")
            source = sources[url]
            anchor = f'{source["id"]}-section-{index}'
            used[url] = (source, anchor)
            return (match.group(0) + f' <a class="source-jump" href="#{anchor}"'
                    f' aria-label="查看 {esc(source["title"])} 的内容摘要">[导读]</a>')

        part = re.sub(r'<a\b[^>]*href="(https?://[^\"]+)"[^>]*>.*?</a>', annotate, part, flags=re.S)
        if used:
            cards = "".join(source_card(source, anchor) for source, anchor in used.values())
            part += ('<aside class="source-notes" aria-label="本节引用导读">'
                     '<h3>本节引用导读</h3>' + cards + '</aside>')
        parts.append(part)
    return "".join(parts)


def get_pages():
    sources, _ = get_sources()
    source = (ROOT / "guide_zh.md").read_text(encoding="utf-8")
    lines = source.splitlines()
    starts = [i for i, line in enumerate(lines) if re.match(r"^# 第 \d+ 章", line)]
    if len(starts) != len(DESCRIPTIONS):
        raise ValueError(f"Expected {len(DESCRIPTIONS)} chapters, got {len(starts)}")
    pages = []
    for index, start in enumerate(starts):
        match = re.match(r"# 第 (\d+) 章 (.+)", lines[start])
        if not match or int(match.group(1)) != index:
            raise ValueError(f"Chapter order mismatch at line {start + 1}")
        body_source = "\n".join(lines[start + 1:starts[index + 1] if index + 1 < len(starts) else len(lines)])
        if index == 0:
            body_source = "\n".join(lines[1:start]) + "\n\n" + body_source
        body = add_source_notes(md_to_html(body_source), sources)
        headings = [(m.group(1), strip_html(m.group(2))) for m in re.finditer(
            r'<h2 id="([^"]+)">(.*?)</h2>', body, flags=re.S
        )]
        sections = []
        for part in re.split(r'(?=<h2 id=")', body):
            heading = re.match(r'<h2 id="([^"]+)">(.*?)</h2>', part, flags=re.S)
            text = strip_html(part)
            if text:
                sections.append({
                    "anchor": heading.group(1) if heading else "",
                    "heading": strip_html(heading.group(2)) if heading else match.group(2),
                    "text": text,
                })
        pages.append({
            "number": index,
            "title": f"第 {index} 章 {match.group(2)}",
            "desc": DESCRIPTIONS[index],
            "url": f"ch{index:02d}.html",
            "body": body,
            "headings": headings,
            "sections": sections,
            "minutes": max(1, round(len(strip_html(body)) / 600)),
        })
    return pages


def navigation(pages, active):
    home_class = " active" if active == "index.html" else ""
    home_current = ' aria-current="page"' if home_class else ""
    nav = [f'<a class="nav-link nav-home{home_class}" href="index.html"'
           f'{home_current}>'
           '<span class="nav-number nav-symbol">⌂</span><span>首页 · 学习地图</span></a>']
    selected = active == "sources.html"
    current = ' aria-current="page"' if selected else ""
    nav.append(f'<a class="nav-link{" active" if selected else ""}" href="sources.html"{current}>'
               '<span class="nav-number nav-symbol">≡</span><span>引用导读 · 内容汇总</span></a>')
    for group, numbers in GROUPS:
        items = []
        for page in pages:
            if page["number"] not in numbers:
                continue
            selected = page["url"] == active
            current = ' aria-current="page"' if selected else ""
            items.append(
                f'<a class="nav-link{" active" if selected else ""}" href="{page["url"]}"'
                f'{current}>'
                f'<span class="nav-number">{page["number"]:02d}</span>'
                f'<span>{esc(page["title"])}</span></a>'
            )
        nav.append(f'<div class="nav-group"><p class="nav-heading">{group}</p>{"".join(items)}</div>')
    return "".join(nav)


def shell(title, description, body, pages, active, toc="", page_class=""):
    return f'''<!doctype html>
<html lang="zh-CN" data-theme="light">
<head>
  <meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
  <meta name="description" content="{esc(description)}"><meta name="theme-color" content="#f8faf8">
  <title>{esc(title)} · Agent Guide</title>
  <link rel="stylesheet" href="assets/pygments.css"><link rel="stylesheet" href="assets/style.css">
  <link rel="stylesheet" href="assets/sources.css">
  <script>try{{const saved=localStorage.getItem("theme");document.documentElement.dataset.theme=saved||(matchMedia("(prefers-color-scheme: dark)").matches?"dark":"light")}}catch(e){{}}</script>
</head>
<body class="{page_class}">
  <div id="progress" aria-hidden="true"></div>
  <header class="topbar">
    <button class="icon-button menu-button" id="menu-button" aria-label="打开目录" aria-expanded="false" aria-controls="sidebar">☰</button>
    <a class="brand" href="index.html"><span class="brand-mark">A<span>·</span></span><span>Agent<span class="brand-light"> Guide</span></span></a>
    <span class="topbar-divider"></span><span class="topbar-context">从经典智能体到前沿工程</span>
    <a class="back-link" href="/">← 返回主博客</a>
    <div class="topbar-actions"><div class="searchbox" id="searchbox">
      <label class="visually-hidden" for="search-input">搜索指南</label>
      <input id="search-input" type="search" placeholder="搜索指南..." autocomplete="off" aria-controls="search-results" aria-expanded="false">
      <kbd class="search-shortcut">⌘ K</kbd><div id="search-results" class="search-results" role="listbox"></div>
    </div><button class="icon-button theme-button" id="theme-toggle" aria-label="切换深色模式" title="切换深色模式">◐</button></div>
  </header>
  <div class="sidebar-scrim" id="sidebar-scrim"></div>
  <aside class="sidebar" id="sidebar" aria-label="站点目录"><div class="sidebar-inner">
    <div class="sidebar-caption">AGENT / LEARNING GUIDE</div><nav>{navigation(pages, active)}</nav>
    <div class="sidebar-bottom"><a href="/infra/">→ 阅读 Infra Guide</a><br><a href="/">← 返回 Jenny Ma 主博客</a><br><span class="status-dot"></span>基于原理与来源持续更新</div>
  </div></aside>
  <div class="site-layout"><main class="content" id="main-content">{body}
    <footer class="site-footer"><span>Agent Guide · 从反馈闭环走向可靠系统</span><span>基于 Markdown 构建</span></footer>
  </main>{toc}</div><script src="assets/app.js" defer></script>
</body></html>'''


def article(page, pages):
    toc = ""
    if page["headings"]:
        links = "".join(f'<a href="#{esc(anchor)}">{esc(text)}</a>' for anchor, text in page["headings"])
        toc = f'<aside class="page-toc" aria-label="本页目录"><p>本页目录</p><nav>{links}</nav></aside>'
    index = page["number"]
    prev_link = (f'<a class="pagination-link prev" href="{pages[index - 1]["url"]}">'
                 f'<span>上一篇</span><strong>{esc(pages[index - 1]["title"])}</strong></a>') if index else ""
    next_link = (f'<a class="pagination-link next" href="{pages[index + 1]["url"]}">'
                 f'<span>下一篇</span><strong>{esc(pages[index + 1]["title"])}</strong></a>') if index + 1 < len(pages) else ""
    body = f'''<div class="article-wrap"><div class="breadcrumb"><a href="index.html">首页</a><span>／</span>知识手册 · 第 {index:02d} 章</div>
    <article class="article"><header class="article-header"><span class="eyebrow">AGENT GUIDE / READING NOTES</span>
      <h1>{esc(page["title"])}</h1><p class="article-lead">{esc(page["desc"])}</p>
      <div class="article-meta"><span>约 {page["minutes"]} 分钟阅读</span><span class="meta-dot">·</span><span>持续更新</span></div>
    </header><div class="prose">{page["body"]}</div></article>
    <nav class="pagination" aria-label="上一篇与下一篇">{prev_link}{next_link}</nav></div>'''
    return shell(page["title"], page["desc"], body, pages, page["url"], toc, "article-page")


def home(pages):
    routes = [
        ("01", "基础与经典", "闭环、传统策略与 ReAct", "ch00.html", "第 0–3 章", "route-map"),
        ("02", "核心能力", "工具、检索、记忆与上下文", "ch04.html", "第 4–6 章", "route-kernel"),
        ("03", "系统工程", "多 Agent、协议、评测与安全", "ch07.html", "第 7–10 章", "route-system"),
        ("04", "前沿与求职", "Harness、前沿工程与五家 Agent 对比", "ch11.html", "第 11–13 章", "route-practice"),
    ]
    route_cards = "".join(
        f'<a class="route-card {css}" href="{url}"><span class="route-index">{number} / {span}</span>'
        f'<span class="route-glyph" aria-hidden="true">↗</span><h3>{name}</h3><p>{desc}</p>'
        '<span class="route-arrow">开始阅读 <span aria-hidden="true">→</span></span></a>'
        for number, name, desc, url, span, css in routes
    )
    chapters = "".join(
        f'<a class="chapter-card" href="{page["url"]}"><span class="chapter-number">{page["number"]:02d}</span>'
        f'<span class="chapter-copy"><strong>{esc(page["title"])}</strong><small>{esc(page["desc"])}</small></span>'
        '<span class="chapter-arrow" aria-hidden="true">↗</span></a>'
        for page in pages
    )
    body = f'''<div class="home-wrap"><section class="home-hero">
      <div class="hero-copy"><span class="hero-eyebrow"><span class="eyebrow-line"></span> AGENT / LEARNING GUIDE</span>
        <h1>从反馈闭环，<br><em>走到可靠 Agent。</em></h1>
        <p>沿着「传统 Agent → 工具循环 → 检索与协作 → Harness 与前沿工程」的路径，学习 Agent 的原理、实现和面试重点。</p>
        <div class="hero-actions"><a class="button button-primary" href="ch00.html">开始学习 <span aria-hidden="true">↗</span></a>
          <a class="button button-secondary" href="#chapters">浏览章节 <span aria-hidden="true">↓</span></a></div>
      </div><div class="hero-visual" aria-label="Agent 学习路线四层结构">
        <div class="visual-header"><span>AGENT STACK</span><span class="visual-pulse"></span></div>
        <div class="visual-layers"><div><span>04</span><strong>Harness 工程</strong><small>恢复 · 验收 · 权限</small></div>
          <div><span>03</span><strong>系统能力</strong><small>检索 · 上下文 · 协作</small></div>
          <div><span>02</span><strong>LLM Agent</strong><small>工具 · 规划 · 反思</small></div>
          <div><span>01</span><strong>传统智能体</strong><small>观察 · 决策 · 反馈</small></div></div>
        <div class="visual-footer"><span>从原理到可验证的项目</span><span>↓</span></div>
      </div></section>
      <div class="home-stats"><div><strong>{len(pages):02d}</strong><span>系统章节</span></div>
        <div><strong>{len(list((ROOT / 'figures').glob('*.svg'))):02d}</strong><span>原创原理图</span></div>
        <div><strong>04</strong><span>学习阶段</span></div></div>
      <section class="home-section" id="roadmap"><div class="section-heading"><span>01 / LEARNING PATH</span>
        <h2>从哪里开始？</h2><p>按基础、能力、系统和前沿逐步推进。</p></div><div class="route-grid">{route_cards}</div></section>
      <section class="home-section" id="chapters"><div class="section-heading"><span>02 / HANDBOOK</span>
        <h2>Agent 工程实践指南</h2><p>{len(pages)} 章从传统 Agent 走到前沿工程、实习项目与产品技术对比。所有外部引用均附内容导读，亦可查看 <a href="sources.html">引用内容汇总</a>。</p></div>
        <div class="chapter-grid">{chapters}</div></section>
      <section class="home-section" id="more"><div class="section-heading"><span>03 / RELATED GUIDE</span>
        <h2>继续阅读</h2><p>模型推理的底层实现与系统性能，见 Infra Guide。</p></div>
        <div class="extra-grid"><a class="extra-card" href="/infra/"><span>AI INFRA</span><strong>Infra 工程实践指南</strong>
        <small>CUDA Kernel、Attention、KV Cache 与 LLM Serving</small><b aria-hidden="true">↗</b></a></div></section></div>'''
    return shell("首页", "从传统智能体到 Agent 工程实践的中文学习指南", body, pages, "index.html", page_class="home-page")


def sources_page(sources, reviewed_on, pages):
    overview = md_to_html((ROOT / "sources_overview_zh.md").read_text(encoding="utf-8"))
    groups = {}
    search = [{"title": "引用导读 · 内容汇总", "url": "sources.html", "anchor": "",
               "heading": "这些资料合起来讲了什么", "text": strip_html(overview)}]
    for source in sources.values():
        groups.setdefault(source["group"], []).append(source)
        search.append({"title": "引用导读 · 内容汇总", "url": "sources.html", "anchor": source["id"],
                       "heading": source["title"], "text": " ".join(source[k] for k in
                       ("kind", "summary", "takeaway", "note"))})
    content = [overview]
    toc_links = ['<a href="#reading-overview">这些资料合起来讲了什么</a>']
    for index, (group, entries) in enumerate(groups.items()):
        anchor = f"source-group-{index}"
        toc_links.append(f'<a href="#{anchor}">{esc(group)} · {len(entries)} 项</a>')
        content.append(f'<h2 id="{anchor}">{esc(group)}</h2>')
        content.extend(source_card(source, source["id"], "h3") for source in entries)
    body = ('<div class="article-wrap"><div class="breadcrumb"><a href="index.html">首页</a>'
            '<span>／</span>引用导读</div><article class="article"><header class="article-header">'
            '<span class="eyebrow">AGENT GUIDE / SOURCE NOTES</span><h1>引用导读 · 内容汇总</h1>'
            f'<p class="article-lead">{len(sources)} 份资料的主要内容、核心要点与阅读提示。先读综合总结，再按主题查看每份原文讲什么。</p>'
            f'<div class="article-meta">资料核对：{esc(reviewed_on)} · 导读为中文转述</div></header>'
            '<div class="prose">' + "".join(content) + '</div></article></div>')
    toc = ('<aside class="page-toc" aria-label="本页目录"><p>本页目录</p><nav>'
           + "".join(toc_links) + '</nav></aside>')
    return shell("引用导读 · 内容汇总", "Agent 指南全部引用来源的中文内容摘要与综合总结", body,
                 pages, "sources.html", toc, "article-page"), search


def build():
    pages = get_pages()
    sources, reviewed_on = get_sources()
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    shutil.copytree(INFRA_ASSETS, OUT / "assets")
    shutil.copyfile(ROOT / "sources.css", OUT / "assets" / "sources.css")
    shutil.copytree(ROOT / "figures", OUT / "figures")
    shutil.copytree(ROOT / "examples", OUT / "examples", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    light = HtmlFormatter(style="friendly").get_style_defs(".codehilite")
    dark = HtmlFormatter(style="monokai").get_style_defs('[data-theme="dark"] .codehilite')
    (OUT / "assets" / "pygments.css").write_text(light + "\n" + dark, encoding="utf-8")
    (OUT / ".nojekyll").touch()
    (OUT / "index.html").write_text(home(pages), encoding="utf-8")
    for page in pages:
        (OUT / page["url"]).write_text(article(page, pages), encoding="utf-8")
    index = [{"title": p["title"], "url": p["url"], **s} for p in pages for s in p["sections"]]
    source_html, source_search = sources_page(sources, reviewed_on, pages)
    (OUT / "sources.html").write_text(source_html, encoding="utf-8")
    index.extend(source_search)
    (OUT / "search-index.json").write_text(json.dumps(index, ensure_ascii=False), encoding="utf-8")
    print(f"Built {len(pages) + 2} pages with {len(sources)} source summaries in {OUT}")


if __name__ == "__main__":
    build()
