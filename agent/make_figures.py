#!/usr/bin/env python3
"""Create source SVG diagrams for the Agent handbook without external assets."""

from html import escape
from math import atan2, cos, sin
from pathlib import Path
import re

OUT = Path(__file__).resolve().parent / "figures"
OUT.mkdir(exist_ok=True)


def card(x, y, w, h, label, detail, tone="teal"):
    colors = {
        "teal": ("#e1f4ee", "#12826e"),
        "blue": ("#e8f0ff", "#4771ba"),
        "orange": ("#fff0df", "#c77732"),
        "violet": ("#f0eafd", "#8260bd"),
    }
    fill, stroke = colors[tone]
    return (f'<g><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="16" '
            f'fill="{fill}" stroke="{stroke}" stroke-width="2"/>'
            f'<text x="{x + 20}" y="{y + 34}" class="label">{escape(label)}</text>'
            f'<text x="{x + 20}" y="{y + 61}" class="detail">{escape(detail)}</text></g>')


def arrow(x1, y1, x2, y2, label=""):
    result = (f'<path d="M{x1} {y1} L{x2} {y2}" stroke="#56716a" stroke-width="2.5" '
              f'fill="none"/>{arrowhead(x1, y1, x2, y2)}')
    if label:
        result += f'<text x="{(x1 + x2) / 2}" y="{min(y1, y2) - 10}" class="edge" text-anchor="middle">{escape(label)}</text>'
    return result


def path(d, label, x, y):
    points = [(int(px), int(py)) for px, py in re.findall(r"[ML](\d+) (\d+)", d)]
    x1, y1 = points[-2]
    x2, y2 = points[-1]
    return (f'<path d="{d}" stroke="#56716a" stroke-width="2.5" fill="none" '
            f'/>{arrowhead(x1, y1, x2, y2)}<text x="{x}" y="{y}" class="edge">{escape(label)}</text>')


def arrowhead(x1, y1, x2, y2):
    angle = atan2(y2 - y1, x2 - x1)
    side = 8
    spread = 0.55
    left = (x2 - side * cos(angle - spread), y2 - side * sin(angle - spread))
    right = (x2 - side * cos(angle + spread), y2 - side * sin(angle + spread))
    return (f'<polygon points="{x2},{y2} {left[0]:.1f},{left[1]:.1f} '
            f'{right[0]:.1f},{right[1]:.1f}" fill="#56716a"/>')


def figure(name, title, subtitle, parts, height=370):
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="960" height="{height}" viewBox="0 0 960 {height}" role="img" aria-labelledby="title desc">
<title id="title">{escape(title)}</title><desc id="desc">{escape(subtitle)}</desc>
<style>.heading{{font:700 25px -apple-system,BlinkMacSystemFont,'PingFang SC',sans-serif;fill:#17302d}}.sub{{font:14px -apple-system,BlinkMacSystemFont,'PingFang SC',sans-serif;fill:#5d756d}}.label{{font:700 18px -apple-system,BlinkMacSystemFont,'PingFang SC',sans-serif;fill:#17302d}}.detail{{font:13px -apple-system,BlinkMacSystemFont,'PingFang SC',sans-serif;fill:#48645b}}.edge{{font:12px -apple-system,BlinkMacSystemFont,'PingFang SC',sans-serif;fill:#536b63}}</style>
<rect width="960" height="{height}" rx="20" fill="#f9fcfa" stroke="#dce8e1"/>
<text x="36" y="48" class="heading">{escape(title)}</text><text x="36" y="75" class="sub">{escape(subtitle)}</text>
{''.join(parts)}
</svg>'''
    (OUT / f"{name}.svg").write_text(svg, encoding="utf-8")


figure("agent_stack", "Agent 学习地图", "从反馈控制出发，沿同一条任务链路逐层增强", [
    card(36, 125, 190, 100, "传统 Agent", "状态 · 策略 · 反馈"),
    card(268, 125, 190, 100, "LLM Agent", "推理 · 工具 · 循环", "blue"),
    card(500, 125, 190, 100, "系统能力", "检索 · 记忆 · 协作", "violet"),
    card(732, 125, 190, 100, "Harness 工程", "恢复 · 评测 · 安全", "orange"),
    arrow(226, 175, 260, 175), arrow(458, 175, 492, 175), arrow(690, 175, 724, 175),
    path("M830 235 L830 294 L128 294 L128 235", "用真实任务和失败反馈迭代", 364, 283),
])

figure("agent_loop", "观察—决策—行动闭环", "模型产生候选动作；运行时执行；环境返回可验证反馈", [
    card(40, 130, 190, 102, "环境与目标", "当前状态 / 用户目标", "blue"),
    card(275, 130, 190, 102, "观察与状态", "历史 / 检索 / 反馈"),
    card(510, 130, 190, 102, "策略与规划", "规则 / 搜索 / 模型", "violet"),
    card(745, 130, 175, 102, "执行动作", "工具 / API / 人", "orange"),
    arrow(230, 181, 267, 181), arrow(465, 181, 502, 181), arrow(700, 181, 737, 181),
    path("M832 241 L832 296 L140 296 L140 241", "环境转移与结果校验", 352, 285),
])

figure("tool_loop", "最小工具调用循环", "把工具调用、结果和停止条件放在同一条可审计轨迹", [
    card(37, 127, 190, 102, "目标与上下文", "用户请求 + 可用工具", "blue"),
    card(273, 127, 190, 102, "模型决策", "工具调用或最终回答", "violet"),
    card(509, 127, 190, 102, "运行时校验", "Schema / 权限 / 预算", "orange"),
    card(745, 127, 177, 102, "工具结果", "执行、错误与证据"),
    arrow(227, 178, 265, 178), arrow(463, 178, 501, 178), arrow(699, 178, 737, 178),
    path("M832 237 L832 296 L365 296 L365 237", "结果成为下一轮观察", 510, 285),
    path("M365 117 L365 101 L125 101 L125 117", "完成 / 达到上限 → 结束", 168, 95),
])

figure("rag_memory", "检索与记忆的数据流", "外部知识提供可引用证据；记忆只保存经过验证且可复用的信息", [
    card(36, 118, 190, 102, "文档库", "解析 · 切块 · 索引", "blue"),
    card(270, 118, 190, 102, "召回与重排", "关键词 + 向量 + 过滤"),
    card(504, 118, 190, 102, "带来源的证据", "片段 + 版本 + 权限", "violet"),
    card(738, 118, 184, 102, "生成与引用", "回答 + 来源检查", "orange"),
    arrow(226, 169, 262, 169), arrow(460, 169, 496, 169), arrow(694, 169, 730, 169),
    card(504, 270, 190, 72, "长期记忆", "验证后写入 / 可删除"),
    path("M829 228 L829 306 L702 306", "有条件写入", 718, 281),
    path("M496 306 L365 306 L365 228", "按需取回", 375, 295),
], 380)

figure("context_budget", "上下文预算如何分配", "固定目标，保留关键状态，原始资料按需读取", [
    card(38, 118, 205, 103, "始终保留", "目标 · 约束 · 权限"),
    card(265, 118, 205, 103, "短期工作区", "最近轨迹 · 工具结果", "blue"),
    card(492, 118, 205, 103, "结构化摘要", "已验证 / 待解决", "violet"),
    card(719, 118, 203, 103, "外部存储", "大文件 · 原始日志", "orange"),
    arrow(243, 169, 257, 169), arrow(470, 169, 484, 169), arrow(697, 169, 711, 169),
    path("M818 231 L818 292 L366 292 L366 231", "需要证据时按引用重新读取", 490, 280),
])

figure("multi_agent", "多 Agent 协作与验收", "委派明确子任务，保留独立证据，主 Agent 负责合并与验收", [
    card(36, 135, 190, 105, "用户目标", "范围 · 验收标准", "blue"),
    card(276, 135, 190, 105, "主 Agent", "拆分 · 分配 · 合并", "violet"),
    card(538, 105, 180, 80, "研究 Agent", "原始资料 + 链接"),
    card(538, 230, 180, 80, "实现 Agent", "代码 + 测试结果"),
    card(772, 135, 150, 105, "验收", "证据 + 结果", "orange"),
    arrow(226, 188, 268, 188),
    path("M466 166 L530 145", "委派", 477, 143),
    path("M466 213 L530 267", "委派", 478, 248),
    path("M718 145 L764 166", "产物", 723, 131),
    path("M718 270 L764 213", "产物", 724, 267),
], 365)

figure("agent_harness", "Agent Harness：从决策到可验收的行动", "运行时组织上下文与执行；持久化状态支撑恢复；完成由验收结果判定", [
    card(36, 118, 200, 96, "任务与上下文", "目标 · 约束 · 相关证据", "blue"),
    card(278, 118, 180, 96, "模型决策", "下一动作 / 完成提议", "violet"),
    card(500, 118, 180, 96, "受控执行", "权限 · 预算 · 工具"),
    card(722, 118, 202, 96, "环境与验收", "文件 · 界面 · 测试", "orange"),
    arrow(236, 166, 270, 166), arrow(458, 166, 492, 166), arrow(680, 166, 714, 166),
    path("M824 224 L824 264 L136 264 L136 224", "观察结果回填；验收未通过则继续或报告阻塞", 315, 252),
    card(278, 316, 402, 92, "持久化状态与恢复", "检查点 · 产物引用 · 调用记录 · 版本", "violet"),
    path("M590 224 L590 308", "保存", 600, 296),
    path("M278 362 L136 362 L136 282", "恢复时核对真实环境", 38, 393),
], 440)

figure("agent_products", "Agent 产品：入口不同，任务边界也不同", "概念图：仓库开发关注补丁闭环；持续助理还需要身份、路由、投递与跨会话知识", [
    card(36, 125, 190, 92, "仓库开发入口", "终端 / IDE / 桌面", "blue"),
    card(268, 125, 190, 92, "工作区与会话", "规则 · 文件 · 历史", "violet"),
    card(500, 125, 190, 92, "编程 Harness", "模型 · 权限 · 工具"),
    card(732, 125, 190, 92, "补丁与验收", "构建 · 测试 · 审查", "orange"),
    arrow(226, 171, 260, 171), arrow(458, 171, 492, 171), arrow(690, 171, 724, 171),
    path("M826 227 L826 258 L128 258 L128 227", "失败反馈推动下一步", 375, 248),
    card(36, 314, 190, 92, "持续任务入口", "消息 / 自动触发", "blue"),
    card(268, 314, 190, 92, "网关与路由", "身份 · 会话 · 队列", "violet"),
    card(500, 314, 190, 92, "所选 Runtime", "上下文 · 工具 · 状态"),
    card(732, 314, 190, 92, "产物与投递", "验收 · 终态 · 回复", "orange"),
    arrow(226, 360, 260, 360), arrow(458, 360, 492, 360), arrow(690, 360, 724, 360),
    path("M826 416 L826 457 L128 457 L128 416", "持久记忆 / Skills 供后续会话复用", 325, 446),
], 490)
