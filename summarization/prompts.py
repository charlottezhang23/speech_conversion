"""Prompt templates for the two summary modes (meeting / qualitative interview)."""
from __future__ import annotations

from typing import Any

MEETING_SECTIONS = [
    "会议信息",
    "会议主题",
    "讨论重点",
    "各参与者主要观点",
    "已确认的结论",
    "决策事项",
    "行动事项",
    "尚未解决的问题",
]

INTERVIEW_SECTIONS = [
    "访谈基本信息",
    "访谈核心摘要",
    "受访者背景信息",
    "使用场景与日常行为",
    "主要痛点和困难",
    "已使用的解决方案",
    "购买行为及决策因素",
    "产品偏好与未满足需求",
    "价格接受度",
    "重要原话及时间戳",
    "潜在需求假设",
    "有待进一步追问的问题",
]

TEMPLATES: dict[str, dict[str, str]] = {
    "meeting": {
        "label": "普通会议纪要",
        "description": "主题、讨论重点、各方观点、结论、决策、行动事项与未解决问题。",
        "sections": MEETING_SECTIONS,
    },
    "interview": {
        "label": "定性研究访谈纪要",
        "description": "背景、场景、痛点、替代方案、购买行为、偏好、价格、原话证据、需求假设与追问清单。",
        "sections": INTERVIEW_SECTIONS,
    },
}

RULES = """\
【绝对规则】
1. 只能使用对话中明确出现的内容。禁止推测、补全、润色成"听起来合理"的说法。
2. 每一个结论、痛点、偏好、数字（价格、频率、数量、期限）都必须附证据时间戳，格式 [HH:MM:SS]。
3. 原文没有提到的信息，写"原文未提及"，不要留空，也不要编造。
4. 保留受访者的原始措辞；引用必须逐字摘录，不得改写。
5. 日语等非中文原话保留原文，并在括号内附中文翻译；**中文原话直接引用，不要添加"（中文翻译）"之类的说明**。
6. 如果某处听不清或语义不完整，标注 [需核对]，不要猜测内容。
7. 不要输出没有证据支持的"洞察"。推理性的内容只能放在"潜在需求假设"一节，并标明依据。
8. 只输出要求的结构，不要输出解释性的开场白或结束语。
"""

EVIDENCE_FORMAT = """\
转录文本每行格式为：[时:分:秒] 说话人: 内容
例：[00:12:34] 受访者: 我平时用的是另一款，但导出很麻烦。
引用时间戳时必须使用行首的时间戳，不要自行推算。
"""

# --------------------------------------------------------------- map (per chunk)
MAP_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "chunk_summary": {"type": "string"},
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "type": {
                        "type": "string",
                        "enum": [
                            "topic",
                            "viewpoint",
                            "decision",
                            "action",
                            "question",
                            "background",
                            "scenario",
                            "pain_point",
                            "current_solution",
                            "buying_behavior",
                            "preference",
                            "unmet_need",
                            "price",
                            "quote",
                        ],
                    },
                    "statement": {"type": "string"},
                    "speaker": {"type": "string"},
                    "time": {"type": "string"},
                    "quote": {"type": "string"},
                    "quote_zh": {"type": "string"},
                    "owner": {"type": "string"},
                    "deadline": {"type": "string"},
                },
                "required": ["type", "statement", "time"],
            },
        },
    },
    "required": ["chunk_summary", "facts"],
}

MAP_SYSTEM = (
    "你是一名严谨的访谈/会议记录分析员。你的任务是抽取事实，而不是写作。\n" + RULES
)

MAP_USER = """\
以下是完整转录的一部分（第 {index}/{total} 段，{start} - {end}）。
请抽取这一段中所有可核查的事实，按 type 分类，并给出时间戳与原文引用。

{evidence_format}

【这一段的内容】
{transcript}

【输出要求】
严格输出 JSON，结构如下：
{{"chunk_summary": "本段 2-3 句客观概括",
  "facts": [
    {{"type": "pain_point", "statement": "受访者认为导出流程步骤太多",
      "speaker": "受访者", "time": "00:12:34",
      "quote": "每次导出一份录音要点五次，太麻烦", "quote_zh": ""}}
  ]}}
type 只能取：topic / viewpoint / decision / action / question / background / scenario /
pain_point / current_solution / buying_behavior / preference / unmet_need / price / quote。
quote 必须是逐字原文（可为空字符串）；非中文原话时把中文翻译放进 quote_zh。
如果这一段没有可提取的事实，返回空数组。

【硬性限制】
- 本段最多输出 30 条事实，只保留最关键的；同一件事不要拆成多条。
- statement 不超过 60 字，quote 不超过 80 字，不要复述整段原文。
- 不要输出任何解释文字，只输出 JSON。
"""

# ------------------------------------------------------------- reduce (merge)
REDUCE_SYSTEM = "你是一名访谈资料整理员，负责合并多段抽取结果，去重但不得添加新信息。\n" + RULES

REDUCE_USER = """\
下面是同一场访谈不同时间段抽取的事实（JSON 数组）。请合并去重：
- 合并重复的事实，保留最完整、时间戳最准确的一条；
- 保留所有不同的观点、数字与引用，不要丢弃少数意见；
- 不得新增任何原文没有的信息；
- speaker 使用原文中的说话人称呼。

【待合并事实】
{facts_json}

【输出要求】
严格输出 JSON：{{"chunk_summary": "整体客观概述", "facts": [ ... 与输入相同的结构 ... ]}}
"""

# ------------------------------------------------------------ final compose
FINAL_MEETING_USER = """\
根据下面的"已抽取事实"撰写中文会议纪要。所有内容必须有事实依据，不得编造。

【基本信息】
{basic_info}

【已抽取事实（已按类型标注时间戳）】
{facts_json}

【输出格式】
严格使用以下 Markdown 结构，不要增加其他一级标题：
# 会议纪要
## 1. 会议信息
（时间、时长、参与人 —— 只写已知信息）
## 2. 会议主题
## 3. 讨论重点
（分条，每条以 [HH:MM:SS] 结尾）
## 4. 各参与者主要观点
（按说话人分小节）
## 5. 已确认的结论
## 6. 决策事项
## 7. 行动事项
（表格：事项 | 负责人 | 期限；原文没有负责人或期限时写"原文未提及"）
## 8. 尚未解决的问题

写作要求：
- 每个条目结尾附证据时间戳，例如：导出流程繁琐 [00:12:34]
- 不要出现"可能""也许""推测"等词，除非在事实中明确是受访者的猜测
"""

FINAL_INTERVIEW_USER = """\
根据下面的"已抽取事实"撰写中文定性研究访谈纪要。所有内容必须有事实依据，不得编造受访者的观点、行为或数字。

【基本信息】
{basic_info}

【已抽取事实（已按类型标注时间戳）】
{facts_json}

【输出格式】
严格使用以下 Markdown 结构，不要增加其他一级标题：
# 访谈纪要
## 1. 访谈基本信息
## 2. 访谈核心摘要
（150-250 字，只写有证据的内容）
## 3. 受访者背景信息
## 4. 使用场景与日常行为
## 5. 主要痛点和困难
## 6. 已使用的解决方案
## 7. 购买行为及决策因素
## 8. 产品偏好与未满足需求
## 9. 价格接受度
（**如果"已抽取事实"里存在 type=price 的条目，必须写出这些价格信息并附时间戳；只有完全没有价格事实时，整节才写"原文未提及价格信息"**）
## 10. 重要原话及时间戳
（编号列表，格式：> "原话"（中文翻译）—— 受访者 [00:12:34]；日语等原话保留原文）
## 11. 潜在需求假设
（每条标注"依据：[HH:MM:SS]"，并说明这是研究者推论而非受访者原话）
## 12. 有待进一步追问的问题
（基于空白或矛盾点提出，每条说明为什么需要追问）

写作要求：
- 每个条目结尾附证据时间戳，例如：导出步骤太多 [00:12:34]
- 原文没有的信息写"原文未提及"，不要留空
- 不要编造价格数字、购买频率、品牌名称
"""

# ------------------------------------------------------------- verification
VERIFY_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "unsupported": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "reason": {"type": "string"},
                    "suggestion": {"type": "string"},
                },
                "required": ["claim", "reason"],
            },
        },
        "verdict": {"type": "string", "enum": ["ok", "minor_issues", "major_issues"]},
        "notes": {"type": "string"},
    },
    "required": ["unsupported", "verdict"],
}

VERIFY_SYSTEM = (
    "你是一名事实核查员。你只判断给定纪要中的陈述能否在给定证据中找到支持，不做改写。\n"
    "判断标准：证据中必须存在语义对应的原句或明确的时间戳记录；仅「听起来合理」不算支持。\n"
    "注意：语义等价的概述、同义改写、把口语整理成书面语，都算有依据，不要列为问题。"
)

VERIFY_USER = """\
【纪要】
{summary}

【证据：抽取事实（含时间戳与原文引用）】
{facts_json}

【任务】
只检查以下四类严重问题，其余都不要报告：
1. 编造：证据中完全没有对应内容（人、事、数字、品牌、决策）；
2. 数字/时间与实际证据不符；
3. 把"受访者可能想要"写成确定事实，或把个人观点写成所有人的共识；
4. 把原文的否定写成肯定（或相反）。

不要因为"措辞更概括""同义改写""合并了两句话"而列入问题。
输出 JSON：{{"verdict": "ok|minor_issues|major_issues", "unsupported": [{{"claim": "...", "reason": "...", "suggestion": "..."}}], "notes": "..."}}
没有问题就返回空数组并把 verdict 设为 ok。
"""


def basic_info_block(meta: dict[str, Any], names: dict[str, str] | None = None) -> str:
    from app.util import hhmmss

    parts = [
        f"- 项目名称：{meta.get('name', '未命名')}",
        f"- 录音时长：{hhmmss(meta.get('duration_sec'))}",
        f"- 原始文件：{meta.get('original_name') or '未知'}",
        f"- 识别语言设置：{meta.get('language') or 'auto'}",
    ]
    if meta.get("asr_model"):
        parts.append(f"- 识别模型：{meta['asr_model']}")
    if names:
        parts.append("- 说话人：" + "；".join(f"{k} = {v}" for k, v in names.items()))
    return "\n".join(parts)
